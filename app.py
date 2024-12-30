import os
import io
import uuid
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Heroku or any headless server
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template_string
from math import sqrt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

################################################################################
# 1) GLOBAL IN-MEMORY DATA STORAGE
#    (For real persistence, store this in MongoDB or a database instead.)
################################################################################
all_accel_data = []  # List of DataFrames (acceleration) from each upload
all_true_data  = []  # List of DataFrames (true velocity) from each upload

MODEL_FILENAME = "model.pkl"
saved_model    = None

# If a model was saved before, try loading it
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        print(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        print(f"Could not load model. Error: {str(e)}")
        saved_model = None

################################################################################
# 2) HELPER FUNCTIONS: Accel->Velocity, True Velocity Expansion, IoU, etc.
################################################################################

def remove_spikes_and_integrate(df,
                                time_col='time',
                                ax_col='ax (m/s^2)',
                                ay_col='ay (m/s^2)',
                                az_col='az (m/s^2)'):
    """
    1) Remove spikes using rolling mean ±3*std
    2) Integrate acceleration -> velocity
    """
    # Rolling means
    df['ax_mean'] = df[ax_col].rolling(window=5, center=True).mean()
    df['ay_mean'] = df[ay_col].rolling(window=5, center=True).mean()
    df['az_mean'] = df[az_col].rolling(window=5, center=True).mean()

    # Rolling std
    df['ax_std'] = df[ax_col].rolling(window=5, center=True).std().fillna(0)
    df['ay_std'] = df[ay_col].rolling(window=5, center=True).std().fillna(0)
    df['az_std'] = df[az_col].rolling(window=5, center=True).std().fillna(0)

    factor = 3
    for col, mcol, scol in zip([ax_col, ay_col, az_col],
                               ['ax_mean','ay_mean','az_mean'],
                               ['ax_std','ay_std','az_std']):
        df[col] = np.where(abs(df[col] - df[mcol]) > factor*df[scol],
                           df[mcol],
                           df[col])

    # Drop temp
    df.drop(columns=['ax_mean','ay_mean','az_mean','ax_std','ay_std','az_std'], inplace=True)

    # Integrate
    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]
    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        dt = df.loc[i, 'time_diff']

        # Dominant axis or total magnitude
        if abs(ax) > abs(ay) and abs(ax) > abs(az):
            accel = ax
        elif abs(ay) > abs(ax) and abs(ay) > abs(az):
            accel = ay
        elif abs(az) > abs(ax) and abs(az) > abs(ay):
            accel = az
        else:
            accel = sqrt(ax**2 + ay**2 + az**2)

        velocity.append(velocity[-1] + accel*dt)

    df['velocity'] = velocity
    return df

def expand_true_velocity(df_true, df_accel,
                         time_col='time',
                         speed_col='speed'):
    """
    Expand the true velocity so it matches # of rows in df_accel
    by randomizing speeds ±5% between each original point.
    """
    df_true = df_true[[time_col, speed_col]]
    n_acc   = len(df_accel)
    n_true  = len(df_true)
    if n_true == 0:
        raise ValueError("True velocity is empty; cannot expand.")

    ratio = n_acc / n_true
    ratio_minus_1 = int(np.floor(ratio - 1)) if ratio>1 else 0

    speeds = []
    for i in range(n_true):
        orig_spd = df_true[speed_col].iloc[i]
        speeds.append(orig_spd)
        for _ in range(ratio_minus_1):
            speeds.append(np.random.uniform(orig_spd*0.95, orig_spd*1.05))

    # Fill remainder
    remainder = n_acc - len(speeds)
    if remainder > 0:
        last_spd = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            speeds.append(np.random.uniform(last_spd*0.95, last_spd*1.05))

    speeds = speeds[:n_acc]
    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': speeds
    })
    return df_expanded

def compute_iou(true_vel, corrected_vel):
    min_vals = np.minimum(true_vel, corrected_vel)
    max_vals = np.maximum(true_vel, corrected_vel)
    return (min_vals.sum() / max_vals.sum()) * 100

################################################################################
# 3) TRAINING FUNCTION: Repeated training on all data until IoU≥95 or max loops
################################################################################

def train_on_all_data(max_loops=5):
    """
    Merges all old + new data from memory, trains once, and repeats if IoU<95%
    up to max_loops times. Returns the final model, DataFrame, metrics, etc.
    The final plot should only represent the *latest* dataset.
    """
    if len(all_accel_data)==0 or len(all_true_data)==0:
        raise ValueError("No data available for training. Please upload first.")

    # Merge everything
    full_accel_df = pd.concat(all_accel_data, ignore_index=True)
    full_true_df  = pd.concat(all_true_data,  ignore_index=True)

    # We'll do repeated training loops
    best_model   = None
    best_metrics = {}
    iou_acc      = 0

    for loop_i in range(max_loops):
        # Step A: Preprocess
        accel_df_proc = remove_spikes_and_integrate(full_accel_df.copy())
        true_df_expanded = expand_true_velocity(full_true_df.copy(), accel_df_proc)

        # Build combined DF
        df = pd.DataFrame()
        df['time']          = accel_df_proc['time']
        df['velocity']      = accel_df_proc['velocity']
        df['true_velocity'] = true_df_expanded['true_velocity']
        df['correction']    = df['true_velocity'] - df['velocity']

        # Step B: Train
        X = df[['time','velocity']].values
        y = df['correction'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_test_pred = model.predict(X_test)
        mae_test  = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Predict on full
        df['predicted_correction'] = model.predict(X)
        df['corrected_velocity']   = df['velocity'] + df['predicted_correction']

        mae_corr = mean_absolute_error(df['true_velocity'], df['corrected_velocity'])
        rmse_corr= np.sqrt(mean_squared_error(df['true_velocity'], df['corrected_velocity']))
        iou_acc  = compute_iou(df['true_velocity'], df['corrected_velocity'])

        # If IoU≥95%, we can break early
        if iou_acc >= 95.0:
            best_model = model
            best_metrics = {
                "df": df,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "mae_corr": mae_corr,
                "rmse_corr": rmse_corr,
                "iou_acc":  iou_acc
            }
            break
        else:
            # Otherwise keep going until max_loops
            best_model = model
            best_metrics = {
                "df": df,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "mae_corr": mae_corr,
                "rmse_corr": rmse_corr,
                "iou_acc":  iou_acc
            }
            # no break, keep looping

    return best_model, best_metrics

################################################################################
# 4) ROUTES
################################################################################

@app.route('/process', methods=['POST'])
def process_endpoint():
    """
    1) Upload acceleration + true velocity
    2) Insert into global in-memory lists
    3) Retrain on *all* data (old + new) with repeated loops if needed
    4) Save model if IoU≥95
    5) Plot only the *new* dataset
    6) Return acknowledgment & stats
    """
    global saved_model
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error": "Need both acceleration_file and true_velocity_file"}), 400

    accel_file = request.files['acceleration_file']
    true_file  = request.files['true_velocity_file']
    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1) Read the CSVs
        accel_df = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        true_df  = pd.read_csv(io.StringIO(true_file.read().decode("utf-8")),  low_memory=False)

        # 2) Lowercase columns
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns  = true_df.columns.str.lower()

        # 3) Check required columns
        accel_req = ['ax (m/s^2)','ay (m/s^2)','az (m/s^2)','time']
        missing_accel = [c for c in accel_req if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration: {missing_accel}"}), 400

        true_req = ['time','speed']
        missing_true = [c for c in true_req if c not in true_df.columns]
        if missing_true:
            return jsonify({"error": f"Missing columns in true velocity: {missing_true}"}), 400

        # 4) Append to global memory
        all_accel_data.append(accel_df)
        all_true_data.append(true_df)

        # 5) Train on all data
        model, metrics = train_on_all_data(max_loops=5)
        df        = metrics["df"]
        iou       = metrics["iou_acc"]
        mae_test  = metrics["mae_test"]
        rmse_test = metrics["rmse_test"]
        mae_corr  = metrics["mae_corr"]
        rmse_corr = metrics["rmse_corr"]

        # 6) If IoU≥95, save
        if iou >= 95.0:
            joblib.dump(model, MODEL_FILENAME)
            saved_model = model
            print("Model saved (IoU≥95%).")

        # 7) We only want to plot the NEW dataset, not old ones
        #    So let's remove spikes & integrate JUST the newly uploaded acceleration,
        #    expand the newly uploaded true velocity, and generate a fresh df_new
        new_accel_proc  = remove_spikes_and_integrate(accel_df.copy())
        new_true_exp    = expand_true_velocity(true_df.copy(), new_accel_proc)

        df_new = pd.DataFrame()
        df_new['time']          = new_accel_proc['time']
        df_new['velocity']      = new_accel_proc['velocity']
        df_new['true_velocity'] = new_true_exp['true_velocity']
        df_new['correction']    = df_new['true_velocity'] - df_new['velocity']

        # Predict corrections using the *final* model we ended up with
        X_new = df_new[['time','velocity']].values
        df_new['predicted_correction'] = model.predict(X_new)
        df_new['corrected_velocity']   = df_new['velocity'] + df_new['predicted_correction']

        # 8) Create the plot for the NEW dataset only
        plt.figure(figsize=(10,6))
        plt.plot(df_new['time'], df_new['true_velocity'], label='True Velocity', linestyle='--')
        plt.plot(df_new['time'], df_new['velocity'],      label='Calculated Velocity')
        plt.plot(df_new['time'], df_new['corrected_velocity'], label='Corrected Velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Latest Training - New Dataset Only')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 9) Build final response
        #    Acknowledge how many total times we've trained
        total_datasets = len(all_accel_data)  # or all_true_data
        ack_msg = f"Trained on {total_datasets} total dataset(s). IoU final: {iou:.2f}%."

        resp = {
            "acknowledgment": ack_msg,
            "average_velocities_on_test_dataset": {
                "Average_Corrected_Velocity": df[df['time'].isin(X_new[:,0])]['corrected_velocity'].mean(),  
                "Average_True_Velocity":     df[df['time'].isin(X_new[:,0])]['true_velocity'].mean()
            },
            "model_evaluation": {
                "Test_Set_MAE": mae_test,
                "Test_Set_RMSE": rmse_test,
                "Corrected_Velocity_MAE": mae_corr,
                "Corrected_Velocity_RMSE": rmse_corr,
                "IoU_Accuracy": iou
            },
            "plot_image_base64": image_b64
        }
        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Only acceleration_file is required.
    We use the *latest saved model* to predict the average corrected velocity.
    """
    global saved_model
    if 'acceleration_file' not in request.files:
        return jsonify({"error":"Please provide acceleration_file"}), 400

    accel_file = request.files['acceleration_file']
    if accel_file.filename == '':
        return jsonify({"error":"No selected file"}), 400

    try:
        # If we never loaded or saved a model yet:
        if not saved_model and os.path.exists(MODEL_FILENAME):
            saved_model = joblib.load(MODEL_FILENAME)
        if not saved_model:
            return jsonify({"error":"No saved model found. Please train first."}), 400

        df_accel = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        df_accel.columns = df_accel.columns.str.lower()

        accel_req = ['ax (m/s^2)','ay (m/s^2)','az (m/s^2)','time']
        missing_accel = [c for c in accel_req if c not in df_accel.columns]
        if missing_accel:
            return jsonify({"error":f"Missing columns in acceleration: {missing_accel}"}), 400

        # Preprocess
        df_accel = remove_spikes_and_integrate(df_accel)
        X_acc = df_accel[['time','velocity']].values

        # Predict
        predicted_corr = saved_model.predict(X_acc)
        corrected_vel  = df_accel['velocity'] + predicted_corr

        avg_corrected_vel = float(corrected_vel.mean())

        return jsonify({
            "message": "Predicted corrected velocity using the saved model.",
            "average_corrected_velocity": avg_corrected_vel
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


################################################################################
# 5) Basic HTML with Two Forms (Train & Predict)
################################################################################

@app.route('/upload', methods=['GET'])
def upload_page():
    """
    Single page with:
      - Train form (accel + true)
      - Predict form (accel only)
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Velocity Processing (No Overlap, Repeated Training)</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        body { margin: 40px; }
        .results { margin-top: 20px; border: 1px solid #ccc; padding: 15px; }
        .plot-img { max-width: 100%; margin-top: 10px; border: 1px solid #ccc; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1 class="text-primary mb-4">Velocity Processing with Repeated Training & No Overlapping Plots</h1>
        <p>Each time you upload new data, the model accumulates old+new. The final plot shows only the new dataset,
           but the model is stronger because it has learned from all previous data.</p>

        <div class="row">
          <!-- Train Form -->
          <div class="col-md-6">
            <div class="card p-3 mb-4">
              <h3>Train</h3>
              <form id="trainForm" method="POST" enctype="multipart/form-data">
                <div class="mb-3">
                  <label>Acceleration CSV</label>
                  <input type="file" name="acceleration_file" class="form-control" required />
                </div>
                <div class="mb-3">
                  <label>True Velocity CSV</label>
                  <input type="file" name="true_velocity_file" class="form-control" required />
                </div>
                <button type="submit" class="btn btn-primary">Train</button>
              </form>
              <div id="trainResults" class="results" style="display:none;"></div>
            </div>
          </div>

          <!-- Predict Form -->
          <div class="col-md-6">
            <div class="card p-3 mb-4">
              <h3>Predict</h3>
              <form id="predictForm" method="POST" enctype="multipart/form-data">
                <div class="mb-3">
                  <label>Acceleration CSV</label>
                  <input type="file" name="acceleration_file" class="form-control" required />
                </div>
                <button type="submit" class="btn btn-success">Predict</button>
              </form>
              <div id="predictResults" class="results" style="display:none;"></div>
            </div>
          </div>
        </div>
      </div>

      <script>
      const trainForm = document.getElementById('trainForm');
      const trainResultsDiv = document.getElementById('trainResults');
      trainForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        trainResultsDiv.style.display = 'block';
        trainResultsDiv.innerHTML = '<b>Training... please wait.</b>';

        const formData = new FormData(trainForm);
        const response = await fetch('/process', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (!response.ok && data.error) {
          trainResultsDiv.innerHTML = '<div class="text-danger">Error: ' + data.error + '</div>';
        } else {
          let html = '';
          if (data.acknowledgment) {
            html += '<h4>' + data.acknowledgment + '</h4>';
          }
          if (data.average_velocities_on_test_dataset) {
            const avgData = data.average_velocities_on_test_dataset;
            html += '<h5>Average Velocities on the New Dataset</h5>';
            html += '<p>Average Corrected Velocity: ' + avgData.Average_Corrected_Velocity.toFixed(3) + '</p>';
            html += '<p>Average True Velocity: ' + avgData.Average_True_Velocity.toFixed(3) + '</p>';
          }
          if (data.model_evaluation) {
            const evalData = data.model_evaluation;
            html += '<h5>Model Evaluation</h5>';
            html += '<p>Test Set MAE: ' + evalData.Test_Set_MAE.toFixed(3) + '</p>';
            html += '<p>Test Set RMSE: ' + evalData.Test_Set_RMSE.toFixed(3) + '</p>';
            html += '<p>Corrected Velocity MAE: ' + evalData.Corrected_Velocity_MAE.toFixed(3) + '</p>';
            html += '<p>Corrected Velocity RMSE: ' + evalData.Corrected_Velocity_RMSE.toFixed(3) + '</p>';
            html += '<p>IoU Accuracy: ' + evalData.IoU_Accuracy.toFixed(2) + '%</p>';
          }
          if (data.plot_image_base64) {
            html += '<h5>Plot (New Dataset Only):</h5>';
            html += '<img class="plot-img" src="data:image/png;base64,' + data.plot_image_base64 + '"/>';
          }
          trainResultsDiv.innerHTML = html;
        }
      });

      const predictForm = document.getElementById('predictForm');
      const predictResultsDiv = document.getElementById('predictResults');
      predictForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        predictResultsDiv.style.display = 'block';
        predictResultsDiv.innerHTML = '<b>Predicting... please wait.</b>';

        const formData = new FormData(predictForm);
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (!response.ok && data.error) {
          predictResultsDiv.innerHTML = '<div class="text-danger">Error: ' + data.error + '</div>';
        } else {
          let html = '<h5>Prediction Result</h5>';
          html += '<p>Average Corrected Velocity: ' + data.average_corrected_velocity.toFixed(3) + '</p>';
          predictResultsDiv.innerHTML = html;
        }
      });
      </script>
    </body>
    </html>
    """
    return render_template_string(html_content)


@app.route('/', methods=['GET'])
def index():
    return """
    <h1>Welcome to Velocity Processing!</h1>
    <p><a href="/upload">Go to Upload Page</a></p>
    """

if __name__ == '__main__':
    app.run(debug=True)
