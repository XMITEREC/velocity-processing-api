import os
import io
import uuid
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Heroku or headless
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
#    (In production, store in MongoDB or a real database to persist across restarts.)
################################################################################
all_accel_data = []  # Each new upload's acceleration DataFrame
all_true_data  = []  # Each new upload's true velocity DataFrame

MODEL_FILENAME = "model.pkl"
saved_model    = None

# If a model was saved previously, try loading it
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        print(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        print(f"Could not load model. Error: {str(e)}")
        saved_model = None

################################################################################
# 2) HELPER FUNCTIONS: Spike Removal, Integration, True Vel Expansion, etc.
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
    df['ax_mean'] = df[ax_col].rolling(5, center=True).mean()
    df['ay_mean'] = df[ay_col].rolling(5, center=True).mean()
    df['az_mean'] = df[az_col].rolling(5, center=True).mean()

    df['ax_std']  = df[ax_col].rolling(5, center=True).std().fillna(0)
    df['ay_std']  = df[ay_col].rolling(5, center=True).std().fillna(0)
    df['az_std']  = df[az_col].rolling(5, center=True).std().fillna(0)

    factor = 3
    for col, mcol, scol in zip([ax_col, ay_col, az_col],
                               ['ax_mean','ay_mean','az_mean'],
                               ['ax_std','ay_std','az_std']):
        df[col] = np.where(abs(df[col] - df[mcol]) > factor*df[scol],
                           df[mcol],
                           df[col])

    df.drop(columns=['ax_mean','ay_mean','az_mean','ax_std','ay_std','az_std'], inplace=True)

    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]
    for i in range(1,len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        dt         = df.loc[i, 'time_diff']

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
    Expand true velocity to match the # of rows in df_accel,
    randomizing speeds ±5%.
    """
    df_true = df_true[[time_col, speed_col]]
    n_acc   = len(df_accel)
    n_true  = len(df_true)
    if n_true==0:
        raise ValueError("True velocity is empty, cannot expand.")

    ratio = n_acc / n_true
    ratio_minus_1 = int(np.floor(ratio - 1)) if ratio>1 else 0

    speeds = []
    for i in range(n_true):
        orig_spd = df_true[speed_col].iloc[i]
        speeds.append(orig_spd)
        for _ in range(ratio_minus_1):
            speeds.append(np.random.uniform(orig_spd*0.95, orig_spd*1.05))

    remainder = n_acc - len(speeds)
    if remainder>0:
        last_spd = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            speeds.append(np.random.uniform(last_spd*0.95, last_spd*1.05))

    speeds = speeds[:n_acc]
    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': speeds
    })
    return df_expanded

def compute_iou(true_v, corrected_v):
    mins = np.minimum(true_v, corrected_v)
    maxs = np.maximum(true_v, corrected_v)
    return (mins.sum() / maxs.sum()) * 100

################################################################################
# 3) TRAIN ON ALL DATA (Multiple Loops), Keep the Best IoU
################################################################################

def train_on_all_data(max_loops=5):
    """
    Combine all old + new data, do repeated training up to `max_loops` times.
    We do NOT stop if IoU≥95%; we keep going to find an even better IoU.
    We store the best IoU model overall and return it.
    """
    if len(all_accel_data)==0 or len(all_true_data)==0:
        raise ValueError("No data to train on. Please upload first.")

    # Merge all data
    accel_all = pd.concat(all_accel_data, ignore_index=True)
    true_all  = pd.concat(all_true_data,  ignore_index=True)

    best_model   = None
    best_metrics = {}
    best_iou     = 0.0

    for loop_i in range(max_loops):
        # Preprocess entire dataset
        accel_proc = remove_spikes_and_integrate(accel_all.copy())
        true_exp   = expand_true_velocity(true_all.copy(), accel_proc)

        # Build DF
        df = pd.DataFrame()
        df['time']          = accel_proc['time']
        df['velocity']      = accel_proc['velocity']
        df['true_velocity'] = true_exp['true_velocity']
        df['correction']    = df['true_velocity'] - df['velocity']

        # Train/test split
        X = df[['time','velocity']].values
        y = df['correction'].values
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_test_pred = model.predict(X_test)
        mae_test  = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Full data predictions
        df['predicted_correction'] = model.predict(X)
        df['corrected_velocity']   = df['velocity'] + df['predicted_correction']
        mae_corr = mean_absolute_error(df['true_velocity'], df['corrected_velocity'])
        rmse_corr= np.sqrt(mean_squared_error(df['true_velocity'], df['corrected_velocity']))
        iou_acc  = compute_iou(df['true_velocity'], df['corrected_velocity'])

        # If it's better than our best so far, store
        if iou_acc > best_iou:
            best_iou = iou_acc
            best_model = model
            best_metrics = {
                "df": df,
                "mae_test":  mae_test,
                "rmse_test": rmse_test,
                "mae_corr":  mae_corr,
                "rmse_corr": rmse_corr,
                "iou_acc":   iou_acc
            }
        # We do NOT break if iou_acc≥95, so we keep looping to find an even stronger model

    return best_model, best_metrics

################################################################################
# 4) ROUTES
################################################################################

@app.route('/process', methods=['POST'])
def process_endpoint():
    global saved_model
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error":"Need both acceleration_file & true_velocity_file"}), 400

    accel_file = request.files['acceleration_file']
    true_file  = request.files['true_velocity_file']
    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error":"No selected file"}), 400

    try:
        # 1) Read CSVs
        accel_df = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        true_df  = pd.read_csv(io.StringIO(true_file.read().decode("utf-8")),  low_memory=False)

        # 2) Lowercase columns
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns  = true_df.columns.str.lower()

        # 3) Check columns
        required_accel = ['ax (m/s^2)','ay (m/s^2)','az (m/s^2)','time']
        missing_a = [c for c in required_accel if c not in accel_df.columns]
        if missing_a:
            return jsonify({"error":f"Missing columns in acceleration: {missing_a}"}), 400

        required_true = ['time','speed']
        missing_t = [c for c in required_true if c not in true_df.columns]
        if missing_t:
            return jsonify({"error":f"Missing columns in true velocity: {missing_t}"}), 400

        # 4) Append to global memory
        all_accel_data.append(accel_df)
        all_true_data.append(true_df)

        # 5) Train on all data, multiple loops, no early stop
        model, metrics = train_on_all_data(max_loops=5)
        iou    = metrics['iou_acc']
        df     = metrics['df']
        mae_t  = metrics['mae_test']
        rmse_t = metrics['rmse_test']
        mae_c  = metrics['mae_corr']
        rmse_c = metrics['rmse_corr']

        # 6) If best IoU≥95, save
        if iou>=95.0:
            joblib.dump(model, MODEL_FILENAME)
            saved_model = model
            print("Model saved (IoU≥95).")

        # 7) Plot only the *new* dataset
        new_accel_proc  = remove_spikes_and_integrate(accel_df.copy())
        new_true_exp    = expand_true_velocity(true_df.copy(), new_accel_proc)
        df_new = pd.DataFrame()
        df_new['time']          = new_accel_proc['time']
        df_new['velocity']      = new_accel_proc['velocity']
        df_new['true_velocity'] = new_true_exp['true_velocity']
        df_new['correction']    = df_new['true_velocity'] - df_new['velocity']

        # Predict for the new data
        X_new = df_new[['time','velocity']].values
        df_new['predicted_correction'] = model.predict(X_new)
        df_new['corrected_velocity']   = df_new['velocity'] + df_new['predicted_correction']

        # 8) Compute Averages & Difference for the new dataset
        avg_corrected_vel = df_new['corrected_velocity'].mean()
        avg_true_vel      = df_new['true_velocity'].mean()
        difference        = abs(avg_corrected_vel - avg_true_vel)

        # 9) Plot
        plt.figure(figsize=(10,6))
        plt.plot(df_new['time'], df_new['true_velocity'], label='True Velocity', linestyle='--')
        plt.plot(df_new['time'], df_new['velocity'],      label='Calculated Velocity')
        plt.plot(df_new['time'], df_new['corrected_velocity'], label='Corrected Velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Training on New Dataset Only')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 10) Build JSON
        total_sets = len(all_accel_data)
        ack_msg    = f"Trained on {total_sets} total dataset(s). Final IoU: {iou:.4f}%."

        resp = {
            "acknowledgment": ack_msg,
            "average_velocities_on_test_dataset": {
                "Average_Corrected_Velocity": float(avg_corrected_vel),
                "Average_True_Velocity": float(avg_true_vel),
                "Difference_Corrected_vs_True": float(difference)
            },
            "model_evaluation": {
                "Test_Set_MAE": mae_t,
                "Test_Set_RMSE": rmse_t,
                "Corrected_Velocity_MAE": mae_c,
                "Corrected_Velocity_RMSE": rmse_c,
                "IoU_Accuracy": iou
            },
            "plot_image_base64": plot_b64
        }
        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Only acceleration_file is required.
    We use the final "strongest" model saved (the best IoU from any training).
    """
    global saved_model
    if 'acceleration_file' not in request.files:
        return jsonify({"error": "Please provide acceleration_file"}), 400

    accel_file = request.files['acceleration_file']
    if accel_file.filename=='':
        return jsonify({"error": "No selected file"}), 400

    try:
        # If no saved model in memory, try loading from disk
        if not saved_model and os.path.exists(MODEL_FILENAME):
            saved_model = joblib.load(MODEL_FILENAME)
        if not saved_model:
            return jsonify({"error":"No saved model found. Please train first."}), 400

        df_accel = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        df_accel.columns = df_accel.columns.str.lower()

        accel_req = ['ax (m/s^2)','ay (m/s^2)','az (m/s^2)','time']
        missing_a = [c for c in accel_req if c not in df_accel.columns]
        if missing_a:
            return jsonify({"error":f"Missing columns in acceleration: {missing_a}"}),400

        # Preprocess
        df_accel = remove_spikes_and_integrate(df_accel)
        X_acc = df_accel[['time','velocity']].values

        # Predict
        corrected_corr = saved_model.predict(X_acc)
        corrected_vel  = df_accel['velocity'] + corrected_corr
        avg_corrected  = float(corrected_vel.mean())

        return jsonify({
            "message":"Predicted corrected velocity using the strongest saved model.",
            "average_corrected_velocity": avg_corrected
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


################################################################################
# 5) Simple HTML Page with Train & Predict Forms
################################################################################

@app.route('/upload', methods=['GET'])
def upload_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Velocity Processing (Strong Model, No Early Stop @ 95%)</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        body { margin: 40px; }
        .results { margin-top: 20px; border: 1px solid #ccc; padding: 15px; }
        .plot-img { max-width: 100%; margin-top: 10px; border: 1px solid #ccc; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1 class="text-primary mb-4">Velocity Processing (No Early Stop at 95%)</h1>
        <p>This app accumulates old+new data, trains multiple times (max 5 loops) without stopping if IoU≥95, 
           and picks the strongest model found. The final graph only shows the new dataset.
           For prediction, only acceleration data is required, and it uses the final strongest model.</p>

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
            html += '<h5>Average Velocities on This New Dataset</h5>';
            html += '<p>Average Corrected Velocity: ' + avgData.Average_Corrected_Velocity.toFixed(3) + '</p>';
            html += '<p>Average True Velocity: ' + avgData.Average_True_Velocity.toFixed(3) + '</p>';
            html += `<p>Difference (Corrected vs True): ${avgData.Difference_Corrected_vs_True.toFixed(3)}</p>`;
          }
          if (data.model_evaluation) {
            const evalData = data.model_evaluation;
            html += '<h5>Model Evaluation (Best in Final Loop)</h5>';
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
