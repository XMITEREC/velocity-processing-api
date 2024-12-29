import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Heroku
import matplotlib.pyplot as plt
import io
import base64
import os
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import joblib  # For saving/loading model

app = Flask(__name__)

MODEL_FILENAME = "model.pkl"

# ========================================
# 1) Try loading a previously saved model
# ========================================
saved_model = None
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        print(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        print(f"Could not load model from {MODEL_FILENAME}. Error: {str(e)}")
        saved_model = None


# ========================================
# A) HELPER FUNCTIONS
# ========================================
def preprocess_acceleration_to_velocity(df, time_col='time', ax_col='ax (m/s^2)', ay_col='ay (m/s^2)', az_col='az (m/s^2)'):
    df['ax_rolling_mean'] = df[ax_col].rolling(window=5, center=True).mean()
    df['ay_rolling_mean'] = df[ay_col].rolling(window=5, center=True).mean()
    df['az_rolling_mean'] = df[az_col].rolling(window=5, center=True).mean()

    df['ax_rolling_std'] = df[ax_col].rolling(window=5, center=True).std().fillna(0)
    df['ay_rolling_std'] = df[ay_col].rolling(window=5, center=True).std().fillna(0)
    df['az_rolling_std'] = df[az_col].rolling(window=5, center=True).std().fillna(0)

    std_multiplier = 3
    for col, mean_col, std_col in zip([ax_col, ay_col, az_col],
                                      ['ax_rolling_mean', 'ay_rolling_mean', 'az_rolling_mean'],
                                      ['ax_rolling_std', 'ay_rolling_std', 'az_rolling_std']):
        df[col] = np.where(
            abs(df[col] - df[mean_col]) > std_multiplier * df[std_col],
            df[mean_col],
            df[col]
        )

    df.drop(columns=['ax_rolling_mean', 'ay_rolling_mean', 'az_rolling_mean',
                     'ax_rolling_std', 'ay_rolling_std', 'az_rolling_std'], inplace=True)

    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]
    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        time_diff = df.loc[i, 'time_diff']

        # Choose "dominant axis" or combined magnitude
        if abs(ax) > abs(ay) and abs(ax) > abs(az):
            accel = ax
        elif abs(ay) > abs(ax) and abs(ay) > abs(az):
            accel = ay
        elif abs(az) > abs(ax) and abs(az) > abs(ay):
            accel = az
        else:
            accel = sqrt(ax**2 + ay**2 + az**2)

        velocity.append(velocity[-1] + accel * time_diff)

    df['velocity'] = velocity
    return df


def preprocess_true_velocity(df_true, df_accel, time_col='time', speed_col='speed'):
    df_true = df_true[[time_col, speed_col]]
    n1 = len(df_accel)
    n2 = len(df_true)
    if n2 == 0:
        raise ValueError("True velocity dataset has 0 rows. Cannot expand.")

    ratio = n1 / n2
    ratio_minus_1_int = int(np.floor(ratio - 1)) if ratio > 1 else 0

    expanded_speeds = []
    for i in range(n2):
        original_speed = df_true[speed_col].iloc[i]
        expanded_speeds.append(original_speed)
        for _ in range(ratio_minus_1_int):
            low_val = original_speed * 0.95
            high_val = original_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    # Handle remainder
    current_length = len(expanded_speeds)
    remainder = n1 - current_length
    if remainder > 0:
        last_speed = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            low_val = last_speed * 0.95
            high_val = last_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    expanded_speeds = expanded_speeds[:n1]

    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded


# ========================================
# B) TRAINING / RETRAINING FUNCTION
# ========================================
def process_data(accel_df, true_df):
    # 1) Integrate acceleration
    accel_df = preprocess_acceleration_to_velocity(
        accel_df, time_col='time'
    )

    # 2) Expand true velocity
    true_df_expanded = preprocess_true_velocity(
        df_true=true_df, df_accel=accel_df, time_col='time'
    )

    # 3) Create combined DataFrame
    time_col = 'time'
    calc_v_col = 'velocity'
    true_v_col = 'true_velocity'
    df = pd.DataFrame()
    df[time_col] = accel_df[time_col]
    df[calc_v_col] = accel_df[calc_v_col]
    df[true_v_col] = true_df_expanded[true_v_col]
    df['correction'] = df[true_v_col] - df[calc_v_col]

    # 4) Train model
    X = df[[time_col, calc_v_col]].values
    y = df['correction'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5) Evaluate on test set
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # 6) Predict corrections for full data
    df['predicted_correction'] = model.predict(X)
    df['corrected_velocity'] = df[calc_v_col] + df['predicted_correction']

    # 7) Evaluate corrected velocity
    mae_corrected = mean_absolute_error(df[true_v_col], df['corrected_velocity'])
    rmse_corrected = np.sqrt(mean_squared_error(df[true_v_col], df['corrected_velocity']))

    # 8) IoU Accuracy
    min_values = np.minimum(df[true_v_col], df['corrected_velocity'])
    max_values = np.maximum(df[true_v_col], df['corrected_velocity'])
    iou_accuracy = (np.sum(min_values) / np.sum(max_values)) * 100
    print("\n=== Model Evaluation ===")
    print(f"IoU Accuracy: {iou_accuracy:.4f}%")

    # 9) If IoU≥95%, save model
    if iou_accuracy >= 95.0:
        joblib.dump(model, MODEL_FILENAME)
        print(f"Model saved to {MODEL_FILENAME} because IoU ≥ 95%.")

    # 10) Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[time_col], df[true_v_col], label='True Velocity', linestyle='--')
    plt.plot(df[time_col], df[calc_v_col], label='Calculated Velocity')
    plt.plot(df[time_col], df['corrected_velocity'], label='Corrected Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity Comparison')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # 11) Test-subset average velocities
    test_times = X_test[:, 0]
    test_df = df[df[time_col].isin(test_times)]
    avg_corrected = test_df['corrected_velocity'].mean()
    avg_true = test_df[true_v_col].mean()
    diff_corr_true = abs(avg_corrected - avg_true)

    results = {
        "average_velocities_on_test_dataset": {
            "Average_Corrected_Velocity": avg_corrected,
            "Average_True_Velocity": avg_true,
            "Difference_Corrected_vs_True": diff_corr_true
        },
        "model_evaluation": {
            "Corrected_Velocity_MAE": mae_corrected,
            "Corrected_Velocity_RMSE": rmse_corrected,
            "Test_Set_MAE": mae_test,
            "Test_Set_RMSE": rmse_test,
            "IoU_Accuracy": iou_accuracy
        },
        "plot_image_base64": image_base64
    }
    return results


# ========================================
# C) INFERENCE-ONLY FUNCTION
# ========================================
def predict_corrected_velocity_only(accel_df, loaded_model):
    # 1) Integrate acceleration
    accel_df = preprocess_acceleration_to_velocity(accel_df)

    # 2) Use model to predict correction
    time_col = 'time'
    calc_v_col = 'velocity'
    X = accel_df[[time_col, calc_v_col]].values
    predicted_correction = loaded_model.predict(X)
    corrected_velocity = accel_df[calc_v_col] + predicted_correction

    # Return as DataFrame so we can convert to JSON or display
    results_df = pd.DataFrame({
        'time': accel_df[time_col],
        'calculated_velocity': accel_df[calc_v_col],
        'predicted_correction': predicted_correction,
        'corrected_velocity': corrected_velocity
    })
    return results_df


# ========================================
# D) ROUTES
# ========================================

@app.route('/process', methods=['POST'])
def process_endpoint():
    """
    1) Upload acceleration_file + true_velocity_file
    2) Train model, compute IoU, optionally save if >=95%
    3) Return JSON with metrics + base64 plot
    """
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error": "Please provide both 'acceleration_file' and 'true_velocity_file'"}), 400

    accel_file = request.files['acceleration_file']
    true_file = request.files['true_velocity_file']

    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        true_df = pd.read_csv(io.StringIO(true_file.stream.read().decode("UTF8")), low_memory=False)

        # Make columns lowercase
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns = true_df.columns.str.lower()

        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400

        required_true = ['time', 'speed']
        missing_true = [c for c in required_true if c not in true_df.columns]
        if missing_true:
            return jsonify({"error": f"Missing columns in true velocity dataset: {missing_true}"}), 400

        results = process_data(accel_df, true_df)
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    1) Upload only acceleration_file
    2) Use previously saved model (if exists) to predict corrected velocity
    3) Return the predictions as JSON (no plot by default)
    """
    global saved_model
    if 'acceleration_file' not in request.files:
        return jsonify({"error": "Please provide 'acceleration_file'"}), 400

    accel_file = request.files['acceleration_file']
    if accel_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # If not loaded yet, attempt to load from disk
        if not saved_model and os.path.exists(MODEL_FILENAME):
            saved_model = joblib.load(MODEL_FILENAME)

        if not saved_model:
            return jsonify({"error": "No saved model found. Please train first (IoU≥95%)."}), 400

        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        accel_df.columns = accel_df.columns.str.lower()

        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400

        results_df = predict_corrected_velocity_only(accel_df, saved_model)
        predictions_json = results_df.to_dict(orient='records')

        return jsonify({
            "message": "Predicted corrected velocity using saved model.",
            "predictions": predictions_json
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['GET'])
def upload_page():
    """
    A single page with two forms:
      1) Train (POST /process) -> shows metrics + base64 plot inline
      2) Predict (POST /predict) -> shows only JSON predictions
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Velocity Processing & Permanent Model</title>
        <!-- Bootstrap 5 CSS (CDN) -->
        <link 
            rel="stylesheet" 
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        >
        <style>
            body {
                margin-top: 40px;
                margin-bottom: 40px;
            }
            .plot-img {
                max-width: 100%;
                border: 1px solid #dee2e6;
                margin-top: 10px;
            }
            .results-block {
                margin-top: 1rem;
                padding: 1rem;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-primary mb-4">Velocity Processing & Permanent Model</h1>
            <p>1) Train with acceleration & true velocity (<code>/process</code>).
               If IoU &ge; 95%, model is saved. 
               Then you can do inference with only acceleration (<code>/predict</code>).
            </p>
            <div class="alert alert-info">
              <strong>Note:</strong> Heroku’s filesystem is ephemeral. The saved model may vanish upon dyno restart.
            </div>

            <div class="row">
                <!-- Training Form -->
                <div class="col-md-6">
                    <div class="card mb-4 shadow-sm">
                        <div class="card-body">
                            <h4 class="card-title">Train/Retrain Model</h4>
                            <form id="trainForm" method="POST" enctype="multipart/form-data">
                                <input type="hidden" name="mode" value="train"/>
                                <div class="mb-3">
                                    <label class="form-label">Acceleration CSV:</label>
                                    <input type="file" name="acceleration_file" class="form-control" required />
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">True Velocity CSV:</label>
                                    <input type="file" name="true_velocity_file" class="form-control" required />
                                </div>
                                <button type="submit" class="btn btn-primary">Train</button>
                            </form>
                            <div id="trainResults" class="results-block" style="display: none;"></div>
                        </div>
                    </div>
                </div>

                <!-- Predict Form -->
                <div class="col-md-6">
                    <div class="card mb-4 shadow-sm">
                        <div class="card-body">
                            <h4 class="card-title">Predict</h4>
                            <form id="predictForm" method="POST" enctype="multipart/form-data">
                                <input type="hidden" name="mode" value="predict"/>
                                <div class="mb-3">
                                    <label class="form-label">Acceleration CSV:</label>
                                    <input type="file" name="acceleration_file" class="form-control" required />
                                </div>
                                <button type="submit" class="btn btn-success">Predict</button>
                            </form>
                            <div id="predictResults" class="results-block" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Bootstrap JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

        <!-- Inline JS to handle form submissions and show results inline -->
        <script>
            const trainForm = document.getElementById('trainForm');
            const trainResultsDiv = document.getElementById('trainResults');
            trainForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                trainResultsDiv.style.display = 'block';
                trainResultsDiv.innerHTML = '<b>Training...</b>';

                const formData = new FormData(trainForm);
                // We'll post to /process
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    // Show error
                    trainResultsDiv.innerHTML = '<div class="text-danger">Error: ' + data.error + '</div>';
                } else {
                    // Display metrics and the base64 plot
                    // We'll do a custom HTML like before
                    let html = '';
                    
                    if (data.average_velocities_on_test_dataset) {
                        const avgData = data.average_velocities_on_test_dataset;
                        html += '<h5>Average Velocities on Test Dataset</h5>';
                        html += '<p>Average Corrected Velocity: ' + avgData.Average_Corrected_Velocity.toFixed(3) + '</p>';
                        html += '<p>Average True Velocity: ' + avgData.Average_True_Velocity.toFixed(3) + '</p>';
                        html += '<p>Difference (Corrected vs True): ' + avgData.Difference_Corrected_vs_True.toFixed(3) + '</p>';
                    }
                    if (data.model_evaluation) {
                        const evalData = data.model_evaluation;
                        html += '<h5>Model Evaluation</h5>';
                        html += '<p>Corrected Velocity MAE: ' + evalData.Corrected_Velocity_MAE.toFixed(3) + '</p>';
                        html += '<p>Corrected Velocity RMSE: ' + evalData.Corrected_Velocity_RMSE.toFixed(3) + '</p>';
                        html += '<p>Test Set MAE: ' + evalData.Test_Set_MAE.toFixed(3) + '</p>';
                        html += '<p>Test Set RMSE: ' + evalData.Test_Set_RMSE.toFixed(3) + '</p>';
                        html += '<p>IoU Accuracy: ' + evalData.IoU_Accuracy.toFixed(3) + '%</p>';
                    }
                    if (data.plot_image_base64) {
                        html += '<h5>Plot:</h5>';
                        html += '<img class="plot-img" src="data:image/png;base64,' + data.plot_image_base64 + '"/>';
                    }

                    trainResultsDiv.innerHTML = html;
                }
            });

            const predictForm = document.getElementById('predictForm');
            const predictResultsDiv = document.getElementById('predictResults');
            predictForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                predictResultsDiv.style.display = 'block';
                predictResultsDiv.innerHTML = '<b>Predicting...</b>';

                const formData = new FormData(predictForm);
                // We'll post to /predict
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    predictResultsDiv.innerHTML = '<div class="text-danger">Error: ' + data.error + '</div>';
                } else {
                    // data.predictions is an array of {time, calculated_velocity, predicted_correction, corrected_velocity}
                    let html = '<h5>Predictions</h5><pre>' + JSON.stringify(data, null, 2) + '</pre>';
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
    <h1 class="text-center">Welcome to Velocity Processing!</h1>
    <p class="text-center"><a href="/upload">Go to Upload Page</a></p>
    """

if __name__ == '__main__':
    app.run(debug=True)
