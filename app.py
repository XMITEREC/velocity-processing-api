import os
import io
import uuid
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Heroku: non-GUI backend
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template_string
from pymongo import MongoClient
from math import sqrt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 1) MongoDB Connection: read from environment variable (or fallback locally)
# -----------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["velocity_db"]   # automatically created on first write
accel_collection = db["accel_data"]
true_collection  = db["true_data"]

MODEL_FILENAME = "model.pkl"
saved_model = None

# Attempt to load any saved model from disk
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        print(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        print(f"Could not load model from {MODEL_FILENAME}. Error: {str(e)}")
        saved_model = None


# -----------------------------------------------------------------------------
# 2) HELPER FUNCTIONS: Acceleration Integration & True Velocity Expansion
# -----------------------------------------------------------------------------
def preprocess_acceleration_to_velocity(df,
                                        time_col='time',
                                        ax_col='ax (m/s^2)',
                                        ay_col='ay (m/s^2)',
                                        az_col='az (m/s^2)'):
    """
    1) Remove spikes via rolling mean/stdev replacement.
    2) Integrate acceleration -> velocity.
    """
    # Rolling means and std for spike removal
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

    # Drop these helper columns
    df.drop(columns=['ax_rolling_mean', 'ay_rolling_mean', 'az_rolling_mean',
                     'ax_rolling_std', 'ay_rolling_std', 'az_rolling_std'], inplace=True)

    # Integrate to get velocity
    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]
    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        dt = df.loc[i, 'time_diff']

        # Pick dominant axis or total magnitude
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


def preprocess_true_velocity(df_true, df_accel,
                             time_col='time',
                             speed_col='speed'):
    """
    Expand the true velocity dataset to match the # of rows in df_accel,
    randomizing speeds ±5% for intermediate points.
    """
    df_true = df_true[[time_col, speed_col]]
    n1 = len(df_accel)
    n2 = len(df_true)
    if n2 == 0:
        raise ValueError("True velocity dataset is empty, cannot expand.")

    ratio = n1 / n2
    ratio_minus_1_int = int(np.floor(ratio - 1)) if ratio > 1 else 0

    expanded_speeds = []
    for i in range(n2):
        original_speed = df_true[speed_col].iloc[i]
        # 1) original row
        expanded_speeds.append(original_speed)
        # 2) add ratio-1 new rows
        for _ in range(ratio_minus_1_int):
            new_val = np.random.uniform(original_speed*0.95, original_speed*1.05)
            expanded_speeds.append(new_val)

    # Fill remainder
    current_length = len(expanded_speeds)
    remainder = n1 - current_length
    if remainder > 0:
        last_speed = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            new_val = np.random.uniform(last_speed*0.95, last_speed*1.05)
            expanded_speeds.append(new_val)

    expanded_speeds = expanded_speeds[:n1]

    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded


# -----------------------------------------------------------------------------
# 3) UTILITY: Insert CSV Data into Mongo, Each with a Unique dataset_id
# -----------------------------------------------------------------------------
def store_csv_in_mongo(collection, df, dataset_id, tag_name):
    """
    Convert df rows into dicts, attach dataset_id + tag_name, and insert them.
    """
    records = df.to_dict(orient="records")
    for r in records:
        r["dataset_id"] = dataset_id
        r["tag"] = tag_name
    collection.insert_many(records)


# -----------------------------------------------------------------------------
# 4) Retrieving *All* Data from Mongo & Re-Training
# -----------------------------------------------------------------------------
def train_on_all_data():
    """
    1) Retrieve all data from both accel_collection & true_collection.
    2) Preprocess, integrate, expand, train on the entire big dataset.
    3) Return the model, metrics, final DataFrame, IoU, etc.
    """
    # Grab all docs
    accel_docs = list(accel_collection.find({}))
    true_docs  = list(true_collection.find({}))

    # If no data => cannot train
    if not accel_docs or not true_docs:
        raise ValueError("No acceleration or true velocity data found in Mongo.")

    # Convert to DataFrame
    accel_df = pd.DataFrame(accel_docs)
    true_df  = pd.DataFrame(true_docs)

    # Drop _id columns
    if "_id" in accel_df.columns:
        accel_df.drop(columns=["_id"], inplace=True)
    if "_id" in true_df.columns:
        true_df.drop(columns=["_id"], inplace=True)

    # Preprocess
    accel_df = preprocess_acceleration_to_velocity(accel_df)
    true_df_expanded = preprocess_true_velocity(true_df, accel_df)

    # Combine
    df = pd.DataFrame()
    df['time']          = accel_df['time']
    df['velocity']      = accel_df['velocity']
    df['true_velocity'] = true_df_expanded['true_velocity']
    df['correction']    = df['true_velocity'] - df['velocity']

    # Train
    X = df[['time', 'velocity']].values
    y = df['correction'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate test set
    y_pred_test = model.predict(X_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Predict full
    df['predicted_correction'] = model.predict(X)
    df['corrected_velocity']   = df['velocity'] + df['predicted_correction']

    mae_corr = mean_absolute_error(df['true_velocity'], df['corrected_velocity'])
    rmse_corr= np.sqrt(mean_squared_error(df['true_velocity'], df['corrected_velocity']))

    # IoU Accuracy
    min_vals = np.minimum(df['true_velocity'], df['corrected_velocity'])
    max_vals = np.maximum(df['true_velocity'], df['corrected_velocity'])
    iou_accuracy = (min_vals.sum() / max_vals.sum()) * 100

    return {
        "model": model,
        "df": df,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "mae_corr": mae_corr,
        "rmse_corr": rmse_corr,
        "iou_accuracy": iou_accuracy
    }


# -----------------------------------------------------------------------------
# 5) Counting Unique Datasets
# -----------------------------------------------------------------------------
def get_number_of_datasets():
    """
    Find how many distinct dataset_ids are present across either accel or true data.
    If both collections store the same dataset_ids, you can just check one.
    But let's check union from both for clarity.
    """
    accel_ids = accel_collection.distinct("dataset_id")
    true_ids  = true_collection.distinct("dataset_id")

    # The union of both sets:
    all_ids = set(accel_ids).union(set(true_ids))
    return len(all_ids)


# -----------------------------------------------------------------------------
# 6) /process Endpoint: Upload CSVs, Insert, Re-Train on *All* Data
# -----------------------------------------------------------------------------
@app.route('/process', methods=['POST'])
def process_endpoint():
    global saved_model
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error": "Please provide acceleration_file + true_velocity_file"}), 400

    accel_file = request.files['acceleration_file']
    true_file  = request.files['true_velocity_file']

    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1) Read CSVs
        accel_df = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        true_df  = pd.read_csv(io.StringIO(true_file.read().decode("utf-8")),  low_memory=False)

        # 2) Lowercase columns
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns  = true_df.columns.str.lower()

        # 3) Quick column checks
        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel  = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration: {missing_accel}"}), 400

        required_true = ['time', 'speed']
        missing_true  = [c for c in required_true if c not in true_df.columns]
        if missing_true:
            return jsonify({"error": f"Missing columns in true velocity: {missing_true}"}), 400

        # 4) Generate dataset_id
        dataset_id = str(uuid.uuid4())

        # 5) Insert to Mongo
        store_csv_in_mongo(accel_collection, accel_df, dataset_id, tag_name="accel")
        store_csv_in_mongo(true_collection,  true_df,  dataset_id, tag_name="true")

        # 6) Re-Train on *all* data from Mongo
        result = train_on_all_data()
        model  = result["model"]
        df     = result["df"]
        iou    = result["iou_accuracy"]

        # 7) If IoU≥95%, save model
        if iou >= 95.0:
            joblib.dump(model, MODEL_FILENAME)
            saved_model = model
            print(f"Model saved (IoU≥95%).")

        # 8) Create a plot
        plt.figure(figsize=(10,6))
        plt.plot(df['time'], df['true_velocity'], label='True Velocity', linestyle='--')
        plt.plot(df['time'], df['velocity'],      label='Calculated Velocity')
        plt.plot(df['time'], df['corrected_velocity'], label='Corrected Velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity Comparison (All Data in Mongo)')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 9) Acknowledgment: how many total datasets do we have now?
        num_datasets = get_number_of_datasets()
        ack_message  = f"Trained on {num_datasets} total dataset(s) so far."

        # 10) Build response
        response = {
            "acknowledgment": ack_message,
            "model_evaluation": {
                "Test_Set_MAE":  result["mae_test"],
                "Test_Set_RMSE": result["rmse_test"],
                "Corrected_Velocity_MAE": result["mae_corr"],
                "Corrected_Velocity_RMSE": result["rmse_corr"],
                "IoU_Accuracy": iou
            },
            "plot_image_base64": image_base64
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# 7) /predict Endpoint: Only Acceleration, Use the Saved Model
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    global saved_model
    if 'acceleration_file' not in request.files:
        return jsonify({"error": "Please provide 'acceleration_file'"}), 400

    accel_file = request.files['acceleration_file']
    if accel_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # If we haven't loaded the model yet, try now
        if not saved_model and os.path.exists(MODEL_FILENAME):
            saved_model = joblib.load(MODEL_FILENAME)
        if not saved_model:
            return jsonify({"error": "No saved model found (Train first with IoU≥95%)."}), 400

        # Read CSV
        accel_df = pd.read_csv(io.StringIO(accel_file.read().decode("utf-8")), low_memory=False)
        accel_df.columns = accel_df.columns.str.lower()

        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel  = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration: {missing_accel}"}), 400

        # Integrate
        accel_df = preprocess_acceleration_to_velocity(accel_df)
        X = accel_df[['time', 'velocity']].values

        # Predict correction
        predicted_correction = saved_model.predict(X)
        corrected_velocity   = accel_df['velocity'] + predicted_correction

        # Return average corrected velocity
        avg_corrected_vel = float(corrected_velocity.mean())

        return jsonify({
            "message": "Predicted corrected velocity using the saved model.",
            "average_corrected_velocity": avg_corrected_vel
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# 8) Basic HTML: /upload route with two forms (Train & Predict)
# -----------------------------------------------------------------------------
@app.route('/upload', methods=['GET'])
def upload_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Velocity Processing (Mongo + All Data Retained)</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            body { margin: 40px; }
            .results { margin-top: 20px; border: 1px solid #ccc; padding: 15px; }
            .plot-img { max-width: 100%; margin-top: 10px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-primary mb-4">Velocity Processing with MongoDB (Cumulative Training)</h1>
            <p>
              1) Each new dataset is stored in Mongo with a unique dataset_id.<br/>
              2) We train on <strong>all</strong> data in Mongo each time, so old data is never lost.<br/>
              3) If IoU ≥ 95%, the model is saved locally. Then you can do /predict with only acceleration.
            </p>

            <div class="row">
                <!-- Train Form -->
                <div class="col-md-6">
                    <div class="card p-3 mb-4">
                        <h3>Train on New Dataset</h3>
                        <form id="trainForm" method="POST" enctype="multipart/form-data">
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
                        <div id="trainResults" class="results" style="display:none;"></div>
                    </div>
                </div>

                <!-- Predict Form -->
                <div class="col-md-6">
                    <div class="card p-3 mb-4">
                        <h3>Predict (Saved Model)</h3>
                        <form id="predictForm" method="POST" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label class="form-label">Acceleration CSV:</label>
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
        trainForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            trainResultsDiv.style.display = 'block';
            trainResultsDiv.innerHTML = '<b>Training on all data... please wait.</b>';

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
                    html += '<h5>' + data.acknowledgment + '</h5>';
                }
                if (data.model_evaluation) {
                    const evalData = data.model_evaluation;
                    html += '<h5>Model Evaluation</h5>';
                    html += '<p>Test Set MAE: ' + evalData.Test_Set_MAE.toFixed(3) + '</p>';
                    html += '<p>Test Set RMSE: ' + evalData.Test_Set_RMSE.toFixed(3) + '</p>';
                    html += '<p>Corrected Vel. MAE: ' + evalData.Corrected_Velocity_MAE.toFixed(3) + '</p>';
                    html += '<p>Corrected Vel. RMSE: ' + evalData.Corrected_Velocity_RMSE.toFixed(3) + '</p>';
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
            predictResultsDiv.innerHTML = '<b>Predicting with saved model...</b>';

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
    <h1>Welcome to Velocity Processing (Mongo + All Data Retained)!</h1>
    <p><a href="/upload">Go to Upload Page</a></p>
    """

if __name__ == '__main__':
    app.run(debug=True)
