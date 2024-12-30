import os
import io
import uuid
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Heroku
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template_string
from pymongo import MongoClient
from math import sqrt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

################################################################################
# 1) MongoDB Connection: Read from environment variable (MONGO_URI)
################################################################################

# Read MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)

# Select database and collections
db = client["velocity_db"]  # Database name
accel_collection = db["accel_data"]  # Collection for acceleration data
true_collection  = db["true_data"]   # Collection for true velocity data

# Model filename
MODEL_FILENAME = "model.pkl"
saved_model = None

# Attempt to load an existing model
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        print(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        print(f"Could not load model from {MODEL_FILENAME}. Error: {str(e)}")
        saved_model = None

################################################################################
# 2) HELPER FUNCTIONS: Data Preprocessing and Training
################################################################################

def preprocess_acceleration_to_velocity(df,
                                        time_col='time',
                                        ax_col='ax (m/s^2)',
                                        ay_col='ay (m/s^2)',
                                        az_col='az (m/s^2)'):
    """
    1) Remove spikes using rolling mean ±3*std
    2) Integrate acceleration -> velocity
    """
    # Rolling means and std for spike removal
    df['ax_mean'] = df[ax_col].rolling(window=5, center=True).mean()
    df['ay_mean'] = df[ay_col].rolling(window=5, center=True).mean()
    df['az_mean'] = df[az_col].rolling(window=5, center=True).mean()

    df['ax_std'] = df[ax_col].rolling(window=5, center=True).std().fillna(0)
    df['ay_std'] = df[ay_col].rolling(window=5, center=True).std().fillna(0)
    df['az_std'] = df[az_col].rolling(window=5, center=True).std().fillna(0)

    factor = 3
    for col, mcol, scol in zip([ax_col, ay_col, az_col],
                               ['ax_mean', 'ay_mean', 'az_mean'],
                               ['ax_std', 'ay_std', 'az_std']):
        df[col] = np.where(
            abs(df[col] - df[mcol]) > factor * df[scol],
            df[mcol],
            df[col]
        )

    # Drop temporary columns
    df.drop(columns=['ax_mean', 'ay_mean', 'az_mean', 'ax_std', 'ay_std', 'az_std'], inplace=True)

    # Integrate to get velocity
    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]  # Initial velocity

    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        dt = df.loc[i, 'time_diff']

        # Choose dominant axis or magnitude
        if abs(ax) > abs(ay) and abs(ax) > abs(az):
            accel = ax
        elif abs(ay) > abs(ax) and abs(ay) > abs(az):
            accel = ay
        elif abs(az) > abs(ax) and abs(az) > abs(ay):
            accel = az
        else:
            accel = sqrt(ax**2 + ay**2 + az**2)

        velocity.append(velocity[-1] + accel * dt)

    df['velocity'] = velocity
    return df

def expand_true_velocity(df_true, df_accel,
                         time_col='time',
                         speed_col='speed'):
    """
    Expand the true velocity dataset to match the number of rows in df_accel
    by randomizing speeds ±5% for intermediate points.
    """
    df_true = df_true[[time_col, speed_col]]
    n_acc = len(df_accel)
    n_true = len(df_true)

    if n_true == 0:
        raise ValueError("True velocity dataset is empty; cannot expand.")

    ratio = n_acc / n_true
    ratio_minus_1_int = int(np.floor(ratio - 1)) if ratio > 1 else 0

    expanded_speeds = []
    for i in range(n_true):
        orig_speed = df_true[speed_col].iloc[i]
        # Original speed
        expanded_speeds.append(orig_speed)
        # Add ratio_minus_1 new speeds
        for _ in range(ratio_minus_1_int):
            new_val = np.random.uniform(orig_speed * 0.95, orig_speed * 1.05)
            expanded_speeds.append(new_val)

    # Fill the remainder if any
    current_length = len(expanded_speeds)
    remainder = n_acc - current_length
    if remainder > 0:
        last_speed = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            new_val = np.random.uniform(last_speed * 0.95, last_speed * 1.05)
            expanded_speeds.append(new_val)

    # Trim to exact length
    expanded_speeds = expanded_speeds[:n_acc]

    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded

def compute_iou(true_vel, corrected_vel):
    """
    Compute Intersection over Union (IoU) accuracy.
    """
    min_vals = np.minimum(true_vel, corrected_vel)
    max_vals = np.maximum(true_vel, corrected_vel)
    return (min_vals.sum() / max_vals.sum()) * 100

def train_on_all_data():
    """
    Train the model on all accumulated data from MongoDB.
    Returns the trained model and evaluation metrics.
    """
    # Retrieve all acceleration data
    accel_docs = list(accel_collection.find({}))
    true_docs  = list(true_collection.find({}))

    if not accel_docs or not true_docs:
        raise ValueError("No data available in MongoDB for training.")

    # Convert to DataFrames
    accel_df = pd.DataFrame(accel_docs)
    true_df  = pd.DataFrame(true_docs)

    # Drop MongoDB's ObjectId if present
    if "_id" in accel_df.columns:
        accel_df.drop(columns=["_id"], inplace=True)
    if "_id" in true_df.columns:
        true_df.drop(columns=["_id"], inplace=True)

    # Preprocess acceleration data
    accel_df = preprocess_acceleration_to_velocity(accel_df)

    # Expand true velocity data
    true_df_expanded = expand_true_velocity(true_df, accel_df)

    # Combine into a single DataFrame for training
    df = pd.DataFrame()
    df['time'] = accel_df['time']
    df['velocity'] = accel_df['velocity']
    df['true_velocity'] = true_df_expanded['true_velocity']
    df['correction'] = df['true_velocity'] - df['velocity']

    # Prepare training data
    X = df[['time', 'velocity']].values
    y = df['correction'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Predict on the entire dataset
    df['predicted_correction'] = model.predict(X)
    df['corrected_velocity'] = df['velocity'] + df['predicted_correction']

    # Compute evaluation metrics
    mae_corr = mean_absolute_error(df['true_velocity'], df['corrected_velocity'])
    rmse_corr = np.sqrt(mean_squared_error(df['true_velocity'], df['corrected_velocity']))
    iou_acc = compute_iou(df['true_velocity'], df['corrected_velocity'])

    return {
        "model": model,
        "df": df,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "mae_corr": mae_corr,
        "rmse_corr": rmse_corr,
        "iou_accuracy": iou_acc
    }

################################################################################
# 3) DATA STORAGE: Insert Uploaded CSVs into MongoDB with Unique dataset_id
################################################################################

def store_csv_in_mongo(collection, df, dataset_id, tag_name):
    """
    Insert DataFrame rows into MongoDB collection with dataset_id and tag_name.
    """
    records = df.to_dict(orient="records")
    for record in records:
        record["dataset_id"] = dataset_id
        record["tag"] = tag_name
    collection.insert_many(records)

################################################################################
# 4) COUNT UNIQUE DATASETS
################################################################################

def get_total_datasets():
    """
    Count the total number of unique dataset_ids across accel and true collections.
    Assumes each dataset has both accel and true data.
    """
    accel_ids = accel_collection.distinct("dataset_id")
    true_ids = true_collection.distinct("dataset_id")
    all_ids = set(accel_ids).union(set(true_ids))
    return len(all_ids)

################################################################################
# 5) ROUTES
################################################################################

@app.route('/process', methods=['POST'])
def process_endpoint():
    """
    Handle training:
    1. Receive acceleration and true velocity CSVs.
    2. Store them in MongoDB with a unique dataset_id.
    3. Train the model on all accumulated data.
    4. Save the model if IoU >= 95%.
    5. Plot only the current dataset's velocities.
    6. Return acknowledgment, metrics, average velocities, and plot.
    """
    global saved_model

    # Check if both files are present
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error": "Please provide both 'acceleration_file' and 'true_velocity_file'"}), 400

    accel_file = request.files['acceleration_file']
    true_file = request.files['true_velocity_file']

    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read CSV files into DataFrames
        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        true_df  = pd.read_csv(io.StringIO(true_file.stream.read().decode("UTF8")), low_memory=False)

        # Normalize column names to lowercase
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns  = true_df.columns.str.lower()

        # Check for required columns
        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400

        required_true = ['time', 'speed']
        missing_true = [c for c in required_true if c not in true_df.columns]
        if missing_true:
            return jsonify({"error": f"Missing columns in true velocity dataset: {missing_true}"}), 400

        # Generate a unique dataset_id
        dataset_id = str(uuid.uuid4())

        # Store the datasets in MongoDB
        store_csv_in_mongo(accel_collection, accel_df, dataset_id, tag_name="accel")
        store_csv_in_mongo(true_collection,  true_df,  dataset_id, tag_name="true")

        # Train the model on all data
        training_result = train_on_all_data()
        model = training_result["model"]
        df_trained = training_result["df"]
        mae_test = training_result["mae_test"]
        rmse_test = training_result["rmse_test"]
        mae_corr = training_result["mae_corr"]
        rmse_corr = training_result["rmse_corr"]
        iou_acc = training_result["iou_accuracy"]

        # Save the model if IoU >= 95%
        if iou_acc >= 95.0:
            joblib.dump(model, MODEL_FILENAME)
            saved_model = model
            print("Model saved (IoU >= 95%).")

        # Fetch only the current dataset's data from MongoDB
        current_accel_docs = list(accel_collection.find({"dataset_id": dataset_id}))
        current_true_docs  = list(true_collection.find({"dataset_id": dataset_id}))

        current_accel_df = pd.DataFrame(current_accel_docs)
        current_true_df  = pd.DataFrame(current_true_docs)

        # Drop MongoDB's ObjectId if present
        if "_id" in current_accel_df.columns:
            current_accel_df.drop(columns=["_id"], inplace=True)
        if "_id" in current_true_df.columns:
            current_true_df.drop(columns=["_id"], inplace=True)

        # Preprocess current dataset
        current_accel_processed = preprocess_acceleration_to_velocity(current_accel_df.copy())
        current_true_expanded  = expand_true_velocity(current_true_df.copy(), current_accel_processed)

        # Combine current dataset into DataFrame
        df_current = pd.DataFrame()
        df_current['time'] = current_accel_processed['time']
        df_current['velocity'] = current_accel_processed['velocity']
        df_current['true_velocity'] = current_true_expanded['true_velocity']
        df_current['correction'] = df_current['true_velocity'] - df_current['velocity']

        # Predict corrections using the trained model
        X_current = df_current[['time', 'velocity']].values
        df_current['predicted_correction'] = model.predict(X_current)
        df_current['corrected_velocity'] = df_current['velocity'] + df_current['predicted_correction']

        # Calculate the difference between corrected and true velocities
        df_current['difference'] = df_current['corrected_velocity'] - df_current['true_velocity']

        # Create a plot for the current dataset only
        plt.figure(figsize=(10, 6))
        plt.plot(df_current['time'], df_current['true_velocity'], label='True Velocity', linestyle='--')
        plt.plot(df_current['time'], df_current['velocity'], label='Calculated Velocity')
        plt.plot(df_current['time'], df_current['corrected_velocity'], label='Corrected Velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Current Dataset: True vs Calculated vs Corrected Velocity')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Calculate average velocities for the current dataset
        avg_corrected_velocity = df_current['corrected_velocity'].mean()
        avg_true_velocity      = df_current['true_velocity'].mean()

        # Get total number of datasets
        total_datasets = get_total_datasets()

        # Build acknowledgment message
        ack_message = f"Trained on {total_datasets} total dataset(s) so far. IoU Accuracy: {iou_acc:.2f}%."

        # Build JSON response
        response = {
            "acknowledgment": ack_message,
            "average_velocities_on_current_dataset": {
                "Average_Corrected_Velocity": avg_corrected_velocity,
                "Average_True_Velocity": avg_true_velocity
            },
            "model_evaluation": {
                "Test_Set_MAE": mae_test,
                "Test_Set_RMSE": rmse_test,
                "Corrected_Velocity_MAE": mae_corr,
                "Corrected_Velocity_RMSE": rmse_corr,
                "IoU_Accuracy": iou_acc
            },
            "plot_image_base64": image_base64,
            "velocity_difference_summary": {
                "Average_Difference": df_current['difference'].mean(),
                "Max_Difference": df_current['difference'].max(),
                "Min_Difference": df_current['difference'].min()
            }
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Handle prediction:
    1. Receive acceleration CSV.
    2. Process it and integrate to velocity.
    3. Use the saved model to predict corrections.
    4. Calculate corrected velocities and return the average.
    """
    global saved_model

    # Check if the file is present
    if 'acceleration_file' not in request.files:
        return jsonify({"error": "Please provide 'acceleration_file'"}), 400

    accel_file = request.files['acceleration_file']
    if accel_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # If model is not loaded, try loading it
        if not saved_model and os.path.exists(MODEL_FILENAME):
            saved_model = joblib.load(MODEL_FILENAME)
            print("Loaded saved model for prediction.")

        if not saved_model:
            return jsonify({"error": "No saved model found. Please train first (IoU ≥ 95%)."}), 400

        # Read acceleration CSV into DataFrame
        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        accel_df.columns = accel_df.columns.str.lower()

        # Check for required columns
        required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel = [c for c in required_accel if c not in accel_df.columns]
        if missing_accel:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400

        # Preprocess acceleration data
        accel_df_processed = preprocess_acceleration_to_velocity(accel_df.copy())

        # Prepare data for prediction
        X = accel_df_processed[['time', 'velocity']].values

        # Predict corrections
        predicted_corrections = saved_model.predict(X)
        corrected_velocities = accel_df_processed['velocity'] + predicted_corrections

        # Calculate average corrected velocity
        avg_corrected_velocity = float(corrected_velocities.mean())

        # Build JSON response
        response = {
            "message": "Predicted corrected velocity using the saved model.",
            "average_corrected_velocity": avg_corrected_velocity
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

################################################################################
# 4) HTML PAGE WITH TWO FORMS: TRAIN AND PREDICT
################################################################################

@app.route('/upload', methods=['GET'])
def upload_page():
    """
    Render an HTML page with two forms:
    1. Train: Upload acceleration and true velocity CSVs.
    2. Predict: Upload only acceleration CSV.
    Display results inline.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Velocity Processing with MongoDB</title>
        <!-- Bootstrap CSS -->
        <link 
            rel="stylesheet" 
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        >
        <style>
            body {
                margin-top: 40px;
                margin-bottom: 40px;
            }
            .results-block {
                margin-top: 1rem;
                padding: 1rem;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .plot-img {
                max-width: 100%;
                border: 1px solid #dee2e6;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-primary mb-4">Velocity Processing with MongoDB</h1>
            <p>
              1) Upload acceleration and true velocity datasets to train.<br/>
              2) The model trains on all accumulated data. If IoU ≥ 95%, the model is saved.<br/>
              3) Predict average corrected velocity using only acceleration data.
            </p>
            <div class="row">
                <!-- Train Form -->
                <div class="col-md-6">
                    <div class="card p-3 mb-4">
                        <h3>Train</h3>
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
                        <div id="trainResults" class="results-block" style="display:none;"></div>
                    </div>
                </div>

                <!-- Predict Form -->
                <div class="col-md-6">
                    <div class="card p-3 mb-4">
                        <h3>Predict</h3>
                        <form id="predictForm" method="POST" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label class="form-label">Acceleration CSV:</label>
                                <input type="file" name="acceleration_file" class="form-control" required />
                            </div>
                            <button type="submit" class="btn btn-success">Predict</button>
                        </form>
                        <div id="predictResults" class="results-block" style="display:none;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bootstrap JS Bundle -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

        <!-- Inline JavaScript to handle form submissions -->
        <script>
            // Handle Train Form Submission
            const trainForm = document.getElementById('trainForm');
            const trainResultsDiv = document.getElementById('trainResults');

            trainForm.addEventListener('submit', async function(e) {
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
                    if (data.average_velocities_on_current_dataset) {
                        const avgData = data.average_velocities_on_current_dataset;
                        html += '<h5>Average Velocities on Current Dataset</h5>';
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
                        html += '<h5>Current Dataset Plot:</h5>';
                        html += '<img class="plot-img" src="data:image/png;base64,' + data.plot_image_base64 + '"/>';
                    }
                    if (data.velocity_difference_summary) {
                        const diffData = data.velocity_difference_summary;
                        html += '<h5>Velocity Difference Summary</h5>';
                        html += '<p>Average Difference: ' + diffData.Average_Difference.toFixed(3) + '</p>';
                        html += '<p>Max Difference: ' + diffData.Max_Difference.toFixed(3) + '</p>';
                        html += '<p>Min Difference: ' + diffData.Min_Difference.toFixed(3) + '</p>';
                    }
                    trainResultsDiv.innerHTML = html;
                }
            });

            // Handle Predict Form Submission
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
    """
    Home page redirecting to the upload page.
    """
    return """
    <h1 class="text-center">Welcome to Velocity Processing with MongoDB!</h1>
    <p class="text-center"><a href="/upload">Go to Upload Page</a></p>
    """

################################################################################
# 6) RUN THE APP
################################################################################

if __name__ == '__main__':
    app.run(debug=True)
