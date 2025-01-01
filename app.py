import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Heroku
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import joblib  # For saving/loading the model
from pymongo import MongoClient, ASCENDING
import gridfs
from bson.objectid import ObjectId
import logging
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# Initialize Flask app
app = Flask(__name__)
 
# ==============================
# Configuration and Initialization
# ==============================
 
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    logger.error("MONGODB_URI environment variable not set.")
    raise EnvironmentError("MONGODB_URI environment variable not set.")
 
try:
    client = MongoClient(MONGODB_URI)
    db = client['velocity_db']
    fs = gridfs.GridFS(db)
    logger.info("Connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise
 
# Collections
models_collection = db['models']
training_datasets_collection = db['training_datasets']
 
# Data Retention Configuration
MAX_TRAINING_DATASETS = 100  # Maximum number of training datasets to retain
 
# Global variable to cache the loaded model
saved_model = None
 
# ==============================
# Helper Functions
# ==============================
 
def load_model():
    """
    Load the saved model from MongoDB using GridFS.
    """
    global saved_model
    if saved_model is None:
        model_doc = models_collection.find_one({'model_name': 'velocity_corrector'})
        if model_doc and 'model_id' in model_doc:
            try:
                model_data = fs.get(model_doc['model_id']).read()
                saved_model = joblib.loads(model_data)
                logger.info("Loaded saved model from MongoDB.")
            except Exception as e:
                logger.error(f"Error loading model from MongoDB: {str(e)}")
                saved_model = None
        else:
            logger.warning("No saved model found in MongoDB.")
            saved_model = None
    return saved_model
 
def save_model(model):
    """
    Save the trained model to MongoDB using GridFS.
    """
    try:
        # Serialize the model
        model_data = joblib.dumps(model)
        # Delete existing model if any
        models_collection.delete_many({'model_name': 'velocity_corrector'})
        # Store the new model in GridFS
        model_id = fs.put(model_data, filename='model.pkl')
        # Insert a document referencing the model
        models_collection.insert_one({'model_name': 'velocity_corrector', 'model_id': model_id})
        global saved_model
        saved_model = model  # Update the cached model
        logger.info("Model saved to MongoDB.")
    except Exception as e:
        logger.error(f"Error saving model to MongoDB: {str(e)}")
 
def preprocess_acceleration_to_velocity(df, time_col='time', ax_col='ax (m/s^2)', ay_col='ay (m/s^2)', az_col='az (m/s^2)'):
    """
    Integrates acceleration into velocity,
    removing spikes via rolling mean thresholds.
    """
    try:
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
 
            # Pick the dominant axis or total magnitude
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
    except Exception as e:
        logger.error(f"Error in preprocess_acceleration_to_velocity: {str(e)}")
        raise
 
def preprocess_true_velocity(df_true, df_accel, time_col='time', speed_col='speed'):
    """
    Expand the 'true velocity' dataset to match the row count of df_accel,
    using linear interpolation.
    """
    try:
        df_true = df_true[[time_col, speed_col]].sort_values(by=time_col).reset_index(drop=True)
        df_accel_sorted = df_accel.sort_values(by=time_col).reset_index(drop=True)
 
        # Merge on time with interpolation
        df_merged = pd.merge_asof(df_accel_sorted, df_true, on=time_col, direction='nearest', tolerance=np.inf)
 
        if df_merged[speed_col].isnull().any():
            raise ValueError("True velocity dataset does not cover all time points in acceleration dataset.")
 
        df_merged.rename(columns={speed_col: 'true_velocity'}, inplace=True)
        return df_merged[['time', 'true_velocity']]
    except Exception as e:
        logger.error(f"Error in preprocess_true_velocity: {str(e)}")
        raise
 
def process_data():
    """
    Process all stored training datasets, train the model, evaluate it,
    and generate a visualization.
    """
    try:
        # Retrieve all datasets, parse into DataFrames
        master_dfs = []
        for dataset in training_datasets_collection.find().sort('upload_time', ASCENDING):
            dataset_df = pd.read_json(dataset['data'], orient='split')
            master_dfs.append(dataset_df)
 
        if master_dfs:
            master_df = pd.concat(master_dfs, ignore_index=True)
            logger.info(f"Retrieved {len(master_dfs)} training datasets from MongoDB.")
        else:
            logger.warning("No training datasets found in MongoDB.")
            return {"error": "No training data available to train the model."}
 
        # Ensure consistent columns
        required_columns = ['time', 'velocity', 'true_velocity', 'correction']
        for col in required_columns:
            if col not in master_df.columns:
                logger.error(f"Missing column '{col}' in master dataset.")
                return {"error": f"Missing column '{col}' in master dataset."}
 
        # 1) Train a random forest on master dataset
        X = master_df[['time', 'velocity']].values
        y = master_df['correction'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
 
        # 2) Evaluate on test set
        y_pred_test = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
 
        # 3) Predict on full data
        master_df['predicted_correction'] = model.predict(X)
        master_df['corrected_velocity'] = master_df['velocity'] + master_df['predicted_correction']
 
        # 4) Evaluate corrected velocity
        mae_corrected = mean_absolute_error(master_df['true_velocity'], master_df['corrected_velocity'])
        rmse_corrected = np.sqrt(mean_squared_error(master_df['true_velocity'], master_df['corrected_velocity']))
 
        # 5) IoU Accuracy
        min_values = np.minimum(master_df['true_velocity'], master_df['corrected_velocity'])
        max_values = np.maximum(master_df['true_velocity'], master_df['corrected_velocity'])
        iou_accuracy = (np.sum(min_values) / np.sum(max_values)) * 100
        logger.info(f"IoU Accuracy: {iou_accuracy:.4f}%")
 
        # 6) If IoU≥95%, save model
        if iou_accuracy >= 95.0:
            save_model(model)
            logger.info(f"Model saved to MongoDB because IoU ≥ 95%.")
        else:
            logger.warning(f"Model not saved because IoU < 95%. IoU: {iou_accuracy:.4f}%")
 
        # 7) Create a comparison plot for master dataset
        plt.figure(figsize=(10, 6))
        plt.plot(master_df['time'], master_df['true_velocity'], label='True Velocity', linestyle='--')
        plt.plot(master_df['time'], master_df['velocity'], label='Calculated Velocity')
        plt.plot(master_df['time'], master_df['corrected_velocity'], label='Corrected Velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity Comparison (Master Dataset)')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        logger.info("Comparison plot generated.")
 
        # 8) Test subset stats
        avg_corrected = mae_corrected  # MAE is used as average difference
        avg_true = 0.0  # MAE between true_velocity and itself is zero
        diff_corr_true = abs(avg_corrected - avg_true)
        logger.info("Test subset statistics calculated.")
 
        # 9) Data Retention: Delete oldest datasets if exceeding threshold
        total_datasets = training_datasets_collection.count_documents({})
        if total_datasets > MAX_TRAINING_DATASETS:
            datasets_to_delete = training_datasets_collection.find().sort('upload_time', ASCENDING).limit(total_datasets - MAX_TRAINING_DATASETS)
            for dataset in datasets_to_delete:
                training_datasets_collection.delete_one({'_id': dataset['_id']})
                logger.info(f"Deleted training dataset with _id: {dataset['_id']} due to retention policy.")
 
        # Prepare results
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
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        return {"error": f"Error during data processing: {str(e)}"}
 
 # ==============================
 # Flask Routes
 # ==============================
 
 @app.route('/process', methods=['POST'])
 def process_endpoint():
     """
     Train the model with acceleration + true velocity, compute IoU, optionally save model.
     Stores training dataset in MongoDB and returns metrics + base64 plot.
     """
     if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
         logger.error("Missing 'acceleration_file' or 'true_velocity_file' in request.")
         return jsonify({"error": "Please provide both 'acceleration_file' and 'true_velocity_file'"}), 400
 
     accel_file = request.files['acceleration_file']
     true_file = request.files['true_velocity_file']
 
     if accel_file.filename == '' or true_file.filename == '':
         logger.error("No selected file for 'acceleration_file' or 'true_velocity_file'.")
         return jsonify({"error": "No selected file"}), 400
 
     try:
         # Read acceleration data
         accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
         # Read true velocity data
         true_df = pd.read_csv(io.StringIO(true_file.stream.read().decode("UTF8")), low_memory=False)
 
         # Standardize column names to lowercase
         accel_df.columns = accel_df.columns.str.lower()
         true_df.columns = true_df.columns.str.lower()
 
         # Validate required columns
         required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
         missing_accel = [c for c in required_accel if c not in accel_df.columns]
         if missing_accel:
             logger.error(f"Missing columns in acceleration dataset: {missing_accel}")
             return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400
 
         required_true = ['time', 'speed']
         missing_true = [c for c in required_true if c not in true_df.columns]
         if missing_true:
             logger.error(f"Missing columns in true velocity dataset: {missing_true}")
             return jsonify({"error": f"Missing columns in true velocity dataset: {missing_true}"}), 400
 
         # Preprocess acceleration data
         accel_df_processed = preprocess_acceleration_to_velocity(accel_df)
         # Preprocess true velocity data
         true_df_processed = preprocess_true_velocity(true_df, accel_df_processed)
 
         # Merge into a single DataFrame
         df = pd.DataFrame()
         df['time'] = accel_df_processed['time']
         df['velocity'] = accel_df_processed['velocity']
         df['true_velocity'] = true_df_processed['true_velocity']
         df['correction'] = df['true_velocity'] - df['velocity']
 
         # Store the merged training dataset in MongoDB
         dataset_json = df.to_json(orient='split')
         training_datasets_collection.insert_one({
             'upload_time': pd.Timestamp.utcnow(),
             'data': dataset_json
         })
         logger.info("Uploaded training dataset stored in MongoDB.")
 
         # Process data and train model
         results = process_data()
         if 'error' in results:
             logger.error(f"Error during processing: {results['error']}")
             return jsonify({"error": results['error']}), 500
         return jsonify(results), 200
 
     except Exception as e:
         logger.error(f"Exception during processing: {str(e)}")
         return jsonify({"error": str(e)}), 500
 
 @app.route('/predict', methods=['POST'])
 def predict_endpoint():
     """
     Use the saved model (if available) to predict corrected velocity from acceleration only.
     Returns only the AVERAGE corrected velocity for the current dataset.
     """
     if 'acceleration_file' not in request.files:
         logger.error("Missing 'acceleration_file' in request.")
         return jsonify({"error": "Please provide 'acceleration_file'"}), 400
 
     accel_file = request.files['acceleration_file']
     if accel_file.filename == '':
         logger.error("No selected file for 'acceleration_file'.")
         return jsonify({"error": "No selected file"}), 400
 
     try:
         # Load the model from MongoDB
         model = load_model()
         if not model:
             logger.error("No saved model found. Please train first (IoU≥95%).")
             return jsonify({"error": "No saved model found. Please train first (IoU≥95%)."}), 400
 
         # Read acceleration data
         accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
         accel_df.columns = accel_df.columns.str.lower()
 
         # Validate required columns
         required_accel = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
         missing_accel = [c for c in required_accel if c not in accel_df.columns]
         if missing_accel:
             logger.error(f"Missing columns in acceleration dataset: {missing_accel}")
             return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel}"}), 400
 
         # Preprocess acceleration to velocity
         accel_df_processed = preprocess_acceleration_to_velocity(accel_df)
 
         # Prepare data for prediction
         X = accel_df_processed[['time', 'velocity']].values
 
         # Predict corrections
         predicted_correction = model.predict(X)
         corrected_velocity = accel_df_processed['velocity'] + predicted_correction
 
         # Calculate average corrected velocity
         avg_corrected_vel = float(corrected_velocity.mean())
 
         # Create a plot for the current dataset
         plt.figure(figsize=(10, 6))
         plt.plot(accel_df_processed['time'], corrected_velocity, label='Corrected Velocity', color='green')
         plt.plot(accel_df_processed['time'], accel_df_processed['velocity'], label='Calculated Velocity', color='blue', linestyle='--')
         plt.xlabel('Time')
         plt.ylabel('Velocity')
         plt.title('Velocity Prediction (Current Dataset)')
         plt.legend()
         buf = io.BytesIO()
         plt.savefig(buf, format='png')
         buf.seek(0)
         image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
         plt.close()
         logger.info("Prediction and plot generation completed.")
 
         return jsonify({
             "message": "Predicted corrected velocity using saved model.",
             "average_corrected_velocity": avg_corrected_vel,
             "plot_image_base64": image_base64
         }), 200
 
     except Exception as e:
         logger.error(f"Exception during prediction: {str(e)}")
         return jsonify({"error": str(e)}), 500
 
 @app.route('/upload', methods=['GET'])
 def upload_page():
     """
     A single page with 2 forms:
       1) Train (acceleration + true velocity) -> /process
       2) Predict (acceleration only) -> /predict
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
             .results-block {
                 margin-top: 1rem;
                 padding: 1rem;
                 border: 1px solid #ccc;
                 border-radius: 5px;
                 background-color: #f8f9fa;
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
             <h1 class="text-primary mb-4">Velocity Processing & Permanent Model</h1>
             <p>1) Train with acceleration &amp; true velocity (<code>/process</code>).
                If IoU &ge; 95%, model is saved. Then you can do inference with only acceleration (<code>/predict</code>).
             </p>
             <div class="alert alert-info">
               <strong>Note:</strong> Ensure that the model is trained successfully before making predictions.
             </div>
 
             <div class="row">
                 <!-- Training Form -->
                 <div class="col-md-6">
                     <div class="card mb-4 shadow-sm">
                         <div class="card-body">
                             <h4 class="card-title">Train/Retrain Model</h4>
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
                             <div id="trainResults" class="results-block" style="display: none;"></div>
                         </div>
                     </div>
                 </div>
 
                 <!-- Predict Form -->
                 <div class="col-md-6">
                     <div class="card mb-4 shadow-sm">
                         <div class="card-body">
                             <h4 class="card-title">Predict (Average Velocity)</h4>
                             <form id="predictForm" method="POST" enctype="multipart/form-data">
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
 
         <!-- Optional Bootstrap JS -->
         <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
 
         <!-- Inline JS to handle form submissions via fetch -->
         <script>
             const trainForm = document.getElementById('trainForm');
             const trainResultsDiv = document.getElementById('trainResults');
             trainForm.addEventListener('submit', async (e) => {
                 e.preventDefault();
                 trainResultsDiv.style.display = 'block';
                 trainResultsDiv.innerHTML = '<b>Training...</b>';
 
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
                 const response = await fetch('/predict', {
                     method: 'POST',
                     body: formData
                 });
                 const data = await response.json();
 
                 if (!response.ok && data.error) {
                     predictResultsDiv.innerHTML = '<div class="text-danger">Error: ' + data.error + '</div>';
                 } else {
                     // Display average_corrected_velocity and plot
                     let html = '<h5>Prediction Result</h5>';
                     html += '<p><strong>Average Corrected Velocity:</strong> ' + data.average_corrected_velocity.toFixed(3) + '</p>';
                     if (data.plot_image_base64) {
                         html += '<h5>Plot:</h5>';
                         html += '<img class="plot-img" src="data:image/png;base64,' + data.plot_image_base64 + '"/>';
                     }
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
 
 # ==============================
 # Run the Flask App
 # ==============================
 
 if __name__ == '__main__':
     app.run(debug=True)
