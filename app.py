import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Heroku or any headless server
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template_string
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from flask_wtf import CSRFProtect

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_secure_random_secret_key'  # Replace with a secure key in production

# Initialize CSRF Protection
csrf = CSRFProtect(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize MongoDB Client
MONGODB_URI = "mongodb+srv://herokuUser:12345@cluster0.jhaoh.mongodb.net/velocity_db?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGODB_URI)
db = client['velocity_db']
accel_collection = db['acceleration_data']
true_velocity_collection = db['true_velocity_data']

# Model Configuration
MODEL_FILENAME = 'model.pkl'
saved_model = None

# If a model was saved before, try loading it
if os.path.exists(MODEL_FILENAME):
    try:
        saved_model = joblib.load(MODEL_FILENAME)
        logger.info(f"Loaded saved model from {MODEL_FILENAME}")
    except Exception as e:
        logger.error(f"Could not load model. Error: {str(e)}")
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
    # Rolling means and std
    rolling_window = 5
    factor = 3

    for axis in ['ax', 'ay', 'az']:
        mean_col = f'{axis}_mean'
        std_col = f'{axis}_std'
        df[f'{axis}_mean'] = df[f'{axis} (m/s^2)'].rolling(window=rolling_window, center=True).mean()
        df[f'{axis}_std'] = df[f'{axis} (m/s^2)'].rolling(window=rolling_window, center=True).std().fillna(0)
        # Remove spikes
        df[f'{axis} (m/s^2)'] = np.where(
            np.abs(df[f'{axis} (m/s^2)'] - df[f'{axis}_mean']) > factor * df[f'{axis}_std'],
            df[f'{axis}_mean'],
            df[f'{axis} (m/s^2)']
        )
    
    # Drop temporary columns
    df.drop(columns=[f'{axis}_mean' for axis in ['ax', 'ay', 'az']] +
                [f'{axis}_std' for axis in ['ax', 'ay', 'az']], inplace=True)

    # Integrate acceleration to velocity using vectorized operations
    df['time_diff'] = df[time_col].diff().fillna(0)
    
    # Determine dominant axis or use magnitude
    accel_array = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].abs().values
    accel = np.empty(len(df))
    accel[:] = np.sqrt(df['ax (m/s^2)']**2 + df['ay (m/s^2)']**2 + df['az (m/s^2)']**2)
    dominant_mask_ax = (accel_array[:, 0] > accel_array[:, 1]) & (accel_array[:, 0] > accel_array[:, 2])
    dominant_mask_ay = (accel_array[:, 1] > accel_array[:, 0]) & (accel_array[:, 1] > accel_array[:, 2])
    dominant_mask_az = (accel_array[:, 2] > accel_array[:, 0]) & (accel_array[:, 2] > accel_array[:, 1])

    accel[dominant_mask_ax] = df.loc[dominant_mask_ax, 'ax (m/s^2)']
    accel[dominant_mask_ay] = df.loc[dominant_mask_ay, 'ay (m/s^2)']
    accel[dominant_mask_az] = df.loc[dominant_mask_az, 'az (m/s^2)']

    # Calculate cumulative velocity
    df['velocity'] = np.cumsum(accel * df['time_diff'])
    
    return df

def expand_true_velocity(df_true, df_accel,
                         time_col='time',
                         speed_col='speed'):
    """
    Expand the true velocity so it matches the number of rows in df_accel
    by randomizing speeds ±5% between each original point.
    """
    df_true = df_true[[time_col, speed_col]].reset_index(drop=True)
    n_acc = len(df_accel)
    n_true = len(df_true)
    
    if n_true == 0:
        raise ValueError("True velocity is empty; cannot expand.")
    
    # Calculate the ratio
    ratio = n_acc / n_true
    ratio_floor = int(np.floor(ratio)) if ratio > 1 else 1
    
    # Repeat each speed ratio_floor times and add random variation
    repeated_speeds = df_true[speed_col].repeat(ratio_floor).values
    repeated_speeds = repeated_speeds[:n_acc]
    
    # Add random variation of ±5%
    random_variation = np.random.uniform(0.95, 1.05, size=len(repeated_speeds))
    expanded_speeds = repeated_speeds * random_variation
    
    # If still not enough, pad with the last speed
    if len(expanded_speeds) < n_acc:
        remainder = n_acc - len(expanded_speeds)
        last_speed = df_true[speed_col].iloc[-1]
        expanded_speeds = np.concatenate([expanded_speeds,
                                          np.random.uniform(last_speed*0.95, last_speed*1.05, size=remainder)])
    else:
        expanded_speeds = expanded_speeds[:n_acc]
    
    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded

def compute_iou(true_vel, corrected_vel):
    """
    Compute Intersection over Union (IoU) between true and corrected velocities.
    """
    min_vals = np.minimum(true_vel, corrected_vel)
    max_vals = np.maximum(true_vel, corrected_vel)
    denominator = max_vals.sum()
    if denominator == 0:
        return 0.0
    return (min_vals.sum() / denominator) * 100

################################################################################
# 3) TRAINING FUNCTION: Repeated training on all data until IoU≥95 or max loops
################################################################################

def train_on_all_data(max_loops=5):
    """
    Merges all old + new data from MongoDB, trains once, and repeats if IoU<95%
    up to max_loops times. Returns the final model and metrics.
    """
    # Fetch all acceleration data from MongoDB
    accel_cursor = accel_collection.find()
    accel_records = list(accel_cursor)
    if not accel_records:
        raise ValueError("No acceleration data available for training.")
    full_accel_df = pd.DataFrame(accel_records)
    
    # Fetch all true velocity data from MongoDB
    true_cursor = true_velocity_collection.find()
    true_records = list(true_cursor)
    if not true_records:
        raise ValueError("No true velocity data available for training.")
    full_true_df = pd.DataFrame(true_records)

    # Ensure 'time' is sorted
    full_accel_df = full_accel_df.sort_values(by='time').reset_index(drop=True)
    full_true_df = full_true_df.sort_values(by='time').reset_index(drop=True)

    # We'll do repeated training loops
    best_model   = None
    best_metrics = {}
    iou_acc      = 0

    for loop_i in range(max_loops):
        logger.info(f"Training loop {loop_i + 1}/{max_loops}")
        try:
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
            logger.info("Model training completed.")

            # Evaluate
            y_test_pred = model.predict(X_test)
            mae_test  = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Predict on full
            df['predicted_correction'] = model.predict(X)
            df['corrected_velocity']   = df['velocity'] + df['predicted_correction']

            mae_corr = mean_absolute_error(df['true_velocity'], df['corrected_velocity'])
            rmse_corr= np.sqrt(mean_squared_error(df['true_velocity'], df['corrected_velocity']))
            iou_acc  = compute_iou(df['true_velocity'].values, df['corrected_velocity'].values)

            logger.info(f"Loop {loop_i + 1}: IoU Accuracy = {iou_acc:.2f}%")

            # Assign current model and metrics
            best_model = model
            best_metrics = {
                "df": df,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "mae_corr": mae_corr,
                "rmse_corr": rmse_corr,
                "iou_acc":  iou_acc
            }

            # If IoU≥95%, we can break early
            if iou_acc >= 95.0:
                logger.info("Desired IoU achieved. Stopping training loops.")
                break

        except Exception as e:
            logger.exception(f"An error occurred during training loop {loop_i + 1}: {str(e)}")
            continue  # Proceed to next loop if any

    return best_model, best_metrics

################################################################################
# 4) ROUTES
################################################################################

@app.route('/process', methods=['POST'])
@csrf.exempt
def process_endpoint():
    """
    1) Upload acceleration + true velocity
    2) Insert into MongoDB
    3) Retrain on *all* data (old + new) with repeated loops if needed
    4) Save model if IoU≥95
    5) Plot only the *new* dataset
    6) Return acknowledgment & stats
    """
    global saved_model
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        logger.warning("Missing acceleration_file or true_velocity_file in the request.")
        return jsonify({"error": "Need both acceleration_file and true_velocity_file"}), 400

    accel_file = request.files['acceleration_file']
    true_file  = request.files['true_velocity_file']

    if accel_file.filename == '' or true_file.filename == '':
        logger.warning("No selected file for acceleration or true velocity.")
        return jsonify({"error": "No selected file"}), 400

    if not (allowed_file(accel_file.filename) and allowed_file(true_file.filename)):
        logger.warning("Uploaded files are not allowed. Only CSV files are accepted.")
        return jsonify({"error": "Only CSV files are allowed."}), 400

    try:
        # Secure filenames
        accel_filename = secure_filename(accel_file.filename)
        true_filename = secure_filename(true_file.filename)

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
            logger.warning(f"Missing columns in acceleration: {missing_accel}")
            return jsonify({"error": f"Missing columns in acceleration: {missing_accel}"}),
