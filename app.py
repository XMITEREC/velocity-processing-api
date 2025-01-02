import os
import io
import base64
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from urllib.parse import urlparse
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# If needed, from sklearn.model_selection import train_test_split

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv("variables.env")  # Make sure this file is present.

ACCELERATION_DATA_S3_PATH = os.getenv("ACCELERATION_DATA_S3_PATH")
VELOCITY_DATA_S3_PATH = os.getenv("VELOCITY_DATA_S3_PATH")
MASTER_DATASET_S3_PATH = os.getenv("MASTER_DATASET_PATH")  # e.g. s3://yourbucket/MasterDataset/

# Temporary local paths
LOCAL_ACCEL_PATH = "/tmp/new_accel_data.csv"
LOCAL_VELOCITY_PATH = "/tmp/new_velocity_data.csv"
LOCAL_MASTER_PATH = "/tmp/MasterDataset.csv"

app = Flask(__name__)

# ---------------------------------------------------------
# Helpers: S3, Parsing, etc.
# ---------------------------------------------------------
def parse_s3_path(s3_path):
    """Parse S3 path into (bucket, prefix)."""
    parsed = urlparse(s3_path)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 path: {s3_path}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix

def s3_file_exists(s3_client, bucket, key):
    """Check if file/key exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False

def download_s3_file(s3_client, bucket, key, local_path):
    """Download from S3 to local path."""
    s3_client.download_file(bucket, key, local_path)

def upload_s3_file(s3_client, local_path, bucket, key):
    """Upload local file to S3."""
    s3_client.upload_file(local_path, bucket, key)

# ---------------------------------------------------------
# Master Dataset Load/Save
# ---------------------------------------------------------
def load_master_dataset_from_s3(s3_client, master_s3_path):
    """
    Download MasterDataset.csv if it exists; otherwise, return empty DataFrame.
    """
    bucket, prefix = parse_s3_path(master_s3_path)
    if prefix.endswith('/'):
        prefix += "MasterDataset.csv"
    master_key = prefix

    if s3_file_exists(s3_client, bucket, master_key):
        download_s3_file(s3_client, bucket, master_key, LOCAL_MASTER_PATH)
        df_master = pd.read_csv(LOCAL_MASTER_PATH)
    else:
        df_master = pd.DataFrame(
            columns=['time','ax','ay','az','true_velocity','dataset_id']
        )
    return df_master, bucket, master_key

def save_master_dataset_to_s3(s3_client, df_master, bucket, key):
    df_master.to_csv(LOCAL_MASTER_PATH, index=False)
    upload_s3_file(s3_client, LOCAL_MASTER_PATH, bucket, key)

# ---------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------
def preprocess_acceleration_to_velocity(df, time_col='time', ax_col='ax', ay_col='ay', az_col='az'):
    """Convert acceleration data to velocity with rolling spike smoothing."""
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Rolling mean & std for dynamic outlier detection
    df['ax_rm'] = df[ax_col].rolling(window=5, center=True).mean()
    df['ay_rm'] = df[ay_col].rolling(window=5, center=True).mean()
    df['az_rm'] = df[az_col].rolling(window=5, center=True).mean()

    df['ax_rs'] = df[ax_col].rolling(window=5, center=True).std().fillna(0)
    df['ay_rs'] = df[ay_col].rolling(window=5, center=True).std().fillna(0)
    df['az_rs'] = df[az_col].rolling(window=5, center=True).std().fillna(0)

    std_multiplier = 3
    for col, rm, rs in zip([ax_col, ay_col, az_col], ['ax_rm','ay_rm','az_rm'], ['ax_rs','ay_rs','az_rs']):
        df[col] = np.where(
            abs(df[col] - df[rm]) > std_multiplier * df[rs],
            df[rm],
            df[col]
        )

    df.drop(columns=['ax_rm','ay_rm','az_rm','ax_rs','ay_rs','az_rs'], inplace=True)

    # Integrate to get velocity
    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0.0]
    for i in range(1, len(df)):
        ax_i = df.loc[i, ax_col]
        ay_i = df.loc[i, ay_col]
        az_i = df.loc[i, az_col]
        dt = df.loc[i, 'time_diff']

        # Choose dominant or magnitude
        if abs(ax_i) > abs(ay_i) and abs(ax_i) > abs(az_i):
            accel = ax_i
        elif abs(ay_i) > abs(ax_i) and abs(ay_i) > abs(az_i):
            accel = ay_i
        elif abs(az_i) > abs(ax_i) and abs(az_i) > abs(ay_i):
            accel = az_i
        else:
            accel = np.sqrt(ax_i**2 + ay_i**2 + az_i**2)

        velocity.append(velocity[-1] + accel*dt)

    df['calculated_velocity'] = velocity
    return df

def expand_true_velocity(df_true, df_accel, time_col='time', speed_col='speed'):
    """Expand velocity data to match row count of acceleration."""
    if df_true.empty:
        return pd.DataFrame(columns=[time_col, 'true_velocity'])

    df_true = df_true[[time_col, speed_col]].copy()
    df_true.rename(columns={speed_col: 'true_velocity'}, inplace=True)

    n1 = len(df_accel)
    n2 = len(df_true)
    if n2 == 0:
        return pd.DataFrame(columns=[time_col, 'true_velocity'])

    ratio = n1 / n2
    expanded_speeds = []
    ratio_minus_1_int = int(np.floor(ratio - 1)) if ratio > 1 else 0

    for i in range(n2):
        orig_speed = df_true['true_velocity'].iloc[i]
        expanded_speeds.append(orig_speed)
        for _ in range(ratio_minus_1_int):
            expanded_speeds.append(np.random.uniform(orig_speed*0.95, orig_speed*1.05))

    remainder = n1 - len(expanded_speeds)
    if remainder > 0:
        last_speed = df_true['true_velocity'].iloc[-1]
        for _ in range(remainder):
            expanded_speeds.append(np.random.uniform(last_speed*0.95, last_speed*1.05))

    expanded_speeds = expanded_speeds[:n1]

    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded

def create_dataset_id(df_master):
    if df_master.empty:
        return 1
    return df_master['dataset_id'].max() + 1

def merge_into_master(df_accel, df_true_expanded, dataset_id):
    """Combine into a single DF with columns: time, ax, ay, az, true_velocity, dataset_id."""
    if 'true_velocity' not in df_true_expanded.columns:
        # no velocity => all NaN
        df_merged = pd.DataFrame({
            'time': df_accel['time'],
            'ax': df_accel['ax'],
            'ay': df_accel['ay'],
            'az': df_accel['az'],
            'true_velocity': np.nan,
            'dataset_id': dataset_id
        })
    else:
        df_merged = pd.DataFrame({
            'time': df_accel['time'],
            'ax': df_accel['ax'],
            'ay': df_accel['ay'],
            'az': df_accel['az'],
            'true_velocity': df_true_expanded['true_velocity'],
            'dataset_id': dataset_id
        })
    return df_merged

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_model_on_master(df_master):
    """Train on all rows that have real velocity (true_velocity != NaN)."""
    df_trainable = df_master.dropna(subset=['true_velocity']).copy()
    if df_trainable.empty:
        return None  # no data to train on

    # For each dataset_id, recalc "calculated_velocity"
    dfs = []
    for d_id in df_trainable['dataset_id'].unique():
        subset = df_trainable[df_trainable['dataset_id'] == d_id].copy()
        subset = preprocess_acceleration_to_velocity(subset)
        subset['correction'] = subset['true_velocity'] - subset['calculated_velocity']
        dfs.append(subset)
    df_trainable = pd.concat(dfs, ignore_index=True)

    X = df_trainable[['time','calculated_velocity']].values
    y = df_trainable['correction'].values
    if len(X) < 2:
        return None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def compute_iou_accuracy(y_true, y_pred):
    """Intersection over Union style: sum(min) / sum(max)."""
    min_vals = np.minimum(y_true, y_pred)
    max_vals = np.maximum(y_true, y_pred)
    return (np.sum(min_vals) / np.sum(max_vals)) * 100.0

def repeated_training_until_95(df_master, max_iter=5):
    """Train multiple times if IoU < 95%, up to max_iter. Return the final model."""
    if df_master['true_velocity'].dropna().empty:
        # no ground-truth => can't compute IoU, just train once
        return train_model_on_master(df_master)

    model = None
    iou_value = 0.0
    for i in range(1, max_iter+1):
        model = train_model_on_master(df_master)
        if not model:
            break  # no model

        # Evaluate IoU on entire master
        df_eval = df_master[~df_master['true_velocity'].isna()].copy()
        if df_eval.empty:
            break
        # Recalc velocity for each dataset
        df_list = []
        for d_id in df_eval['dataset_id'].unique():
            sub = df_eval[df_eval['dataset_id'] == d_id].copy()
            sub = preprocess_acceleration_to_velocity(sub)
            pred_corr = model.predict(sub[['time','calculated_velocity']].values)
            sub['corrected_velocity'] = sub['calculated_velocity'] + pred_corr
            df_list.append(sub)
        df_eval = pd.concat(df_list, ignore_index=True)

        y_true = df_eval['true_velocity'].values
        y_pred = df_eval['corrected_velocity'].values
        iou_value = compute_iou_accuracy(y_true, y_pred)
        print(f"[Train Iter {i}] IoU = {iou_value:.2f}%")

        if iou_value >= 95.0:
            print("Reached IoU >= 95%. Stopping repeated training.")
            break

    return model

# ---------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------
def run_training_pipeline(accel_path, velocity_path, has_velocity):
    """
    1) Read & preprocess data
    2) Merge into Master
    3) Retrain model
    4) Return metrics & base64 plot (if velocity is present)
    """
    s3_client = boto3.client('s3')

    # 1) Read Acceleration
    df_accel = pd.read_csv(accel_path)
    df_accel.columns = df_accel.columns.str.lower()
    required_cols = ['time','ax (m/s^2)','ay (m/s^2)','az (m/s^2)']
    for c in required_cols:
        if c not in df_accel.columns:
            raise ValueError("Acceleration CSV missing columns: time, ax (m/s^2), ay (m/s^2), az (m/s^2)")
    df_accel = df_accel[required_cols].rename(columns={
        'ax (m/s^2)': 'ax',
        'ay (m/s^2)': 'ay',
        'az (m/s^2)': 'az'
    })

    # 2) Possibly read velocity
    if has_velocity:
        df_vel = pd.read_csv(velocity_path)
        df_vel.columns = df_vel.columns.str.lower()
        if 'time' not in df_vel.columns or 'speed' not in df_vel.columns:
            # treat as no velocity
            df_vel = pd.DataFrame()
            has_velocity = False
    else:
        df_vel = pd.DataFrame()

    # expand velocity if present
    df_accel_proc = preprocess_acceleration_to_velocity(df_accel.copy())
    df_vel_expanded = expand_true_velocity(df_vel.copy(), df_accel_proc)

    # 3) Load Master
    df_master, master_bucket, master_key = load_master_dataset_from_s3(s3_client, MASTER_DATASET_S3_PATH)

    # 4) Create dataset_id & merge
    new_id = create_dataset_id(df_master)
    df_new = merge_into_master(df_accel, df_vel_expanded, new_id)
    df_master = pd.concat([df_master, df_new], ignore_index=True)
    df_master.drop_duplicates(subset=['time','dataset_id'], inplace=True)

    # 5) Save Master
    save_master_dataset_to_s3(s3_client, df_master, master_bucket, master_key)

    # 6) Retrain Model
    model = repeated_training_until_95(df_master, max_iter=5)
    if model is None:
        return {"error": "No valid model trained. Possibly no velocity data in entire master."}

    # 7) Evaluate & Plot for the new dataset only
    df_new_sorted = df_new.sort_values(by='time').reset_index(drop=True)
    df_new_sorted = preprocess_acceleration_to_velocity(df_new_sorted)
    predicted_corr = model.predict(df_new_sorted[['time','calculated_velocity']].values)
    df_new_sorted['corrected_velocity'] = df_new_sorted['calculated_velocity'] + predicted_corr

    # Prepare results
    results = {}
    # Make a plot in memory
    plt.figure(figsize=(8, 5))
    plt.plot(df_new_sorted['time'], df_new_sorted['corrected_velocity'], label='Corrected Velocity')
    plt.plot(df_new_sorted['time'], df_new_sorted['calculated_velocity'], label='Calculated Velocity', linestyle='--')

    if not df_new_sorted['true_velocity'].isna().all():
        # we do have real velocity
        valid_mask = ~df_new_sorted['true_velocity'].isna()
        y_true = df_new_sorted.loc[valid_mask, 'true_velocity'].values
        y_pred = df_new_sorted.loc[valid_mask, 'corrected_velocity'].values

        if len(y_true) >= 2:
            iou = compute_iou_accuracy(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = sqrt(mse)
            results.update({
                "iou_accuracy": round(iou, 3),
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4)
            })

        plt.plot(df_new_sorted['time'], df_new_sorted['true_velocity'], label='True Velocity', linestyle=':')

    plt.title(f"Velocity Comparison (Dataset ID={new_id})")
    plt.xlabel('Time')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    results["plot_base64"] = plot_base64

    # Additional average velocities
    avg_calc = df_new_sorted['calculated_velocity'].mean()
    avg_corr = df_new_sorted['corrected_velocity'].mean()
    results["avg_calculated_velocity"] = round(float(avg_calc), 3)
    results["avg_corrected_velocity"] = round(float(avg_corr), 3)
    if not df_new_sorted['true_velocity'].isna().all():
        avg_true = df_new_sorted['true_velocity'].mean()
        results["avg_true_velocity"] = round(float(avg_true), 3)
        results["difference_corr_vs_true"] = round(float(avg_corr - avg_true), 3)

    return results

def run_prediction_pipeline(accel_path):
    """
    1) Use existing trained model from Master (if any).
    2) Predict on new acceleration data only (no velocity).
    3) Return corrected velocities + base64 plot.
    """
    s3_client = boto3.client('s3')

    # 1) Load Master & train model (only on those with real velocity)
    df_master, master_bucket, master_key = load_master_dataset_from_s3(s3_client, MASTER_DATASET_S3_PATH)
    model = repeated_training_until_95(df_master, max_iter=5)
    if model is None:
        return {"error": "No valid model trained. Master dataset has no velocity data."}

    # 2) Read new acceleration
    df_accel = pd.read_csv(accel_path)
    df_accel.columns = df_accel.columns.str.lower()
    required_cols = ['time','ax (m/s^2)','ay (m/s^2)','az (m/s^2)']
    for c in required_cols:
        if c not in df_accel.columns:
            raise ValueError("Acceleration CSV missing columns: time, ax (m/s^2), ay (m/s^2), az (m/s^2)")
    df_accel = df_accel[required_cols].rename(columns={
        'ax (m/s^2)': 'ax',
        'ay (m/s^2)': 'ay',
        'az (m/s^2)': 'az'
    })

    df_accel_proc = preprocess_acceleration_to_velocity(df_accel.copy())
    predicted_corr = model.predict(df_accel_proc[['time','calculated_velocity']].values)
    df_accel_proc['corrected_velocity'] = df_accel_proc['calculated_velocity'] + predicted_corr

    # (Optional) If you want to store these predicted rows in Master with true_velocity=NaN, do so:
    # new_id = create_dataset_id(df_master)
    # df_new = merge_into_master(df_accel, pd.DataFrame(), new_id)
    # save_master_dataset_to_s3(...)  # Caution: you might not want to pollute the dataset.

    # 3) Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_accel_proc['time'], df_accel_proc['corrected_velocity'], label='Corrected Velocity')
    plt.plot(df_accel_proc['time'], df_accel_proc['calculated_velocity'], label='Calculated Velocity', linestyle='--')
    plt.title("Velocity Prediction (No Ground Truth)")
    plt.xlabel('Time')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Return results
    avg_calc = df_accel_proc['calculated_velocity'].mean()
    avg_corr = df_accel_proc['corrected_velocity'].mean()
    return {
        "plot_base64": plot_base64,
        "avg_calculated_velocity": round(float(avg_calc), 3),
        "avg_corrected_velocity": round(float(avg_corr), 3)
    }

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------
@app.route('/train', methods=['POST'])
def train_endpoint():
    """
    POST /train
    Form-data:
      accel_csv: the acceleration CSV
      velocity_csv: the velocity CSV (optional)
    Returns JSON with metrics + base64 plot
    """
    accel_file = request.files.get('accel_csv')
    velocity_file = request.files.get('velocity_csv')

    if not accel_file:
        return jsonify({"error": "accel_csv file is required"}), 400

    accel_file.save(LOCAL_ACCEL_PATH)
    has_velocity = False
    if velocity_file:
        velocity_file.save(LOCAL_VELOCITY_PATH)
        has_velocity = True

    try:
        results = run_training_pipeline(LOCAL_ACCEL_PATH, LOCAL_VELOCITY_PATH, has_velocity)
        if "error" in results:
            return jsonify(results), 500
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    POST /predict
    Form-data:
      accel_csv: the acceleration CSV
    Returns JSON with predicted velocities + base64 plot
    """
    accel_file = request.files.get('accel_csv')
    if not accel_file:
        return jsonify({"error": "accel_csv file is required"}), 400

    accel_file.save(LOCAL_ACCEL_PATH)
    try:
        results = run_prediction_pipeline(LOCAL_ACCEL_PATH)
        if "error" in results:
            return jsonify(results), 500
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Uncomment for local debugging:
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
