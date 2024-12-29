import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import random

app = Flask(__name__)

# ---------------------------
# 1) Preprocess Acceleration
# ---------------------------
def preprocess_acceleration_to_velocity(df, time_col='time', ax_col='ax (m/s^2)', ay_col='ay (m/s^2)', az_col='az (m/s^2)'):
    """Convert acceleration data to velocity using integration, with error handling for sudden spikes."""
    # Compute rolling mean and std for dynamic thresholding
    df['ax_rolling_mean'] = df[ax_col].rolling(window=5, center=True).mean()
    df['ay_rolling_mean'] = df[ay_col].rolling(window=5, center=True).mean()
    df['az_rolling_mean'] = df[az_col].rolling(window=5, center=True).mean()

    df['ax_rolling_std'] = df[ax_col].rolling(window=5, center=True).std().fillna(0)
    df['ay_rolling_std'] = df[ay_col].rolling(window=5, center=True).std().fillna(0)
    df['az_rolling_std'] = df[az_col].rolling(window=5, center=True).std().fillna(0)

    # Define a dynamic threshold as multiple of standard deviation
    std_multiplier = 3

    # Replace spikes with rolling mean
    for col, mean_col, std_col in zip([ax_col, ay_col, az_col],
                                      ['ax_rolling_mean', 'ay_rolling_mean', 'az_rolling_mean'],
                                      ['ax_rolling_std', 'ay_rolling_std', 'az_rolling_std']):
        df[col] = np.where(
            abs(df[col] - df[mean_col]) > std_multiplier * df[std_col],
            df[mean_col],
            df[col]
        )

    # Drop intermediate columns
    df.drop(columns=['ax_rolling_mean', 'ay_rolling_mean', 'az_rolling_mean',
                     'ax_rolling_std', 'ay_rolling_std', 'az_rolling_std'], inplace=True)

    # Calculate velocity by integration
    df['time_diff'] = df[time_col].diff().fillna(0)
    velocity = [0]  # Starting velocity

    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        time_diff = df.loc[i, 'time_diff']

        # Determine dominant axis or combined magnitude
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

# -------------------------------------------
# 2) Preprocess True Velocity (Expansion)
# -------------------------------------------
def preprocess_true_velocity(df_true, df_accel, time_col='time', speed_col='speed'):
    """
    Expand df_true so that it has the same number of rows as df_accel.
    - Ratio = len(df_accel) / len(df_true).
    - For each original row in df_true, add (ratio - 1) new rows.
    - Keep only [time_col, speed_col] from df_true.
    - Fill new rows' speed with random values near the original row's speed.
    - Assign time from df_accel.
    """
    # Keep only time and speed
    df_true = df_true[[time_col, speed_col]]

    # Calculate final row count
    n1 = len(df_accel)  # target number of rows
    n2 = len(df_true)   # original number of rows
    if n2 == 0:
        raise ValueError("True velocity dataset has 0 rows. Cannot expand.")

    ratio = n1 / n2
    # We'll use the integer part of (ratio - 1) for expansions after each original row:
    ratio_minus_1_int = int(np.floor(ratio - 1)) if ratio > 1 else 0

    # Prepare a list to hold the expanded speeds
    expanded_speeds = []
    original_index = 0

    for i in range(n2):
        original_speed = df_true[speed_col].iloc[i]
        # 1) Append the original row's speed
        expanded_speeds.append(original_speed)

        # 2) Append (ratio - 1) new rows (integer part)
        for _ in range(ratio_minus_1_int):
            # Generate a random speed ~5% around the original_speed
            low_val  = original_speed * 0.95
            high_val = original_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    # We may still have a remainder to get exactly n1 rows
    current_length = len(expanded_speeds)
    remainder = n1 - current_length

    if remainder > 0:
        # We'll base it on the last original speed
        last_speed = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            low_val  = last_speed * 0.95
            high_val = last_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    # Now we have at least n1 rows, but we need exactly n1
    # (In rare cases, we could slightly overshoot if ratio < 2, but let's ensure we slice to n1)
    expanded_speeds = expanded_speeds[:n1]

    # Create a new DataFrame with the same time index as df_accel
    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,   # copy the entire time column from df_accel
        'true_velocity': expanded_speeds       # the newly expanded speeds
    })

    return df_expanded

# -------------------------------------------
# 3) Processing Function
# -------------------------------------------
def process_data(accel_df, true_df):
    # Preprocess acceleration to velocity
    accel_df = preprocess_acceleration_to_velocity(
        accel_df,
        time_col='time',
        ax_col='ax (m/s^2)',
        ay_col='ay (m/s^2)',
        az_col='az (m/s^2)'
    )

    # Expand True Velocity to Match Acceleration Rows
    true_df_expanded = preprocess_true_velocity(
        df_true=true_df,
        df_accel=accel_df,
        time_col='time',
        speed_col='speed'  # must match your column name in the true velocity file
    )

    # Create a single DataFrame with 'calculated_velocity' and 'true_velocity'
    time_col = 'time'
    calculated_velocity_col = 'velocity'
    true_velocity_col = 'true_velocity'

    # We can directly align them since they now have the same length and same times:
    df = pd.DataFrame()
    df[time_col] = accel_df[time_col]
    df[calculated_velocity_col] = accel_df[calculated_velocity_col]
    df[true_velocity_col] = true_df_expanded[true_velocity_col]

    # Add a correction column
    df['correction'] = df[true_velocity_col] - df[calculated_velocity_col]

    # Prepare data for ML
    X = df[[time_col, calculated_velocity_col]].values
    y = df['correction'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on test set
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Predict corrections on the full dataset
    df['predicted_correction'] = model.predict(X)
    df['corrected_velocity'] = df[calculated_velocity_col] + df['predicted_correction']

    # Evaluate corrected velocity
    mae_corrected = mean_absolute_error(df[true_velocity_col], df['corrected_velocity'])
    rmse_corrected = np.sqrt(mean_squared_error(df[true_velocity_col], df['corrected_velocity']))

    # Retrain condition (if 95%+ accuracy)
    if mae_corrected / df[true_velocity_col].mean() <= 0.05:
        model.fit(X, y)  # Retrain on the new data

    # Prepare visualization as base64 encoded image
    plt.figure(figsize=(10, 6))
    plt.plot(df[time_col], df[true_velocity_col], label='True Velocity', linestyle='--')
    plt.plot(df[time_col], df[calculated_velocity_col], label='Calculated Velocity')
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

    # Compute averages for the test dataset
    test_times = X_test[:, 0]  # the times used in the test split
    test_df = df[df[time_col].isin(test_times)]

    avg_corrected_velocity = test_df['corrected_velocity'].mean()
    avg_true_velocity = test_df[true_velocity_col].mean()
    velocity_difference = abs(avg_corrected_velocity - avg_true_velocity)

    # Compile results
    results = {
        "model_evaluation": {
            "Test_Set_MAE": mae_test,
            "Test_Set_RMSE": rmse_test,
            "Corrected_Velocity_MAE": mae_corrected,
            "Corrected_Velocity_RMSE": rmse_corrected
        },
        "average_velocities_on_test_dataset": {
            "Average_Corrected_Velocity": avg_corrected_velocity,
            "Average_True_Velocity": avg_true_velocity,
            "Difference_Corrected_vs_True": velocity_difference
        },
        "plot_image_base64": image_base64
    }

    return results

# -------------------------------------------
# 4) API Endpoint
# -------------------------------------------
@app.route('/process', methods=['POST'])
def process_endpoint():
    if 'acceleration_file' not in request.files or 'true_velocity_file' not in request.files:
        return jsonify({"error": "Please provide both 'acceleration_file' and 'true_velocity_file'"}), 400

    accel_file = request.files['acceleration_file']
    true_file = request.files['true_velocity_file']

    if accel_file.filename == '' or true_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read CSV files into DataFrames
        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        true_df = pd.read_csv(io.StringIO(true_file.stream.read().decode("UTF8")), low_memory=False)

        # Convert all column names to lowercase for consistency
        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns = true_df.columns.str.lower()

        # Check required columns
        required_accel_columns = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel_cols = [col for col in required_accel_columns if col not in accel_df.columns]
        if missing_accel_cols:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel_cols}"}), 400

        required_true_columns = ['time', 'speed']
        missing_true_cols = [col for col in required_true_columns if col not in true_df.columns]
        if missing_true_cols:
            return jsonify({"error": f"Missing columns in true velocity dataset: {missing_true_cols}"}), 400

        # Process data
        results = process_data(accel_df, true_df)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: A simple homepage for testing
@app.route('/', methods=['GET'])
def index():
    return """
    <h1>Velocity Processing API</h1>
    <p>Use the /process endpoint to upload acceleration and true velocity CSV files.</p>
    """

if __name__ == '__main__':
    app.run(debug=True)
