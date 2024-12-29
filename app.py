import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Important for Heroku: non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

app = Flask(__name__)

# ---------------------------
# 1) Preprocess Acceleration
# ---------------------------
def preprocess_acceleration_to_velocity(df, time_col='time', ax_col='ax (m/s^2)', ay_col='ay (m/s^2)', az_col='az (m/s^2)'):
    """Convert acceleration data to velocity using integration, with error handling for sudden spikes."""
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
    velocity = [0]  # Starting velocity

    for i in range(1, len(df)):
        ax, ay, az = df.loc[i, [ax_col, ay_col, az_col]]
        time_diff = df.loc[i, 'time_diff']

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
            low_val  = original_speed * 0.95
            high_val = original_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    current_length = len(expanded_speeds)
    remainder = n1 - current_length
    if remainder > 0:
        last_speed = df_true[speed_col].iloc[-1]
        for _ in range(remainder):
            low_val  = last_speed * 0.95
            high_val = last_speed * 1.05
            new_speed = np.random.uniform(low_val, high_val)
            expanded_speeds.append(new_speed)

    expanded_speeds = expanded_speeds[:n1]

    df_expanded = pd.DataFrame({
        time_col: df_accel[time_col].values,
        'true_velocity': expanded_speeds
    })
    return df_expanded

# -------------------------------------------
# 3) Processing Function
# -------------------------------------------
def process_data(accel_df, true_df):
    accel_df = preprocess_acceleration_to_velocity(
        accel_df,
        time_col='time',
        ax_col='ax (m/s^2)',
        ay_col='ay (m/s^2)',
        az_col='az (m/s^2)'
    )

    true_df_expanded = preprocess_true_velocity(
        df_true=true_df,
        df_accel=accel_df,
        time_col='time',
        speed_col='speed'
    )

    time_col = 'time'
    calc_v_col = 'velocity'
    true_v_col = 'true_velocity'
    df = pd.DataFrame()
    df[time_col] = accel_df[time_col]
    df[calc_v_col] = accel_df[calc_v_col]
    df[true_v_col] = true_df_expanded[true_v_col]
    df['correction'] = df[true_v_col] - df[calc_v_col]

    X = df[[time_col, calc_v_col]].values
    y = df['correction'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    df['predicted_correction'] = model.predict(X)
    df['corrected_velocity'] = df[calc_v_col] + df['predicted_correction']

    mae_corrected = mean_absolute_error(df[true_v_col], df['corrected_velocity'])
    rmse_corrected = np.sqrt(mean_squared_error(df[true_v_col], df['corrected_velocity']))

    if (mae_corrected / df[true_v_col].mean()) <= 0.05:
        model.fit(X, y)

    # ------------------------------
    # IoU Accuracy Calculation
    # ------------------------------
    min_values = np.minimum(df[true_v_col], df['corrected_velocity'])
    max_values = np.maximum(df[true_v_col], df['corrected_velocity'])
    iou_accuracy = (np.sum(min_values) / np.sum(max_values)) * 100

    # Print to server logs
    print("\n=== Model Evaluation ===")
    print(f"IoU Accuracy: {iou_accuracy:.4f}%")

    # Create a comparison plot
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

    # Evaluate on the test subset
    test_times = X_test[:, 0]
    test_df = df[df[time_col].isin(test_times)]
    avg_corrected = test_df['corrected_velocity'].mean()
    avg_true = test_df[true_v_col].mean()
    diff_corr_true = abs(avg_corrected - avg_true)

    # Return all results, including IoU
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
            "IoU_Accuracy": iou_accuracy  # Added to JSON response
        },
        "plot_image_base64": image_base64
    }
    return results

# -------------------------------------------
# 4) API Endpoint: /process
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
        accel_df = pd.read_csv(io.StringIO(accel_file.stream.read().decode("UTF8")), low_memory=False)
        true_df = pd.read_csv(io.StringIO(true_file.stream.read().decode("UTF8")), low_memory=False)

        accel_df.columns = accel_df.columns.str.lower()
        true_df.columns = true_df.columns.str.lower()

        required_accel_columns = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'time']
        missing_accel_cols = [col for col in required_accel_columns if col not in accel_df.columns]
        if missing_accel_cols:
            return jsonify({"error": f"Missing columns in acceleration dataset: {missing_accel_cols}"}), 400

        required_true_columns = ['time', 'speed']
        missing_true_cols = [col for col in required_true_columns if col not in true_df.columns]
        if missing_true_cols:
            return jsonify({"error": f"Missing columns in true velocity dataset: {missing_true_cols}"}), 400

        results = process_data(accel_df, true_df)
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------
# 5) HTML Page Route: /upload (Bootstrap UI)
# -------------------------------------------
@app.route('/upload', methods=['GET'])
def upload_page():
    """
    Render an HTML page that uses Bootstrap 5 for a nicer look & feel.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Velocity Processing</title>
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
            .results-card {
                margin-top: 20px;
            }
            .plot-img {
                max-width: 100%;
                border: 1px solid #dee2e6;
                margin-top: 10px;
            }
            .card-title {
                margin-bottom: 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-primary mb-4">Velocity Processing</h1>
            <p class="mb-4">Upload your Acceleration CSV and True Velocity CSV here.</p>
            
            <div class="card p-4 shadow-sm">
                <form id="uploadForm" class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label fw-bold">Acceleration File:</label>
                        <input 
                            type="file" 
                            name="acceleration_file" 
                            accept=".csv" 
                            class="form-control" 
                            required
                        />
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold">True Velocity File:</label>
                        <input 
                            type="file" 
                            name="true_velocity_file" 
                            accept=".csv" 
                            class="form-control" 
                            required
                        />
                    </div>
                    <div class="col-12 mt-3">
                        <button type="submit" class="btn btn-primary">
                            Submit
                        </button>
                    </div>
                </form>
            </div>

            <div id="message" class="text-danger fw-bold mt-3"></div>
            
            <div id="results" class="results-card card mt-4 p-4" style="display: none;">
                <h2 class="card-title mb-3">Results:</h2>
                <div id="resultsJson"></div>
                <h4 class="mt-4">Plot:</h4>
                <img id="plotImage" class="plot-img" src="" alt="Velocity Plot"/>
            </div>
        </div>

        <!-- Bootstrap 5 JS (CDN) - for optional dynamic features -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

        <script>
            const form = document.getElementById('uploadForm');
            form.addEventListener('submit', async function(event) {
                event.preventDefault();

                const messageDiv = document.getElementById('message');
                const resultsDiv = document.getElementById('results');
                const resultsJson = document.getElementById('resultsJson');
                const plotImage = document.getElementById('plotImage');

                // Clear previous messages
                messageDiv.textContent = '';
                resultsDiv.style.display = 'none';
                resultsJson.innerHTML = '';
                plotImage.src = '';

                // Prepare form data
                const formData = new FormData(form);

                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        // If the response returned an error, display it
                        if (data.error) {
                            messageDiv.textContent = "Error: " + data.error;
                        } else {
                            messageDiv.textContent = "An unknown error occurred.";
                        }
                        return;
                    }

                    // Show the results section
                    resultsDiv.style.display = 'block';

                    // Extract data
                    const avgData = data.average_velocities_on_test_dataset;
                    const evalData = data.model_evaluation;

                    // Helper for rounding
                    function fmt(num, decimals=3) {
                        return parseFloat(num).toFixed(decimals);
                    }

                    // Build HTML
                    const customResultsHtml = `
                        <h4 class="mb-3">Average Velocities on Test Dataset</h4>
                        <ul class="list-unstyled ps-3">
                            <li><strong>Average Corrected Velocity:</strong> ${fmt(avgData.Average_Corrected_Velocity)}</li>
                            <li><strong>Average True Velocity:</strong> ${fmt(avgData.Average_True_Velocity)}</li>
                            <li><strong>Difference (Corrected vs True):</strong> ${fmt(avgData.Difference_Corrected_vs_True)}</li>
                        </ul>
                        <h4 class="mt-4 mb-3">Model Evaluation</h4>
                        <ul class="list-unstyled ps-3">
                            <li><strong>Corrected Velocity MAE:</strong> ${fmt(evalData.Corrected_Velocity_MAE)}</li>
                            <li><strong>Corrected Velocity RMSE:</strong> ${fmt(evalData.Corrected_Velocity_RMSE)}</li>
                            <li><strong>Test Set MAE:</strong> ${fmt(evalData.Test_Set_MAE)}</li>
                            <li><strong>Test Set RMSE:</strong> ${fmt(evalData.Test_Set_RMSE)}</li>
                            <li><strong>IoU Accuracy:</strong> ${fmt(evalData.IoU_Accuracy)}</li>
                        </ul>
                    `;
                    
                    resultsJson.innerHTML = customResultsHtml;

                    if (data.plot_image_base64) {
                        plotImage.src = "data:image/png;base64," + data.plot_image_base64;
                    }

                } catch (error) {
                    console.error("Error uploading files:", error);
                    messageDiv.textContent = "Error uploading files: " + error;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

# -------------------------------------------
# 6) Simple Home Route (optional)
# -------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return """
    <h1 class="text-center">Welcome to Velocity Processing!</h1>
    <p class="text-center"><a href="/upload">Go to Upload Page</a></p>
    """

if __name__ == '__main__':
    app.run(debug=True)
