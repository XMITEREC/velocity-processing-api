# ... same imports and global definitions as before ...

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
