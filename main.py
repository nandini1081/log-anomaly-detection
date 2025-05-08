import os
import pandas as pd
import numpy as np
import pickle
import json
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from models.autoencoder import AutoencoderModel
from models.predictor import SequencePredictor
from models.classifier import ClassifierModel

def create_sequences(data, seq_length=3):
    """
    Create input sequences and target values for sequence prediction.
    
    Args:
        data: Input data array
        seq_length: Length of each sequence
        
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def load_log_data(log_path, output_file=None):
    """
    Load and parse log data using Drain3 template miner.
    
    Args:
        log_path: Path to the log file
        output_file: Path to save the processed logs DataFrame
        
    Returns:
        DataFrame containing parsed log templates
    """
    print(f"📥 Loading log file from: {log_path}")
    
    # Configure Drain3 template miner
    config = TemplateMinerConfig()
    # Try multiple possible paths for the drain3.ini file
    config_paths = [
        "drain3.ini",
        os.path.join("Drain3", "examples", "drain3.ini"),
        os.path.join("config", "drain3.ini")
    ]
    
    config_loaded = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"Loading Drain3 configuration from: {config_path}")
            config.load(config_path)
            config_loaded = True
            break
    
    if not config_loaded:
        print("Warning: Drain3 configuration file not found. Using default settings.")
    
    template_miner = TemplateMiner(config=config)

    
    logs = []
    log_timestamps = []
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
            for line_num, line in enumerate(file):
                try:
                    result = template_miner.add_log_message(line.strip())
                    if result is not None:
                        logs.append(result["template_mined"])
                        log_timestamps.append(result.get("timestamp", None))
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return pd.DataFrame(columns=["log_template", "timestamp"])
    except Exception as e:
        print(f"Error reading log file: {e}")
        return pd.DataFrame(columns=["log_template", "timestamp"])
    
    if not logs:
        print("Warning: No logs were successfully parsed!")
        return pd.DataFrame(columns=["log_template", "timestamp"])
    
    df = pd.DataFrame({
        "log_template": logs,
        "timestamp": log_timestamps
    })
    
    # Save processed logs if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved processed logs to: {output_file}")
    
    return df

def encode_logs(logs, method='tfidf', output_file=None):
    """
    Encode log templates using TF-IDF.
    
    Args:
        logs: DataFrame containing log templates
        method: Encoding method (only 'tfidf' supported now)
        output_file: Path to save the encoded data and vectorizer
        
    Returns:
        X: Encoded log data
        vectorizer: Fitted vectorizer object
    """
    print(f"📊 Encoding logs using {method.upper()}")
    
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(logs["log_template"]).toarray()
    
    # Save encoded data if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(f"{output_file}_data.npy", X)
        print(f"Saved encoded data to: {output_file}_data.npy")
        with open(f"{output_file}_vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Saved vectorizer to: {output_file}_vectorizer.pkl")
    
    return X, vectorizer

def generate_synthetic_labels(X, anomaly_ratio=0.1, output_file=None):
    """
    Generate synthetic labels for evaluation using LSTM-based autoencoder.
    
    Args:
        X: Input data
        anomaly_ratio: Ratio of data points to label as anomalies
        output_file: Path to save the labels
        
    Returns:
        Binary labels array (1 for anomaly, 0 for normal)
    """
    print("Generating synthetic labels for evaluation")
    
    input_dim = X.shape[1]
    
    # Simple autoencoder for reconstruction error
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, RepeatVector
    
    temp_model = Sequential([
        LSTM(32, input_shape=(1, input_dim)),
        RepeatVector(1),
        LSTM(32, return_sequences=True),
        Dense(input_dim)
    ])
    temp_model.compile(optimizer='adam', loss='mse')
    
    # Reshape for LSTM
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Train autoencoder
    temp_model.fit(X_reshaped, X_reshaped, epochs=5, batch_size=32, verbose=0)
    
    # Calculate reconstruction error
    predictions = temp_model.predict(X_reshaped, verbose=0)
    mse = np.mean(np.square(X_reshaped - predictions), axis=(1, 2))
    
    # Set threshold to identify the top anomaly_ratio of points as anomalies
    threshold = np.percentile(mse, 100 * (1 - anomaly_ratio))
    labels = (mse > threshold).astype(int)
    
    # Save labels if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, labels)
        print(f"Saved synthetic labels to: {output_file}")
    
    return labels

def run_models(X, os_type, output_dir="data/results"):
    """
    Run all anomaly detection models and save results.
    
    Args:
        X: Encoded log data
        os_type: Operating system type ('windows' or 'linux')
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing model results
    """
    print(f"\n🔎 Running models for {os_type.upper()} logs with TF-IDF encoding")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    input_dim = X.shape[1]
    
    # Train-test split with temporal order preserved
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=False)
    
    # Save train/test split
    train_file = f"{output_dir}/{os_type}_tfidf_X_train.npy"
    test_file = f"{output_dir}/{os_type}_tfidf_X_test.npy"
    np.save(train_file, X_train)
    print(f"Saved training data to: {train_file}")
    np.save(test_file, X_test)
    print(f"Saved test data to: {test_file}")
    
    # Generate synthetic labels for test set evaluation
    y_test_file = f"{output_dir}/{os_type}_tfidf_y_test.npy"
    y_test = generate_synthetic_labels(
        X_test, 
        anomaly_ratio=0.1,
        output_file=y_test_file
    )
    
    # Autoencoder
    print("\n🔧 LSTM Autoencoder:")
    ae_model = AutoencoderModel(input_dim)
    ae_history = ae_model.train(X_train, epochs=15, batch_size=32)
    ae_anomalies, ae_scores = ae_model.detect_anomalies(X_test)
    ae_accuracy = accuracy_score(y_test, ae_anomalies)
    print(f"Detected {sum(ae_anomalies)} anomalies | Accuracy: {ae_accuracy:.4f}")
    
    # Save autoencoder model and results
    ae_model_file = f"{output_dir}/{os_type}_tfidf_autoencoder_model.keras"
    ae_anomalies_file = f"{output_dir}/{os_type}_tfidf_autoencoder_anomalies.npy"
    ae_scores_file = f"{output_dir}/{os_type}_tfidf_autoencoder_scores.npy"
    
    try:
        ae_model.autoencoder.save(ae_model_file)
        print(f"Saved autoencoder model to: {ae_model_file}")
    except Exception as e:
        print(f"Error saving autoencoder model: {e}")
    
    np.save(ae_anomalies_file, ae_anomalies)
    print(f"Saved autoencoder anomalies to: {ae_anomalies_file}")
    np.save(ae_scores_file, ae_scores)
    print(f"Saved autoencoder scores to: {ae_scores_file}")
    
    # LSTM Sequence Predictor
    print("\n🔧 LSTM Sequence Predictor:")
    sp_model = SequencePredictor(input_dim)
    
    # Create sequence data
    X_seq, y_seq = create_sequences(X_train, seq_length=3)
    
    # Train sequence predictor
    sp_history = sp_model.train(X_seq, y_seq, epochs=15, batch_size=32)
    
    # Create test sequences
    X_seq_test, y_seq_test = create_sequences(X_test, seq_length=3)
    
    # Generate synthetic labels for sequence data (adjust for sequence length)
    y_test_seq = y_test[3:]  # Align with sequence length
    if len(y_test_seq) > len(X_seq_test):
        y_test_seq = y_test_seq[:len(X_seq_test)]
    
    # Save sequence data
    seq_test_file = f"{output_dir}/{os_type}_tfidf_X_seq_test.npy"
    seq_test_labels_file = f"{output_dir}/{os_type}_tfidf_y_seq_test.npy"
    seq_labels_file = f"{output_dir}/{os_type}_tfidf_y_test_seq.npy"
    np.save(seq_test_file, X_seq_test)
    print(f"Saved sequence test data to: {seq_test_file}")
    np.save(seq_test_labels_file, y_seq_test)
    print(f"Saved sequence test labels to: {seq_test_labels_file}")
    np.save(seq_labels_file, y_test_seq)
    print(f"Saved sequence synthetic labels to: {seq_labels_file}")
    
    # Detect anomalies
    sp_anomalies, sp_scores = sp_model.detect_anomalies(X_seq_test, y_seq_test)
    sp_accuracy = accuracy_score(y_test_seq, sp_anomalies) if len(sp_anomalies) > 0 else 0.0
    print(f"Detected {sum(sp_anomalies)} anomalies | Accuracy: {sp_accuracy:.4f}")
    
    # Save sequence predictor model and results
    sp_model_file = f"{output_dir}/{os_type}_tfidf_sequence_predictor_model.keras"
    sp_anomalies_file = f"{output_dir}/{os_type}_tfidf_sequence_predictor_anomalies.npy"
    sp_scores_file = f"{output_dir}/{os_type}_tfidf_sequence_predictor_scores.npy"
    
    try:
        sp_model.model.save(sp_model_file)
        print(f"Saved sequence predictor model to: {sp_model_file}")
    except Exception as e:
        print(f"Error saving sequence predictor model: {e}")
    
    np.save(sp_anomalies_file, sp_anomalies)
    print(f"Saved sequence predictor anomalies to: {sp_anomalies_file}")
    np.save(sp_scores_file, sp_scores)
    print(f"Saved sequence predictor scores to: {sp_scores_file}")
    
    # LSTM-based Classifier
    print("\n🔧 LSTM-based Classifier:")
    clf_model = ClassifierModel(input_dim)
    clf_history = clf_model.fit(X_train)
    clf_anomalies = clf_model.predict(X_test)
    clf_accuracy = accuracy_score(y_test, clf_anomalies)
    print(f"Detected {sum(clf_anomalies)} anomalies | Accuracy: {clf_accuracy:.4f}")
    
    # Save classifier model and results
    clf_model_file = f"{output_dir}/{os_type}_tfidf_classifier_model.keras"
    clf_anomalies_file = f"{output_dir}/{os_type}_tfidf_classifier_anomalies.npy"
    
    try:
        clf_model.model.save(clf_model_file)
        print(f"Saved classifier model to: {clf_model_file}")
    except Exception as e:
        print(f"Error saving classifier model: {e}")
    
    np.save(clf_anomalies_file, clf_anomalies)
    print(f"Saved classifier anomalies to: {clf_anomalies_file}")
    
    # Calculate ensemble predictions (majority voting)
    seq_indices = range(len(X_test) - 3)
    ensemble_results = np.zeros(len(X_test))
    
    # Count votes for each sample
    for i in range(len(X_test)):
        votes = 0
        count = 0
        
        # Vote from autoencoder
        votes += ae_anomalies[i]
        count += 1
        
        # Vote from classifier
        votes += clf_anomalies[i]
        count += 1
        
        # Vote from sequence predictor (if available)
        if i in seq_indices:
            seq_idx = i - seq_indices[0]
            if seq_idx < len(sp_anomalies):
                votes += sp_anomalies[seq_idx]
                count += 1
        
        # Mark as anomaly if majority of available models vote for it
        ensemble_results[i] = votes >= (count / 2)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_results)
    print("\n🔧 Ensemble Model (Majority Voting):")
    print(f"Detected {int(sum(ensemble_results))} anomalies | Accuracy: {ensemble_accuracy:.4f}")
    
    # Save ensemble results
    ensemble_file = f"{output_dir}/{os_type}_tfidf_ensemble_results.npy"
    np.save(ensemble_file, ensemble_results)
    print(f"Saved ensemble results to: {ensemble_file}")
    
    # Save all results to summary files
    results = {
        "LSTM Autoencoder": {"anomalies": int(sum(ae_anomalies)), "accuracy": float(ae_accuracy)},
        "LSTM Sequence Predictor": {"anomalies": int(sum(sp_anomalies)) if len(sp_anomalies) > 0 else 0, "accuracy": float(sp_accuracy)},
        "LSTM Classifier": {"anomalies": int(sum(clf_anomalies)), "accuracy": float(clf_accuracy)},
        "Ensemble (Majority)": {"anomalies": int(sum(ensemble_results)), "accuracy": float(ensemble_accuracy)}
    }
    
    # Save as pickle
    results_pkl_file = f"{output_dir}/{os_type}_tfidf_results.pkl"
    with open(results_pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results pickle to: {results_pkl_file}")
    
    # Save as JSON for human readability
    results_json_file = f"{output_dir}/{os_type}_tfidf_results.json"
    with open(results_json_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results JSON to: {results_json_file}")
    
    return results

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/encoded", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    print("Created directories: data/processed, data/encoded, data/results")
    
    # Define log files - can be a single file or multiple by OS
    log_files = {
        "linux": "data/raw/linux/sample_log.txt",
        "windows": "data/raw/windows/sample_log.txt"
    }
    
    # Fallback to single file if the OS-specific files don't exist
    if not os.path.exists(log_files["linux"]) and not os.path.exists(log_files["windows"]):
        default_log_file = "data/sample_log.txt"
        if os.path.exists(default_log_file):
            print(f"OS-specific log files not found. Using default log file: {default_log_file}")
            log_files = {"default": default_log_file}
        else:
            print("ERROR: No log files found!")
            exit(1)
    
    # Process each log file
    all_results = {}
    
    for os_type, log_file in log_files.items():
        print(f"\n=== Processing {os_type.upper()} logs ===")
        
        # Skip if file doesn't exist
        if not os.path.exists(log_file):
            print(f"WARNING: Log file for {os_type} not found at {log_file}. Skipping.")
            continue
        
        try:
            # Load and process logs
            logs_df = load_log_data(
                log_file, 
                output_file=f"data/processed/{os_type}_processed_logs.csv"
            )
            
            if logs_df.empty:
                print(f"ERROR: No valid logs found for {os_type}. Skipping.")
                continue
            
            print(f"\n✅ Log data loaded. Sample:")
            print(logs_df.head())
            
            # Encode logs (using only TF-IDF as requested)
            X_encoded, vectorizer = encode_logs(
                logs_df, 
                method='tfidf',
                output_file=f"data/encoded/{os_type}_tfidf"
            )
            
            # Run models and get results
            model_results = run_models(
                X_encoded, 
                os_type=os_type,
                output_dir="data/results"
            )
            
            all_results[os_type] = model_results
            
        except Exception as e:
            print(f"ERROR processing {os_type} logs: {e}")
            continue
    
    # Compare Results across all OS types
    if all_results:
        print("\n📊 Comparative Analysis:")
        
        # Create summary DataFrame
        summary_data = {
            "OS": [],
            "Model": [],
            "Anomalies": [],
            "Accuracy": []
        }
        
        for os_type, os_results in all_results.items():
            for model_name, metrics in os_results.items():
                summary_data["OS"].append(os_type)
                summary_data["Model"].append(model_name)
                summary_data["Anomalies"].append(metrics["anomalies"])
                summary_data["Accuracy"].append(metrics["accuracy"])
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary results
        summary_file = "data/results/all_results_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved full results summary to: {summary_file}")
        print("\nFull Results Summary:")
        print(summary_df)
        
        # Calculate best performing model
        best_accuracy = summary_df.loc[summary_df["Accuracy"].idxmax()]
        print(f"\n📈 Summary:")
        print(f"Best Model by Accuracy:")
        print(f"  OS: {best_accuracy['OS']}")
        print(f"  Model: {best_accuracy['Model']}")
        print(f"  Accuracy: {best_accuracy['Accuracy']:.4f}")
        
        # Calculate average performance by OS
        print(f"\nAverage Performance by OS:")
        avg_by_os = summary_df.groupby("OS")["Accuracy"].mean().reset_index()
        print(avg_by_os)
        
        # Save the final summary
        final_summary_file = "data/results/final_summary.txt"
        with open(final_summary_file, "w") as f:
            f.write("Log Anomaly Detection - Final Summary\n")
            f.write("===================================\n\n")
            f.write("Best Model by Accuracy:\n")
            f.write(f"  OS: {best_accuracy['OS']}\n")
            f.write(f"  Model: {best_accuracy['Model']}\n")
            f.write(f"  Accuracy: {best_accuracy['Accuracy']:.4f}\n\n")
            f.write("Average Performance by OS:\n")
            f.write(str(avg_by_os))
        print(f"Saved final summary to: {final_summary_file}")
        
        print("\n=== Analysis complete! All results saved to data/results/ ===")
    else:
        print("\nERROR: No results were generated. Please check the log files and try again.")