# Log Anomaly Detection System

A machine learning-based system for detecting anomalies in system logs using LSTM-based neural networks.

## Overview

This project implements three different deep learning models for log anomaly detection:

1. **LSTM Autoencoder**: Learns to reconstruct normal log patterns and identifies anomalies based on reconstruction error.
2. **LSTM Sequence Predictor**: Predicts the next log entry in a sequence and flags unexpected patterns as anomalies.
3. **LSTM Classifier**: A semi-supervised classifier that identifies anomalous log patterns.

The system also uses an ensemble approach that combines predictions from all three models using majority voting.

## Features

- Log parsing and template extraction using [Drain3](https://github.com/logpai/Drain3)
- TF-IDF vectorization of log templates
- Multiple LSTM-based anomaly detection models
- Ensemble model combining multiple detection techniques
- Support for both Windows and Linux logs
- Comprehensive results analysis and visualization

## Installation

### Prerequisites

- Python 3.11.9 (required, other versions may cause compatibility issues)
- pip (latest version recommended)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nandini1081/log-anomaly-detection.git
cd log-anomaly-detection
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Handling Dependency Issues

If you encounter dependency issues, try:

For ml-dtypes errors (common on Windows):
```bash
pip install ml-dtypes==0.5.0 --no-build-isolation
```

For TensorFlow  errors, try:

```bash
# For CPU-only TensorFlow
pip install "tensorflow<2.13.0"

# For GPU support
pip install "tensorflow-gpu<2.13.0"
```

## Usage

### Basic Usage

1. Place your log files in the appropriate directory:
   - Linux logs: `data/raw/linux/sample_log.txt`
   - Windows logs: `data/raw/windows/sample_log.txt`

2. Run the main script:
```bash
python main.py
```

3. Check results in the `data/results/` directory

### Advanced Usage

The system can be customized by modifying parameters in the scripts:

- Adjust anomaly detection thresholds in each model class
- Modify sequence length for the Sequence Predictor
- Change the TF-IDF vectorization parameters

## Project Structure

- `models/`: Neural network model definitions
  - `autoencoder.py`: LSTM Autoencoder implementation
  - `classifier.py`: LSTM Classifier implementation
  - `predictor.py`: LSTM Sequence Predictor implementation
- `config/`: Configuration files including Drain3 settings
- `data/`: Data directories
- `main.py`: Main execution script

## Results

The system generates several output files:

- Model weights (`.keras` files)
- Anomaly detection results (`.npy` files)
- Performance metrics (`.json` and `.pkl` files)
- Summary analysis (`.csv` and `.txt` files)



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Drain3](https://github.com/logpai/Drain3) for log parsing
- TensorFlow and Keras for deep learning implementation
