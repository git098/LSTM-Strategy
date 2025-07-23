# LSTM-Based Algorithmic Trading Strategy

This repository contains an LSTM (Long Short-Term Memory) neural network model designed for algorithmic trading. The model is trained on historical stock data to predict future price movements, and its performance is evaluated through a backtesting simulation.

## Project Structure

```
gitpush/
├── data/                     # Contains historical stock data (e.g., COROMANDEL.csv, DABUR.csv)
├── models/                   # LSTM model definition (lstm_model.py)
├── notebooks/                # Jupyter notebooks for analysis and proof (model_performance.ipynb)
├── scripts/                  # Main training and backtesting script (train_lstm_strategy.py)
├── saved_models/             # Directory to store trained LSTM models (.h5 files)
├── utils/                    # Utility functions (data loading, indicator calculation, signal generation)
├── .gitignore                # Specifies intentionally untracked files to ignore
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
└── ... (other files like tests, backtests, etc.)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>/gitpush
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Strategy

To train the LSTM model on `COROMANDEL.csv` and backtest its performance on `DABUR.csv` (avoiding lookahead bias), run the main script:

```bash
python scripts/train_lstm_strategy.py
```

This script will:
*   Train the LSTM model using `data/COROMANDEL.csv`.
*   Save the trained model to `saved_models/coromandel_trained_model.h5`.
*   Load the saved model and make predictions on `data/DABUR.csv`.
*   Run a backtesting simulation based on these predictions.
*   Print a detailed performance report.

## Trading Performance (LSTM Model: Training on Coromandel, Test on Dabur):

```
--- LSTM Backtest Report (Train: COROMANDEL, Test: DABUR) ---
Initial Equity:       ₹100,000.00
Final Equity:         ₹155,106.55
Total Return:         55.11%
Max Drawdown:         -30.87%
Total Trades:         8550
Win Rate:             46.20%
------------------------------------------------------------
