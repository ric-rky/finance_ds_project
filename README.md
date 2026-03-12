# Finance Data Science Project

A comprehensive quantitative finance analysis pipeline built with Python and PyTorch. This project walks through an end-to-end workflow — from exploratory data analysis to deep learning models, anomaly detection, and risk management — on daily OHLCV stock data.

## Overview

The notebook covers **499 trading days** (2022-01-04 to 2023-12-01) and implements:

| Analysis | Technique | Key Result |
|---|---|---|
| EDA | Price trends, distributions, correlation heatmap | Data understanding & feature selection |
| Technical Indicators | Bollinger Bands, SMA(20), EMA(20) | Visual trend & volatility bands |
| Classification | Feedforward NN (4 layers + BatchNorm + Dropout) | 52% test accuracy on direction prediction |
| Time-Series | LSTM (2 layers, hidden=64, seq=20) | MAE: 2.86 USD, RMSE: 3.72 USD |
| Anomaly Detection | Autoencoder (5→3→5 bottleneck) | 25 anomalies flagged (5% of days) |
| Risk Analysis | Historical VaR, Parametric VaR, CVaR | VaR: 2.73%, CVaR: 3.91% (95% confidence) |
| Explainability | SHAP (DeepExplainer), Permutation Importance | Low price & daily return most predictive |
| Simulation | Monte Carlo (500 paths × 60 days) | Future price distribution & percentiles |

## Project Structure

```
├── data/
│   ├── finance_sample.csv                  # Raw OHLCV dataset (500 rows)
│   └── finance_with_pytorch_predictions.csv # Exported model predictions
├── notebooks/
│   └── finance_data_science_project.ipynb  # Main analysis notebook (49 cells)
├── scripts/
│   └── finance_pytorch_model.pt            # Saved feedforward NN weights
├── .github/
│   └── copilot-instructions.md             # Copilot workspace instructions
├── requirements.txt
└── README.md
```

## Requirements

- **Python** 3.10+
- **PyTorch** 2.6+ (CPU is sufficient)
- **NumPy** 2.x

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural networks (feedforward, LSTM, autoencoder) |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, metrics, train/test split |
| `shap` | Model explainability (DeepExplainer) |
| `scipy` | Parametric VaR (normal distribution) |
| `ipywidgets` | Interactive hyperparameter widget |

## Getting Started

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook** in VS Code or JupyterLab:
   ```bash
   jupyter lab notebooks/finance_data_science_project.ipynb
   ```

4. **Run all cells** sequentially from top to bottom. The notebook is self-contained — all data loading, preprocessing, training, and evaluation happens in order.

## Notebook Sections

1. **Imports & Setup** — Load libraries and configure plotting
2. **Data Loading** — Read `finance_sample.csv`, inspect shape and types
3. **Data Cleaning** — Handle missing values, compute daily returns and direction labels
4. **Descriptive Statistics** — Summary stats, skewness, kurtosis
5. **EDA Visualizations** — Close price trend, return distribution, volume bars, correlation heatmap
6. **Feature Engineering** — Select OHLCV + daily return as model features
7. **PyTorch Classification NN** — Train a 4-layer feedforward network (150 epochs) to predict price direction
8. **Model Evaluation** — Accuracy, confusion matrix, classification report
9. **Hyperparameter Sensitivity** — Learning rate sweep with interactive widget
10. **Permutation Importance** — Rank feature contributions
11. **Predictions vs Actuals** — Scatter plot of predicted probabilities
12. **Model Export** — Save model weights (`.pt`) and predictions (`.csv`)
13. **Correlation Heatmap** — Visualize feature relationships
14. **Monte Carlo Simulation** — Simulate 1,000 future price paths over 252 days
15. **SHAP Explainability** — DeepExplainer waterfall/summary plots
16. **LSTM Model** — Train a recurrent network (300 epochs) for close price forecasting
17. **LSTM Loss Curve** — Training convergence visualization
18. **LSTM Predictions** — Test set actual vs predicted with MAE/RMSE
19. **Bollinger Bands** — Technical indicator overlay with volume subplot
20. **Autoencoder Anomaly Detection** — Train an autoencoder (200 epochs), flag top 5% reconstruction errors
21. **Anomaly Visualization** — Close price with anomalies highlighted + reconstruction error bars
22. **Value at Risk** — Historical VaR, Parametric VaR, CVaR with return distribution plot
23. **Conclusion** — Full metrics summary, key takeaways, and next steps

## Model Artifacts

| File | Description |
|---|---|
| `scripts/finance_pytorch_model.pt` | Saved state dict of the feedforward classification NN |
| `data/finance_with_pytorch_predictions.csv` | Dataset with appended `pytorch_prediction` column |

## License

This project is for educational and demonstration purposes.
