## 🧠 AI Stock Trend Predictor — Robust Version

This project implements a comprehensive financial dashboard using advanced machine learning models (LSTM and SVM) to predict short-term stock price movements and provide deep explainability through SHAP and LIME.

The application is built using Streamlit, making it easy to interact with and visualize the results directly in a web browser.

# ✨ Features

Dual-Model Prediction:

LSTM Regression: Predicts the Next Day's Close Price by modeling the daily percentage return (change) for increased stability.

SVM Classification: Predicts the Next Day's Intraday Trend (Close Price vs. Open Price).

Explainability: Provides detailed insights into model decisions.

SHAP (SHapley Additive exPlanations): Explains which features (e.g., MACD, RSI, Volume) contributed most to the LSTM's percentage change prediction.

LIME (Local Interpretable Model-agnostic Explanations): Provides a local, visual explanation for why the SVM predicted an UP or DOWN trend for the latest available day. 

Recursive Forecasting: Generates multi-day future predictions by iteratively feeding previous predictions back into the model to recalculate technical indicators, offering a simulated multi-step view.

Portfolio Simulation: Simple simulation based on the multi-day forecast.

Data Visualization: Displays historical price charts and an overlay of the historical data vs. the multi-day forecast.

# 🚀 Getting Started

Follow these steps to set up and run the application locally.

Prerequisites

You need Python installed (Python 3.8+ is recommended).

1. Clone the Repository

If you haven't already, clone this repository to your local machine:

git clone [https://github.com/CoderAD4/AI_Proj.git](https://github.com/CoderAD4/AI_Proj.git)
cd AI_Proj


2. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

#Create the environment
python -m venv venv

#Activate the environment
#On Windows PowerShell:
.\venv\Scripts\Activate
#On Linux/macOS or Git Bash:
source venv/bin/activate


3. Install Dependencies

Install all required packages listed in requirements.txt:

pip install -r requirements.txt


(Dependencies include: streamlit, yfinance, pandas, numpy, tensorflow, shap, lime, etc.)

4. Run the Application

Execute the Streamlit application from the root directory:

streamlit run mainai.py


The application will open automatically in your web browser (usually at http://localhost:8501).

# ⚙️ Project Architecture

The core of the system uses a dual-model approach optimized for financial time series prediction:

Data Preparation: Historical stock data (fetched via yfinance) is cleaned and augmented with standard Technical Indicators (RSI, MACD, Moving Averages).

Target Transformation: The absolute price series is non-stationary. To make the prediction task easier and more stable, the LSTM is trained to predict the percentage change ($\text{Close}_t / \text{Close}_{t-1}$) instead of the raw price.

LSTM Model: A Long Short-Term Memory (LSTM) recurrent neural network processes windows of historical features (e.g., last 60 days of data) to forecast the next day's percentage return.

SVM Model: A Support Vector Machine (SVM) classifies the overall direction ($\text{Close}>\text{Open}$ or $\text{Close}\leq\text{Open}$) using the static features of the latest day.

Explainability Pipeline:

SHAP (LSTM): Uses the Kernel Explainer approach (for stability) to show the aggregate feature contribution to the predicted percentage change.

LIME (SVM): Generates a local linear model to explain the final classification decision.

# ⚠️ Important Note on Stability

This project uses advanced machine learning techniques (TensorFlow/Keras and SHAP) in a rapidly re-running environment (Streamlit). If you encounter the LookupError: gradient registry has no entry for: shap_DivNoNan error when changing parameters:

The code contains internal fixes (localizing SHAP imports and using tf.keras.backend.clear_session()) designed to prevent this.

If it persists, a full terminal restart is usually required, as the issue stems from deep library conflicts.

# 🤝 Contribution and Disclaimer

This project is a prototype for educational and demonstrative purposes only. It is NOT financial advice. Stock market predictions are inherently uncertain, and historical performance is not indicative of future results.
