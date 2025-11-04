📈 Stock Price Prediction using LSTM 🤖

This project predicts future stock prices using a **Long Short-Term Memory (LSTM)** neural network model trained on historical stock data. 
It demonstrates the use of **deep learning** for time series forecasting, data preprocessing, and visualization of financial trends.

─────────────────────────────
🎯 Overview:
The main goal of this project is to analyze stock price movements and predict future prices based on past performance.
The model was trained on **Google (GOOG)** stock data from **2012 to 2022** using **LSTM (RNN)** – a network well-suited for sequential data.

─────────────────────────────
⚙️ Tech Stack:
- 🐍 Python
- 📚 Libraries: NumPy, Pandas, Matplotlib, yfinance, scikit-learn, TensorFlow/Keras, Streamlit (for deployment)

─────────────────────────────
✨ Features:
✅ Fetches real-time stock data using Yahoo Finance API  
✅ Visualizes 100-day and 200-day moving averages  
✅ Performs data preprocessing and scaling using MinMaxScaler  
✅ Creates time-sequenced training data for LSTM input  
✅ Builds a multi-layer LSTM model with dropout regularization  
✅ Predicts and visualizes actual vs predicted stock prices  
✅ Saves the trained model for reuse or deployment  

─────────────────────────────
🧩 Model Architecture:

Layer Type | Units | Activation | Return Sequences | Dropout
------------|--------|-------------|------------------|----------
LSTM | 50 | ReLU | True | 0.2
LSTM | 60 | ReLU | True | 0.3
LSTM | 80 | ReLU | True | 0.4
LSTM | 120 | ReLU | False | 0.5
Dense | 1 | Linear | — | —

🧠 Loss Function: Mean Squared Error  
⚡ Optimizer: Adam  
⏳ Epochs: 50  
📦 Batch Size: 32  

─────────────────────────────
💻 Installation & Setup:

1️⃣ Clone the repository  
   git clone https://github.com/yourusername/Stock-Price-Prediction-LSTM.git  
   cd Stock-Price-Prediction-LSTM  

2️⃣ Install dependencies  
   pip install -r requirements.txt  

3️⃣ Run the Python script  
   python stock_prediction.py  

4️⃣ (Optional) Launch Streamlit app  
   streamlit run app.py  

─────────────────────────────
📂 File Structure:

📁 Stock-Price-Prediction-LSTM  
│
├── stock_prediction.py            → Main model script  
├── app.py                         → Streamlit web app (optional)  
├── requirements.txt               → Dependencies  
├── Stock Predictions Model.keras  → Saved trained model  
├── README.txt                     → Project documentation  
└── data/                          → (Optional) CSV data folder  

─────────────────────────────
🚀 Working Steps:

1️⃣ Data Collection – Fetch historical stock prices using yfinance.  
2️⃣ Data Visualization – Plot 100 & 200-day moving averages.  
3️⃣ Data Preprocessing – Scale data and split into training/testing sets.  
4️⃣ Sequence Creation – Create 100-day sequences for LSTM input.  
5️⃣ Model Training – Train a stacked LSTM network to minimize loss.  
6️⃣ Prediction – Compare predicted vs actual closing prices.  
7️⃣ Deployment – Save model and optionally deploy via Streamlit.  

─────────────────────────────
📊 Results:
The model successfully captures stock price trends, providing close approximations of actual movements. 
While not a financial advisor tool, it demonstrates the **potential of deep learning** for time-series forecasting.

─────────────────────────────
💡 Key Learnings:
🌱 Understanding LSTM networks for sequential/time-series data  
⚙️ Importance of data scaling and windowed sequence creation  
📈 Visualization of long-term stock trends using moving averages  
🧰 Hands-on experience with Keras, TensorFlow, and Streamlit  

─────────────────────────────
🚧 Future Enhancements:
🔹 Integrate real-time stock prediction dashboard  
🔹 Add multi-feature input (Open, High, Low, Volume)  
🔹 Improve accuracy with GRU or Bi-directional LSTM  
🔹 Incorporate sentiment analysis from financial news  

─────────────────────────────
👩‍💻 Author:
👤 Shaili Chauhan  
🏫 Graphic Era University  
 shailichauhan06052004@gmail.com

─────────────────────────────
⚠️ Disclaimer:
This project is for educational and research purposes only. 
Stock market predictions are uncertain and should not be used for actual trading decisions.
