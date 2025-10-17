# 🪙 Cryptocurrency Price Forecasting Engine  
**Predicting Ethereum (ETH) prices using LSTM neural networks and on-chain Uniswap v3 data**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📘 Overview
This project presents a **deep learning–based approach** for forecasting **Ethereum (ETH)** prices using **Long Short-Term Memory (LSTM)** networks.  
Unlike traditional models that focus on a single cryptocurrency, this model integrates **multi-token on-chain data** (price and liquidity) retrieved from **Uniswap v3** via **The Graph API**, offering a comprehensive view of the decentralized market.

The system demonstrates how on-chain market data and advanced time-series models can be combined for **robust and interpretable price prediction**.

---

## 🚀 Key Features
- 🔗 **On-chain data integration:** fetches Uniswap v3 token data (price, liquidity, volume) using GraphQL via The Graph.  
- ⚙️ **Automated preprocessing:** handles missing data, interpolates values, and normalizes features.  
- 🧠 **Model comparison:** evaluates **LSTM**, **GRU**, and **Transformer** architectures.  
- 📊 **Performance metrics:** assesses results using RMSE, MSE, MAE, and R².  
- 🧾 **Experimental findings:** compares ETH-only vs multi-token datasets to evaluate the impact on accuracy.  

---

## 🧠 Methodology

### 1️⃣ Data Extraction
Data is collected automatically from **Uniswap v3’s subgraph** using **GraphQL** queries.  
The pipeline retrieves:
- Token prices  
- Liquidity and volume  
- Time-based pool data  

The results are cleaned, interpolated, and saved as structured **JSON files** for model training.

### 2️⃣ Feature Engineering
- Retains **High**, **Low**, and **Close** price features.  
- Removes irrelevant data such as raw transaction volume.  
- Standardizes prices for model input.  

### 3️⃣ Model Architectures
Three deep learning models were implemented:

| Model | Highlights |
|--------|-------------|
| **LSTM** | Best performance in learning temporal dependencies |
| **GRU** | Lightweight and efficient recurrent architecture |
| **Transformer (Encoder-only)** | Uses attention for contextual time-series representation |

### 4️⃣ Evaluation Metrics
| Metric | Description |
|---------|-------------|
| **MSE** | Mean Squared Error – measures average squared deviation |
| **RMSE** | Root Mean Squared Error – penalizes large prediction errors |
| **MAE** | Mean Absolute Error – robust to outliers |
| **R²** | Coefficient of Determination – how well predictions explain variance |

---

## 📊 Results Summary

| Model | Sequence Length | RMSE | MSE | MAE | R² |
|--------|----------------|------|------|------|----|
| **LSTM** | 5 | 🟢 **0.03907** | **0.00153** | **0.03360** | **0.89620** |
| GRU | 4 | 0.03848 | 0.00551 | 0.03321 | 0.89983 |
| Transformer | 5 | 0.04962 | 0.00246 | 0.03939 | 0.83260 |

> ✅ **Best Model:** LSTM with sequence length 5  
> Stable training, lowest error, and no overfitting observed.

---

## 🧩 Key Findings
- **ETH-only training** yields higher accuracy than multi-crypto datasets (ETH, BTC, LINK).  
- **LSTM** demonstrates superior performance in stability and interpretability.  
- **Transformer** models require larger batch sizes (≥8) to stabilize learning.  
- Using multiple token datasets **increases computation time** without improving results.

---

## 🔮 Future Improvements
- Incorporate **external features** such as macroeconomic indicators or social sentiment.  
- Explore **hybrid Transformer–LSTM** architectures for time-series forecasting.  
- Add **decoder layers** to Transformer models for full-sequence prediction.  
- Extend evaluation to **multi-asset prediction** in DeFi markets.

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.x |
| **Libraries** | TensorFlow, Keras, NumPy, Pandas, Matplotlib |
| **APIs** | GraphQL via The Graph |
| **Data Source** | Uniswap v3 Subgraph |
| **Modeling** | LSTM, GRU, Transformer |
| **Visualization** | Matplotlib, Seaborn |

---

## ⚙️ Installation & Usage

### 🧩 Requirements
Ensure you have the following installed:
- Python 3.8+
- pip

Then install dependencies:
```bash
pip install -r requirements.txt


▶️ Running the Model
To fetch and preprocess data:

python src/data_fetch.py
python src/preprocess.py

To train the LSTM model:
python src/train_lstm.py


To evaluate the trained model:
python src/evaluate.py

👥 Authors
Western University — Department of Electrical and Computer Engineering
Name	Email
James Zhong	pzhong22@uwo.ca
Xi Feng	xfeng269@uwo.ca
Wenyi Yao	wyao45@uwo.ca
Yanhua Zhang	yzha5778@uwo.ca
Zelin Zhang	zzha973@uwo.ca

📚 Citation
If you use this work in your research, please cite:
Zhang, Y., Zhong, J., Feng, X., Yao, W., & Zhang, Z. (2025).
Cryptocurrency Price Forecasting Engine Using LSTM with On-Chain Uniswap Data.
Western University, London, Canada.
