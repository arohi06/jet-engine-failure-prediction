# ✈️ Jet Engine Failure Prediction using ML

This project uses the NASA C-MAPSS FD001 dataset to predict the Remaining Useful Life (RUL) of turbofan engines. We apply a Random Forest model combined with a modular preprocessing pipeline and feature importance analysis to build an interpretable and demo-ready machine learning solution.

---

## 📁 Dataset

- **Source**: NASA C-MAPSS FD001 subset  
- **Description**: Multivariate time-series data from 100 engines, each running until failure  
- **Features**: 3 operational settings + 21 sensor measurements per time cycle  
- **Target**: Remaining Useful Life (RUL) per engine

---

## 🎯 Project Goal

Predict the RUL of each engine at its last observed cycle using only sensor data, and evaluate model performance using RMSE and R² score.

---

## 🧠 Model

- **Algorithm**: Random Forest Regressor  
- **Why**: Fast, interpretable, and effective for tabular sensor data  
- **Evaluation Metrics**:  
  - Root Mean Squared Error (RMSE)  
  - Coefficient of Determination (R² Score)

---

## 🧪 Results

- **RMSE**: 20.76  
- **R² Score**:  0.75
- **Feature Importance**: Top 10 most influential sensors visualized using a horizontal bar chart
- **True vs Predicted RUL**: Scatter plot showing how closely the model's predictions match actual RUL values  


---

## 🧩 Key Features

- Modular preprocessing pipeline (grouping, filtering, feature selection)
- Engine-wise RUL prediction using last cycle data
- Feature importance chart for model transparency
- Clean, reproducible code structure

---

