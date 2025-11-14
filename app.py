import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Page setup
st.set_page_config(page_title="Jet Engine RUL Predictor", page_icon="✈️", layout="wide")
st.title("🚀 Jet Engine RUL Predictor")
st.markdown("Predict Remaining Useful Life (RUL) of turbofan engines using sensor data. ✈️🔧")

# Load model
model = joblib.load("random_forest_model.pkl")

# Column names
column_names = [
    'unitNumber', 'timeCycles',
    'opSetting1', 'opSetting2', 'opSetting3',
    'Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5',
    'Sensor6', 'Sensor7', 'Sensor8', 'Sensor9', 'Sensor10',
    'Sensor11', 'Sensor12', 'Sensor13', 'Sensor14', 'Sensor15',
    'Sensor16', 'Sensor17', 'Sensor18', 'Sensor19', 'Sensor20', 'Sensor21'
]

# Load test data
df_test = pd.read_csv("dataset/test_FD001.txt", sep=r"\s+", header=None).iloc[:, :26]
df_test.columns = column_names
df_test.drop(['opSetting1', 'opSetting2', 'opSetting3'], axis=1, inplace=True)

# Engine selector
engine_id = st.selectbox("🔍 Select Engine ID", sorted(df_test["unitNumber"].unique()))

# Last cycle data
last_cycle = df_test[df_test["unitNumber"] == engine_id].iloc[-1]
features = last_cycle.drop(["unitNumber", "timeCycles"])
rul_pred = model.predict([features])[0]

# Display prediction
st.subheader("📟 Sensor Readings at Last Cycle")
st.dataframe(last_cycle.to_frame().T)
st.metric("🎯 Predicted RUL", f"{rul_pred:.2f} cycles")

# Evaluation setup
X_test = df_test.groupby('unitNumber').last().drop(['timeCycles'], axis=1)
rul_test = pd.read_csv("dataset/RUL_FD001.txt", sep=r"\s+", header=None)
rul_test.columns = ['RUL']
actual_rul = rul_test['RUL']
predicted_rul = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(actual_rul, predicted_rul))
r2 = r2_score(actual_rul, predicted_rul)

# Show metrics
st.markdown("---")
st.subheader("📊 Model Evaluation")
col1, col2 = st.columns(2)
col1.metric("📉 RMSE", f"{rmse:.2f}")
col2.metric("📈 R² Score", f"{r2:.2f}")

# Feature importance chart (Top 10)
with st.expander("🧠 Top 10 Feature Importance"):
    fig1, ax1 = plt.subplots()
    importances = model.feature_importances_
    feature_names = X_test.columns
    sorted_idx = np.argsort(importances)[::-1][:10]
    ax1.barh(feature_names[sorted_idx], importances[sorted_idx])
    ax1.set_title("Top 10 Feature Importance")
    ax1.invert_yaxis()
    st.pyplot(fig1)

# True vs Predicted RUL chart
with st.expander("🎯 True vs Predicted RUL"):
    fig2, ax2 = plt.subplots()
    ax2.scatter(actual_rul, predicted_rul, alpha=0.7)
    ax2.plot([actual_rul.min(), actual_rul.max()], [actual_rul.min(), actual_rul.max()], 'r--', label="Ideal")
    ax2.set_xlabel("Actual RUL")
    ax2.set_ylabel("Predicted RUL")
    ax2.set_title("True vs Predicted RUL")
    ax2.legend()
    st.pyplot(fig2)