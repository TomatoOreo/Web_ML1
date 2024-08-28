import pandas as pd
import numpy as np
from IPython.display import (display, display_html, display_png, display_svg)

from matplotlib import pyplot
# import matplotlib.pyplot as plt
pyplot.rcParams['font.sans-serif'] = ['Microsoft YaHei']
pyplot.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestClassifier
import shap

import joblib

import streamlit as st

# Load the model
model = joblib.load('rf.pkl')

# Define feature names
feature_names = ['CD4', 'CD4/CD8 ratio', 'CD8', 'HGB', 'WBC', 'TBIL',
                 'Age at ART initial', 'PLT', "VL"]
# Streamlit user interface
st.title("INR Predictor")

# age: numerical input
cd4 = st.number_input("CD4:")
# sex: categorical selection
cd4_cd8 = st.number_input("CD4/CD8 ratio:")
# cp: categorical selection
cd8 = st.number_input("CD8:")
# trestbps: numerical input
hgb = st.number_input("HGB:")
# chol: numerical input
wbc = st.number_input("WBC:")
# fbs: categorical selection
tbil = st.number_input("TBIL:")
# restecg: categorical selection
age = st.number_input("Age at ART initial:")
# thalach: numerical input
plt = st.number_input("PLT:")
# exang: categorical selection
vl = st.number_input("VL:")

# Process inputs and make predictions
feature_values = [cd4, cd4_cd8, cd8, hgb, wbc, tbil, age, plt, vl]
features = np.array([feature_values])
result = {1: "INR", 0: "Non-INR"}

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]

    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {result[predicted_class]}")

    st.write(f"**Based on feature values, predicted possibility of INR is:** {(predicted_proba[1] * 100):.1f}%")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.initjs()
    # fig = plt.figure()

    # if predicted_class == 1:
    #     shap.force_plot(explainer.expected_value[1], shap_values[1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True, show=False)
    # else:
    #     shap.force_plot(explainer.expected_value[0], shap_values[0],
    #                     pd.DataFrame([feature_values], columns=feature_names), matplotlib=True, show=False)
    shap.force_plot(explainer.expected_value[1], shap_values[1], pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True, show=False)
    pyplot.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")