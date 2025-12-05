import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
import pandas as pd
import joblib
import os

def show():

    st.title("So sánh ROC giữa các mô hình — Sepsis")

    @st.cache_resource
    def load_data():
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")
        return X_test, y_test

    @st.cache_resource
    def load_models():
        models = {
            "Ada Boost": joblib.load("AdaBoost.joblib"),
            "Extra Trees": joblib.load("ExtraTrees.joblib"),
            "Gradient Boosting": joblib.load("GradientBoosting.joblib"),
            "Random Forest": joblib.load("RandomForest.joblib"),
        }
        return models

    X_test, y_test = load_data()
    base_models = load_models()

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in base_models.items():

        pred = model.predict_proba(X_test)[:, 1]
        roc_auc = auc(*roc_curve(y_test, pred)[:2])

        RocCurveDisplay.from_predictions(
            y_test, pred, ax=ax, name=f"{name} (AUC = {roc_auc:.3f})"
        )

    ax.set_title("ROC — Sepsis", fontsize=16)
    ax.grid(linestyle="--")
    plt.tight_layout()

    st.pyplot(fig)
