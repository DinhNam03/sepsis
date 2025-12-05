import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
import pandas as pd
from pandas import read_csv
import joblib
import os

def show():

    st.title("So sánh ROC giữa các mô hình — Sepsis")
    BASE_DIR = os.path.dirname(__file__)

    @st.cache_resource
    def load_data():
        return read_csv(os.path.join(BASE_DIR, 'X_test.csv')), read_csv(os.path.join(BASE_DIR, 'y_test.csv'))

    @st.cache_resource
    def load_models():
        return {
        'Ada Boost': joblib.load(os.path.join(BASE_DIR, 'AdaBoost.joblib')),
        'Extra Trees': joblib.load(os.path.join(BASE_DIR, 'ExtraTrees.joblib')),
        'Gradient Boosting': joblib.load(os.path.join(BASE_DIR, 'GradientBoosting.joblib')),
        'Random Forest': joblib.load(os.path.join(BASE_DIR, 'RandomForest.joblib'))
    }

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
