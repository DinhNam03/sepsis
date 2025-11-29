import streamlit as st
import matplotlib.pyplot as plt
import joblib

def show():
    st.title("ROC Curves for Sepsis Prediction")

    roc_results = joblib.load("roc_tyg.joblib")

    model1 = roc_results["model1"]
    model2 = roc_results["model2"]

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(model1["fpr"], model1["tpr"], color="black", lw=2,
            label=f"{model1['label']}\nAUC={model1['auc']:.3f} (95% CI {model1['ci'][0]:.3f}–{model1['ci'][1]:.3f})")
    
    ax.plot(model2["fpr"], model2["tpr"], color="red", lw=2,
            label=f"{model2['label']}\nAUC={model2['auc']:.3f} (95% CI {model2['ci'][0]:.3f}–{model2['ci'][1]:.3f})")

    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
