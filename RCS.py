import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def format_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"{p:.3f}"

def show():

    st.title(" (TyG → Sepsis)")

    df = pd.read_csv("tyg1.csv")
 
    model_linear = smf.glm(
        formula="sepsis ~ tyg",
        data=df,
        family=sm.families.Binomial()
    )
    result_linear = model_linear.fit()

    model_linear = smf.glm(formula='sepsis ~ tyg', data=df, 
                       family = sm.families.Binomial())
    result_linear = model_linear.fit()

    n_knof = 3
    model_spline = smf.glm(formula=f'sepsis ~ bs(tyg, df = {n_knof}, include_intercept = False)', 
                        data=df, family=sm.families.Binomial())
    result_spline = model_spline.fit()


    LR = 2 * (result_spline.llf - result_linear.llf)
    df_diff = result_spline.df_model - result_linear.df_model
    p_nonlinear = chi2.sf(LR, df_diff)


    x_pred = pd.DataFrame({
        "tyg": np.linspace(df["tyg"].min(), df["tyg"].max(), 100)
    })

    result_pred = result_spline.get_prediction(x_pred)
    pred_mean = result_pred.predicted_mean
    ci = result_pred.conf_int()

    # Vẽ biểu đồ RCS
    fig = plt.figure(figsize=(10, 6))

    plt.plot(x_pred["tyg"], pred_mean, color="red", linewidth=2, label="Spline fitted")
    plt.fill_between(x_pred["tyg"], ci[:, 0], ci[:, 1], color="red", alpha=0.2)

    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(df["tyg"].median(), color="gray", linestyle="--", linewidth=0.7)

    plt.xlabel("TyG index", fontsize=12)
    plt.ylabel("Sepsis ", fontsize=12)
    plt.title("TyG vs Sepsis", fontsize=14)

    plt.text(
        0.98, 0.92,
        f"P overall < 0.001\nP non-linear = {format_p(p_nonlinear)}",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="right",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    st.pyplot(fig)

