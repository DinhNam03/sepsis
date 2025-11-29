import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import math
import numpy as np

def show():
    df = pd.read_csv("tyg.csv")
    st.header("Boxplot Chart 1")

    cols = ['glucose', 'triglycerides', 'tyg']
    titles = ['Glucose', 'Triglycerides', 'TyG']
    palette = sns.color_palette("Set2", 2)

    # FIGURE 1
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))

    for i in range(3):
        sns.boxplot(
            data=df, x="sepsis", y=cols[i],
            palette=palette, hue="sepsis",
            ax=axes1[i], legend=False
        )
        axes1[i].set_xlabel("Sepsis")
        axes1[i].set_ylabel(titles[i])

    st.pyplot(fig1)

    # ======================================================
    st.header("Boxplot Chart 2")

    cols2 = [
        'age','wbc', 'rbc', 'rdw', 'hemoglobin',
        'platelets', 'aniongap', 'bicarbonate', 'calcium', 'chloride',
        'potassium', 'creatinine', 'glucose', 'triglycerides', 'alt', 'alp',
        'ast', 'pt', 'ptt', 'inr'
    ]

    titles2 = [
        'Age', 'WBC', 'RBC','RDW','Hemoglobin','Platelets', 'Anion Gap', 'Bicarbonate',
        'Calcium', 'Chloride', 'Potassium', 'Creatinine', 'Glucose', 'Triglycerides',
        'ALT', 'ALP', 'AST', 'PT', 'PTT', 'INR'
    ]

    ncols = 5
    nvars = len(cols2)
    nrows = math.ceil(nvars / ncols)
    palette = sns.color_palette("Set2", 2)

    # FIGURE 2
    fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes2 = np.array(axes2).flatten()

    for i in range(len(cols2)):
        ax = axes2[i]
        sns.boxplot(
            data=df, x='sepsis', y=cols2[i],
            palette=palette, hue='sepsis',
            ax=ax, legend=False
        )
        ax.set_ylabel(titles2[i])

    # Xóa subplot dư
    for j in range(len(cols2), len(axes2)):
        fig2.delaxes(axes2[j])

    st.pyplot(fig2)



# import streamlit as st
# import joblib

# def show():
#     st.header("Boxplot Chart")
    
#     fig = joblib.load("boxplot_fig.pkl")
#     st.pyplot(fig)

#     st.header("Boxplot Chart")
    
#     fig = joblib.load("boxplot_fig_all.pkl")
#     st.pyplot(fig)