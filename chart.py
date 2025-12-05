import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import math
import numpy as np

def show():
    st.title(" Trực quan hóa Boxplot ")

    # Load data
    df = pd.read_csv("tyg6.csv")

    # ================================
    # Hàm remove outliers theo IQR
    # ================================
    def remove_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    # ==================================
    # BOX PLOT 1 — glucose / TG / TyG
    # ==================================
    st.header("Boxplot Chart 1")

    df_iqr = df.copy()
    df_iqr = remove_outliers_iqr(df_iqr, 'glucose_max')
    df_iqr = remove_outliers_iqr(df_iqr, 'triglycerides_max')
    df_iqr = remove_outliers_iqr(df_iqr, 'tyg_max')

    cols = ['glucose_max', 'triglycerides_max', 'tyg_max']
    titles = ['Glucose', 'Triglycerides', 'TyG']
    palette = sns.color_palette('Set2', n_colors=2)

    fig, axes = plt.subplots(ncols=3, figsize=(20, 6))

    for i in range(len(cols)):
        ax = axes[i]
        sns.boxplot(
            data=df_iqr,
            x='sepsis',
            y=cols[i],
            palette=palette,
            hue='sepsis',
            legend=False,
            ax=ax
        )
        ax.set_ylabel(titles[i])
        ax.set_xlabel('Sepsis')

    st.pyplot(fig)

    # ==================================
    # BOX PLOT 2 — 18 biomarkers
    # ==================================
    st.header("Boxplot Chart 2")

    cols2 = [
        'age','wbc','rbc','rdw','hemoglobin','platelets','aniongap','bicarbonate',
        'calcium','chloride','potassium','creatinine','alt','alp','ast','pt','ptt','inr'
    ]

    titles2 = [
        'Age','WBC','RBC','RDW','Hemoglobin','Platelets','Anion Gap','Bicarbonate',
        'Calcium','Chloride','Potassium','Creatinine','ALT','ALP','AST','PT','PTT','INR'
    ]

    df_iqr2 = df.copy()
    for col in cols2:
        df_iqr2 = remove_outliers_iqr(df_iqr2, col)

    ncols = 3
    nvars = len(cols2)
    nrows = math.ceil(nvars / ncols)

    fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes2 = np.array(axes2).flatten()

    for i in range(len(cols2)):
        ax = axes2[i]
        sns.boxplot(
            data=df_iqr2,
            x='sepsis',
            y=cols2[i],
            palette=palette,
            hue='sepsis',
            legend=False,
            ax=ax
        )
        ax.set_ylabel(titles2[i])
        ax.set_xlabel('Sepsis')

    plt.tight_layout()
    st.pyplot(fig2)











# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st
# import math
# import numpy as np

# def show():
#     df = pd.read_csv("tyg5.csv")
#     st.header("Boxplot Chart 1")

#     cols = ['glucose', 'triglycerides', 'tyg']
#     titles = ['Glucose', 'Triglycerides', 'TyG']
#     palette = sns.color_palette("Set2", 2)

#     # FIGURE 1
#     fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))

#     for i in range(3):
#         sns.boxplot(
#             data=df, x="sepsis", y=cols[i],
#             palette=palette, hue="sepsis",
#             ax=axes1[i], legend=False
#         )
#         axes1[i].set_xlabel("Sepsis")
#         axes1[i].set_ylabel(titles[i])

#     st.pyplot(fig1)

#     # ======================================================
#     st.header("Boxplot Chart 2")

#     cols2 = [
#         'age','wbc', 'rbc', 'rdw', 'hemoglobin',
#         'platelets', 'aniongap', 'bicarbonate', 'calcium', 'chloride',
#         'potassium', 'creatinine', 'glucose', 'triglycerides', 'alt', 'alp',
#         'ast', 'pt', 'ptt', 'inr'
#     ]

#     titles2 = [
#         'Age', 'WBC', 'RBC','RDW','Hemoglobin','Platelets', 'Anion Gap', 'Bicarbonate',
#         'Calcium', 'Chloride', 'Potassium', 'Creatinine', 'Glucose', 'Triglycerides',
#         'ALT', 'ALP', 'AST', 'PT', 'PTT', 'INR'
#     ]

#     ncols = 5
#     nvars = len(cols2)
#     nrows = math.ceil(nvars / ncols)
#     palette = sns.color_palette("Set2", 2)

#     # FIGURE 2
#     fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
#     axes2 = np.array(axes2).flatten()

#     for i in range(len(cols2)):
#         ax = axes2[i]
#         sns.boxplot(
#             data=df, x='sepsis', y=cols2[i],
#             palette=palette, hue='sepsis',
#             ax=ax, legend=False
#         )
#         ax.set_ylabel(titles2[i])

#     # Xóa subplot dư
#     for j in range(len(cols2), len(axes2)):
#         fig2.delaxes(axes2[j])

#     st.pyplot(fig2)



# import streamlit as st
# import joblib

# def show():
#     st.header("Boxplot Chart")
    
#     fig = joblib.load("boxplot_fig.pkl")
#     st.pyplot(fig)

#     st.header("Boxplot Chart")
    
#     fig = joblib.load("boxplot_fig_all.pkl")
#     st.pyplot(fig)