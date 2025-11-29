import streamlit as st
import joblib

def show():
    st.header("Boxplot Chart")
    
    fig = joblib.load("boxplot_fig.pkl")
    st.pyplot(fig)

    st.header("Boxplot Chart")
    
    fig = joblib.load("boxplot_fig_all.pkl")
    st.pyplot(fig)