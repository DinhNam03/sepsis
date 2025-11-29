import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


def show():
    BASE_DIR = os.path.dirname(__file__)

    st.title("Dự báo nguy cơ nhiễm khuẩn huyết (Sepsis) ở bệnh nhân viêm tụy cấp")
    st.write("Nhập các chỉ số xét nghiệm của bệnh nhân viêm tụy cấp để dự báo nguy cơ xuất hiện nhiễm khuẩn huyết.")

    @st.cache_resource
    def load_model():
        return joblib.load('rfc.joblib')

    @st.cache_data
    def load_data():
        return joblib.load('X_train.joblib')
    rfc = load_model()
    X_train = load_data() # type: ignore

    explainer = shap.Explainer(model=rfc, masker=X_train, feature_names=X_train.shape)

    # INPUT FORM 
    col1, col2 = st.columns(2)
    vars = {}

    with col1:
        vars['age'] = st.number_input('Age', min_value=20, max_value=93, step=1)
        vars['wbc'] = st.number_input('White blood cell (WBC)', min_value=0.3, max_value=187.1, step=0.1)
        vars['rbc'] = st.number_input('Red blood cell (RBC)', min_value=1.9, max_value=5.9, step=0.1)
        vars['rdw'] = st.number_input('Red blood cell distribution width (RDW)', min_value=11.9, max_value=32.8, step=0.1)
        vars['platelets'] = st.number_input('Platelets', min_value=5, max_value=747, step=1)
        vars['potassium'] = st.number_input('Potassium', min_value=3.1, max_value=5.7, step=0.1)
        
    with col2:
        vars['creatinine'] = st.number_input('Creatinine', min_value=0.3, max_value=10.1, step=0.1)
        vars['glucose'] = st.number_input('Glucose', min_value= 4, max_value=215, step=1)
        vars['alp'] = st.number_input('Alkaline phosphatase (ALP)', min_value=27.3, max_value=4153.5, step=0.1)
        vars['ptt'] = st.number_input('Partial activated thromboplastin time (PTT)', min_value=20.4, max_value=119.8, step=0.1)
        vars['inr'] = st.number_input('international normalized ratio (INR)', min_value=0.9, max_value=5.8, step=0.1)
        vars['tyg'] = st.number_input('TyG index', min_value=5.9, max_value=11.9, step=0.1)
        
        
    st.markdown("---")
    arr = ['Không nhiễm khuẩn huyết', 'Nhiễm khuẩn huyết']

    # PREDICTION 
    if st.button('🔍 Dự đoán'):
        # st.subheader("🧾 Input Summary")
        df_pred = pd.DataFrame([vars])
        # st.write(df_pred.iloc[0])
        pred = rfc.predict(df_pred.iloc[:1])[0]
        pred_prob = rfc.predict_proba(df_pred.iloc[:1])[0]
        st.write(df_pred.iloc[-1:])

        st.subheader("Kết quả dự đoán")
        st.success(f"**Dự đoán:** {arr[pred]}")
        st.info(f"**Xác xuất:** {pred_prob[pred]:.2f}")

        shap_values = explainer(df_pred.iloc[:1])
        #shap.plots.w
        fig = shap.plots.force(shap_values[0, :, 1], matplotlib=True)
        st.pyplot(fig)
        #fig = shap.plots.waterfall(shap_values[0, :, 1], matplotlib=True)
        #st.pyplot(fig)
        # waterfall
        fig, ax = plt.subplots(figsize=(8,4))
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)

        # Beeswarm cho toàn bộ X_train, class dự đoán
        shap_values_train = explainer(X_train)
        fig, ax = plt.subplots(figsize=(8,4))
        shap.plots.beeswarm(shap_values_train[:, :, pred], show=False)
        st.pyplot(fig)              
            
        
       
 