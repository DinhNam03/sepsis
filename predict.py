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

# age               20.000000
#  wbc_max            1.700000
#  rbc                1.931818
#  rdw_max           12.000000
#  chloride          81.000000
#  potassium_max      3.600000
#  creatinine_max     0.300000
#  alp_max           36.000000
#  pt_max             9.400000
#  ptt_max           20.400000
#  inr_max            0.900000
#  tyg_max            7.347622
#  dtype: float64,
#  age                  93.000000
#  wbc_max           12500.000000
#  rbc                   5.924759
#  rdw_max              34.900000
#  chloride            135.000000
#  potassium_max        10.000000
#  creatinine_max       15.200000
#  alp_max            5006.000000
#  pt_max              150.000000
#  ptt_max             150.000000
#  inr_max              15.600000
#  tyg_max              15.839774

    with col1:
        vars['age'] = st.number_input('Age', min_value=20, max_value=93, step=1)
        vars['wbc_max'] = st.number_input('White blood cell (WBC)', min_value=1.7, max_value=12500.0, step=0.1)
        vars['rbc'] = st.number_input('Red blood cell (RBC)', min_value=1.9, max_value=5.9, step=0.1)
        vars['rdw_max'] = st.number_input('Red blood cell distribution width (RDW)', min_value=12.0, max_value=34.9, step=0.1)
        vars['potassium_max'] = st.number_input('Potassium', min_value=3.6, max_value=10.0, step=0.1)
        vars['creatinine_max'] = st.number_input('Creatinine', min_value=0.3, max_value=15.2, step=0.1)
    with col2:
        vars['alp_max'] = st.number_input('Alkaline phosphatase (ALP)', min_value=36, max_value=5006, step=1)
        vars['pt_max'] = st.number_input('Prothrombin Time (PT)', min_value= 9.4, max_value=150.0, step=0.1)
        vars['ptt_max'] = st.number_input('Partial activated thromboplastin time (PTT)', min_value=20.4, max_value=150.0, step=0.1)
        vars['inr_max'] = st.number_input('international normalized ratio (INR)', min_value=0.9, max_value=15.6, step=0.1)
        vars['tyg_max'] = st.number_input('TyG index', min_value=7.3, max_value=15.8, step=0.1)
        vars['aki'] = st.selectbox(label='Acute Kidney Injury (AKI)', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        
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
       
        # waterfall
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap_values[0, :, 1],
            show=False,
            max_display=len(df_pred.columns)
        )
        st.pyplot(fig)

                 
            
        
       
 