import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


def show():
    BASE_DIR = os.path.dirname(__file__)

    st.title("🔮 Dự báo nguy cơ nhiễm khuẩn huyết (Sepsis) ở bệnh nhân viêm tụy cấp")
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

# 'age', 'wbc', 'rbc', 'rdw', 'platelets', 'potassium', 'creatinine',
#         'glucose', 'alp', 'ptt', 'inr', 'tyg'
        
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
    arr = ['Nhiễm khuẩn huyết', 'Không nhiễm khuẩn huyết']

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
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)
        
       
            

    # st.markdown("---")
    # st.caption("Model: Random Forest Classifier (RFC)")



# age           20.000000
#  wbc            0.335714
#  rbc            1.931818
#  rdw           11.921875
#  platelets      5.000000
#  potassium      3.096552
#  creatinine     0.210976
#  glucose        4.000000
#  alp           27.269978
#  ptt           20.400000
#  inr            0.893949
#  tyg            5.918894
#  dtype: float64,
#  age             93.000000
#  wbc            187.101339
#  rbc              5.924759
#  rdw             32.833617
#  platelets      747.000000
#  potassium        5.706122
#  creatinine      10.115159
#  glucose        215.000000
#  alp           4153.545455
#  ptt            119.797458
#  inr              5.814231
#  tyg             11.484881









# import streamlit as st
# import pandas as pd
# import joblib
# import os

# def show():
#     BASE_DIR = os.path.dirname(__file__)
#     @st.cache_resource
#     def load_model():
#         return joblib.load(os.path.join(BASE_DIR, 'rfc1.joblib'))
#     rfc = load_model()
#     st.title('AP Predict')
#     col1, col2 = st.columns(2)
#     vars = {}
#     with col1:
#         vars['bilirubin_total_max'] = st.number_input(label='Bilirubin total', min_value=-23.2, max_value=51.2, step=0.1)
#         vars['rdw_max'] = st.number_input(label='RDW', min_value=11.8, max_value=34.9, step=0.1)
#         vars['NPAR'] = st.number_input(label='NPAR', min_value=1.36, max_value=71.5, step=0.1)
#         vars['NLR'] = st.number_input(label='NLR', min_value=0.04, max_value=270.2, step=0.1)
#         vars['sapsii'] = st.number_input(label='SAPSII', min_value=6, max_value=94, step=1)
#         vars['sofa'] = st.number_input(label='SOFA', min_value=0, max_value=21, step=1)
#     with col2:
#         vars['cci'] = st.number_input(label='CCI', min_value=0, max_value=17, step=1)
#         vars['apsiii'] = st.number_input(label='APSIII', min_value=7, max_value=159, step=1)
#         vars['temperature_mean'] = st.number_input(label='Temperature body', min_value=33.6, max_value=40.1, step=0.1)
#         vars['vasopressin'] = st.selectbox(label='Vasopressin', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#         vars['crrt'] = st.selectbox(label='CRRT', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#         vars['has_sepsis'] = st.selectbox(label='Has sepsis', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]

#     arr = ['Survival', 'Died']

#     if st.button('Predict'):
#         df_pred = pd.DataFrame([vars])
#         # st.write(df_pred.iloc[0])
#         pred = rfc.predict(df_pred.iloc[:1])[0]
#         pred_prob = rfc.predict_proba(df_pred.iloc[:1])[0]
#         st.write(df_pred.iloc[-1:])

#         st.write(f'Predict: {arr[pred]}, Probability: {pred_prob[pred]:.2f}')





