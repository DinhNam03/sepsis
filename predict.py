import streamlit as st
import pandas as pd
import joblib
import os

def show():
    BASE_DIR = os.path.dirname(__file__)

    st.title("🔮 Acute Pancreatitis Prediction")
    st.write("Nhập các chỉ số lâm sàng của bệnh nhân để dự đoán khả năng sống sót:")

    @st.cache_resource
    def load_model():
        return joblib.load(os.path.join(BASE_DIR, 'rfc1.joblib'))

    rfc = load_model()

    # ================== INPUT FORM ==================
    col1, col2 = st.columns(2)
    vars = {}

    with col1:
        vars['bilirubin_total_max'] = st.number_input('Bilirubin total', min_value=-23.2, max_value=51.2, step=0.1)
        vars['rdw_max'] = st.number_input('RDW', min_value=11.8, max_value=34.9, step=0.1)
        vars['NPAR'] = st.number_input('NPAR', min_value=1.36, max_value=71.5, step=0.1)
        vars['NLR'] = st.number_input('NLR', min_value=0.04, max_value=270.2, step=0.1)
        vars['sapsii'] = st.number_input('SAPSII', min_value=6, max_value=94, step=1)
        vars['sofa'] = st.number_input('SOFA', min_value=0, max_value=21, step=1)

    with col2:
        vars['cci'] = st.number_input('CCI', min_value=0, max_value=17, step=1)
        vars['apsiii'] = st.number_input('APSIII', min_value=7, max_value=159, step=1)
        vars['temperature_mean'] = st.number_input('Temperature body (°C)', min_value=33.6, max_value=40.1, step=0.1)
        vars['vasopressin'] = st.selectbox('Vasopressin', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        vars['crrt'] = st.selectbox('CRRT', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        vars['has_sepsis'] = st.selectbox('Has sepsis', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]

    st.markdown("---")
    arr = ['Survival', 'Died']

    # ================== PREDICTION ==================
    if st.button('🔍 Predict'):
        df_pred = pd.DataFrame([vars])
        pred = rfc.predict(df_pred)[0]
        pred_prob = rfc.predict_proba(df_pred)[0]
        
        col_result1, col_result2 = st.columns([2, 1])
        with col_result1:
            st.subheader("🧾 Input Summary")
            st.dataframe(df_pred.style.format(precision=2))

        with col_result2:
            st.subheader("🧠 Prediction Result")
            st.success(f"**Prediction:** {arr[pred]}")
            st.info(f"**Probability:** {pred_prob[pred]:.2f}")

    st.markdown("---")
    st.caption("Model: Random Forest Classifier (RFC)")






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


