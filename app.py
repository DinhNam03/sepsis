import streamlit as st
import predict
import model


# ================== CONFIG ==================
st.set_page_config(
    page_title="Ứng dụng viêm tụy cấp",
    page_icon="🩺",
    layout="wide",
)

# ================== SIDEBAR ==================
st.sidebar.title("🩺 Ứng dụng viêm tụy cấp")
page = st.sidebar.radio(
    "Điều hướng",
    ["Dự đoán", "Mô hình đánh giá"]
)

# ================== PAGE ROUTING ==================
if page == "Dự đoán":
    predict.show()
else:
    model.show()

# ================== FOOTER ==================
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Pham Dinh Nam 💻")





# import streamlit as st
# import predict
# import model

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to:", ["Prediction", "Model Evaluation"])

# if page == "Prediction":
#     predict.show()
# else:
#     model.show()
