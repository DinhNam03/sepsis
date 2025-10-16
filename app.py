import streamlit as st
import predict
import model

# ================== CONFIG ==================
st.set_page_config(
    page_title="Acute Pancreatitis App",
    page_icon="🩺",
    layout="wide",
)

# ================== SIDEBAR ==================
st.sidebar.title("🩺 Acute Pancreatitis App")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prediction", "📊 Model Evaluation"]
)

# ================== PAGE ROUTING ==================
if page == "🔮 Prediction":
    predict.show()
else:
    model.show()

# ================== FOOTER ==================
st.sidebar.markdown("---")
st.sidebar.caption("Developed by [Your Team] 💻")





# import streamlit as st
# import predict
# import model

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to:", ["Prediction", "Model Evaluation"])

# if page == "Prediction":
#     predict.show()
# else:
#     model.show()
