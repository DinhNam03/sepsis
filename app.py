import streamlit as st
import predict
import model
import kaplanMeier
import roctyg
import chart


# ================== CONFIG ==================
st.set_page_config(
    page_title="Ứng dụng viêm tụy cấp",
    page_icon="🩺",
    layout="wide",
)


# ================== SIDEBAR ==================
st.sidebar.title("Viêm tụy cấp")
page = st.sidebar.radio(
    " ",
    ["Dự đoán", "Mô hình đánh giá", "BoxChart", "Kaplan–Meier", "ROC", "RCS" ]
)

# ================== PAGE ROUTING ==================
if page == "Dự đoán":
    predict.show()

elif page == "Mô hình đánh giá":
    model.show()

elif page == "BoxChart":
    chart.show()

elif page == "ROC":
    roctyg.show()

elif page == "Kaplan–Meier":
    kaplanMeier.show()

# ================== CUSTOM FOOTER ==================
st.markdown(
    """
    <div class="custom-footer">
        Developed by Pham Dinh Nam 💻
    </div>
    """,
    unsafe_allow_html=True
)



# ================== CUSTOM CSS ==================
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 15px;
        width: 20px;
        padding: 10px 10px;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        color: #0b5394;
        font-size: 30px;
        font-weight: bold;
    }

    /* Sidebar radio button */
    .css-1n76uvr {
        font-size: 18px;
        color: #073763;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    .custom-footer {
        text-align: center;
        font-size: 14px;
        color: #555;
        padding: 10px;
        border-top: 1px solid #ccc;
    }

    /* Main page header */
    .css-1d391kg h1 {
        color: #0b5394;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Streamlit buttons (nút chọn thời điểm KM) */
    div.stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        background-color: #0b5394;
        color: white;
        border-radius: 10px;
        height: 40px;
        width: 200px;  /* bạn có thể chỉnh width */
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)