import streamlit as st
import importlib

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")
st.write("ยินดีต้อนรับ! โปรดเลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:")

st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

if st.sidebar.button("เริ่มพยากรณ์"):
    if model_choice == "KNN":
        import KNN
        importlib.reload(KNN)
    elif model_choice == "Random Forest":
        import DForest
        importlib.reload(DForest)
    elif model_choice == "Naive Bayes":
        import Bay
        importlib.reload(Bay)
