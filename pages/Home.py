import streamlit as st

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")
st.write("ยินดีต้อนรับ! โปรดเลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:")

# Sidebar
st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

if st.sidebar.button("เริ่มพยากรณ์"):
    if model_choice == "KNN":
        st.write("คุณเลือก KNN")
        exec(open("KNN.py").read())
    elif model_choice == "Random Forest":
        st.write("คุณเลือก Random Forest")
        exec(open("DForest.py").read())
    elif model_choice == "Naive Bayes":
        st.write("คุณเลือก Naive Bayes")
        exec(open("Bay.py").read())
