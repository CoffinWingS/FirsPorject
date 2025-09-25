import streamlit as st

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")
st.write("ยินดีต้อนรับ! โปรดเลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:")

st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

# แสดงลิงก์ไปยังหน้าโมเดล
if model_choice == "KNN":
    st.write("[➡️ ไปหน้า KNN](KNN_page)")
elif model_choice == "Random Forest":
    st.write("[➡️ ไปหน้า Random Forest](RF_page)")
elif model_choice == "Naive Bayes":
    st.write("[➡️ ไปหน้า Naive Bayes](NB_page)")
