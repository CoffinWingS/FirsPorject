import streamlit as st

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# ====== Sidebar ======
st.sidebar.title("เลือกโมเดลสำหรับพยากรณ์")
model_option = st.sidebar.radio(
    "โมเดลที่ต้องการใช้:",
    ("KNN", "Random Forest", "Naive Bayes")
)

# ====== Main Page ======
st.title("ระบบพยากรณ์การซื้อของลูกค้า")
st.write("ยินดีต้อนรับ! กรุณาเลือกโมเดลจากแถบด้านซ้ายเพื่อเริ่มพยากรณ์ข้อมูลลูกค้า")

# ====== นำเข้าโมเดลที่ผู้ใช้เลือก ======
if model_option == "KNN":
    st.write("คุณเลือก KNN Model")
    import KNN  # เรียกไฟล์ KNN.py
elif model_option == "Random Forest":
    st.write("คุณเลือก Random Forest Model")
    import DForest  # เรียกไฟล์ DForest.py
elif model_option == "Naive Bayes":
    st.write("คุณเลือก Naive Bayes Model")
    import Bay  # เรียกไฟล์ Bay.py

