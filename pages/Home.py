import streamlit as st

# ====== หน้า Home ======
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")
st.write("""
ยินดีต้อนรับ!  
โปรดเลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:
""")

# ====== Sidebar ======
st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

# ====== ปุ่มไปยังโมเดล ======
if st.sidebar.button("เริ่มพยากรณ์"):
    if model_choice == "KNN":
        st.write("คุณเลือก KNN")
        # import หรือ run ไฟล์ KNN.py
        import KNN
    elif model_choice == "Random Forest":
        st.write("คุณเลือก Random Forest")
        import DForest
    elif model_choice == "Naive Bayes":
        st.write("คุณเลือก Naive Bayes")
        import Bay
