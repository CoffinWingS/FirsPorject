import streamlit as st

# ตั้งชื่อหน้า
st.set_page_config(page_title="Customer Purchase Prediction", page_icon="🛍️", layout="centered")

# ส่วนหัว
st.title("🛍️ Customer Purchase Prediction")
st.subheader("เลือกโมเดลที่ต้องการใช้งานจากแถบด้านซ้าย")

st.write("""
ยินดีต้อนรับสู่ระบบพยากรณ์การซื้อของลูกค้า  
คุณสามารถเลือกโมเดล Machine Learning ที่ต้องการทดสอบได้จาก **Sidebar**  
- 🔹 KNN  
- 🔹 Random Forest  
- 🔹 Naive Bayes  
""")

st.info("➡️ ไปที่แถบด้านซ้าย (Sidebar) เพื่อเลือกโมเดลที่ต้องการ", icon="ℹ️")

st.image("https://cdn-icons-png.flaticon.com/512/2331/2331954.png", width=200)

