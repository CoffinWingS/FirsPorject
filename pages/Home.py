import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# ========================
# หน้า Home
# ========================
st.title("🛍️ Customer Purchase Prediction System")
st.write("โปรดเลือกโมเดลที่ต้องการใช้ในการพยากรณ์การซื้อของลูกค้า")

# ========================
# แถบเมนูด้านข้าง
# ========================
menu = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการ:",
    ("🏠 Home", "KNN", "Random Forest", "Naive Bayes")
)

if menu == "🏠 Home":
    st.subheader("Welcome 👋")
    st.write("""
        ระบบนี้ถูกออกแบบมาเพื่อทำนายการซื้อสินค้าของลูกค้า  
        โดยคุณสามารถเลือกใช้งานโมเดลได้ 3 แบบ คือ:
        - KNN  
        - Random Forest  
        - Naive Bayes  

        เลือกโมเดลจากแถบด้านซ้ายเพื่อเริ่มต้นใช้งานได้เลย ✅
    """)

elif menu == "KNN":
    st.subheader("🔍 Running KNN Model...")
    os.system(f"{sys.executable} KNN.py")

elif menu == "Random Forest":
    st.subheader("🌲 Running Random Forest Model...")
    os.system(f"{sys.executable} DForest.py")

elif menu == "Naive Bayes":
    st.subheader("📊 Running Naive Bayes Model...")
    os.system(f"{sys.executable} Bay.py")

