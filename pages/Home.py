import streamlit as st
import os

st.set_page_config(page_title="Customer Purchase Prediction", page_icon="🛍️", layout="centered")

# ==========================
# Header
# ==========================
st.title("🛍️ ระบบพยากรณ์การซื้อของลูกค้า")
st.markdown("เลือกโมเดล Machine Learning ที่คุณต้องการทดสอบ:")

# ==========================
# สร้างปุ่มเลือกโมเดล
# ==========================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔹 K-Nearest Neighbors (KNN)"):
        os.system("streamlit run KNN.py")

with col2:
    if st.button("🌲 Random Forest"):
        os.system("streamlit run DForest.py")

with col3:
    if st.button("📊 Naive Bayes"):
        os.system("streamlit run Bay.py")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("ระบบนี้สร้างขึ้นเพื่อเปรียบเทียบผลการพยากรณ์ของ 3 โมเดล: KNN, Random Forest, และ Naive Bayes")
