import streamlit as st

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")

# สร้าง sidebar navigation
page = st.sidebar.selectbox("เลือกโมเดล", ["Home", "KNN", "Random Forest", "Naive Bayes"])

if page == "Home":
    st.write("ยินดีต้อนรับ! โปรดเลือกโมเดลจาก sidebar")
elif page == "KNN":
    import KNN
elif page == "Random Forest":
    import DForest
elif page == "Naive Bayes":
    import Bay
