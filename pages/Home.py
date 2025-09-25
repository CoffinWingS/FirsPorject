import streamlit as st

st.title("Customer Purchase Prediction System")
st.write("ยินดีต้อนรับ! โปรดเลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:")

st.sidebar.title("เลือกหน้า")
page = st.sidebar.selectbox("ไปยังโมเดล", ["Home", "KNN", "Random Forest", "Naive Bayes"])

if page == "Home":
    st.write("นี่คือหน้า Home")
elif page == "KNN":
    st.markdown("[ไป KNN](KNN)")
elif page == "Random Forest":
    st.markdown("[ไป Random Forest](DForest)")
elif page == "Naive Bayes":
    st.markdown("[ไป Naive Bayes](Bay)")
