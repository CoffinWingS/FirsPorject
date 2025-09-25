import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_model(page_name):
    st.session_state.page = page_name

if st.session_state.page == "home":
    st.title("Customer Purchase Prediction System")
    st.write("เลือกโมเดลสำหรับพยากรณ์การซื้อของลูกค้า:")

    model_choice = st.radio("เลือกโมเดล:", ["KNN", "Random Forest", "Naive Bayes"])
    
    if st.button("เริ่มพยากรณ์"):
        if model_choice == "KNN":
            go_to_model("KNN_page")
        elif model_choice == "Random Forest":
            go_to_model("RF_page")
        elif model_choice == "Naive Bayes":
            go_to_model("NB_page")

elif st.session_state.page == "KNN_page":
    st.title("KNN Model Prediction")
    import KNN_page  # หน้า KNN จะมี input และ predict
    if st.button("กลับ Home"):
        st.session_state.page = "home"

elif st.session_state.page == "RF_page":
    st.title("Random Forest Model Prediction")
    import RF_page
    if st.button("กลับ Home"):
        st.session_state.page = "home"

elif st.session_state.page == "NB_page":
    st.title("Naive Bayes Model Prediction")
    import NB_page
    if st.button("กลับ Home"):
        st.session_state.page = "home"
