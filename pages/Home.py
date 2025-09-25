import streamlit as st
from KNN import run_knn
from DForest import run_dforest
from Bay import run_bayes

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Customer Purchase Prediction System")

st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

# เรียกฟังก์ชันตามโมเดล
if model_choice == "KNN":
    run_knn()
elif model_choice == "Random Forest":
    run_dforest()
elif model_choice == "Naive Bayes":
    run_bayes()
