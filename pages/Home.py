import streamlit as st
from KNN import run_knn
from DForest import run_dforest
from Bay import run_bayes

st.title("Customer Purchase Prediction")

st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio("เลือกโมเดล", ["KNN", "Random Forest", "Naive Bayes"])

if model_choice == "KNN":
    run_knn()
elif model_choice == "Random Forest":
    run_dforest()
elif model_choice == "Naive Bayes":
    run_bayes()
