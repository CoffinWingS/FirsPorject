import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# โหลดข้อมูล
df = pd.read_csv("./data/shopping.csv")

# ====== ฟังก์ชันโมเดล ======
def run_knn():
    st.subheader("KNN Prediction")
    # ใส่โค้ด KNN ที่รับ input user และพยากรณ์ที่นี่
    st.write("นี่คือหน้าของ KNN")

def run_rf():
    st.subheader("Random Forest Prediction")
    st.write("นี่คือหน้าของ Random Forest")

def run_nb():
    st.subheader("Naive Bayes Prediction")
    st.write("นี่คือหน้าของ Naive Bayes")

# ====== หน้า Home ======
st.title("Customer Purchase Prediction System")
st.sidebar.title("เลือกโมเดล")
model_choice = st.sidebar.radio(
    "เลือกโมเดลที่ต้องการใช้",
    ("KNN", "Random Forest", "Naive Bayes")
)

if model_choice == "KNN":
    run_knn()
elif model_choice == "Random Forest":
    run_rf()
elif model_choice == "Naive Bayes":
    run_nb()
