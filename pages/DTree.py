import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Mapping dictionary ===
gender_map = {1: 'Male', 2: 'Female'}
size_map = {1: 'S', 2: 'M', 3: 'L', 4: 'XL'}
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}

st.header("Decision Tree for Classification")
df = pd.read_csv("./data/shopping.csv")
st.write(df.head(10))

# ====== Boxplot Section ======
st.subheader("📊 Boxplot Visualization of Features")
numeric_features = ['Age', 'Size', 'Purchase Amount']
categorical_features = ['Item Purchased', 'Location', 'Season', 'Shipping Type']

# Numeric Features
for col in numeric_features:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f'Boxplot of {col}')
    st.pyplot(fig)

# Categorical Features (ใช้ Purchase Amount เป็นแกน y)
for col in categorical_features:
    if col in df.columns:  # เช็คว่ามีคอลัมน์นี้ใน DataFrame
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x=df[col], y=df['Purchase Amount'], ax=ax)
        ax.set_title(f'Boxplot of Purchase Amount by {col}')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ====== เตรียม Features และ Target ======
X = df.drop('Category', axis=1)
y = df['Category']

# แปลง categorical ให้เป็น dummy ตัวเลข
X = pd.get_dummies(X)

# แบ่ง train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# ====== ตรวจสอบ distribution ของ target ======
st.subheader("Class Distribution")
st.write(y.value_counts())   # ดูจำนวนแต่ละคลาส

# ====== สร้างและเทรนโมเดล ======
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

# ====== Input จากผู้ใช้ ======
st.subheader("กรุณาป้อนข้อมูล (ใช้ตัวเลขแทนค่า)")

f1 = st.number_input('Age', min_value=1, max_value=100, step=1)
f2 = st.number_input('Gender (1=Male, 2=Female)', min_value=1, max_value=2, step=1)
f3 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, step=1)
f4 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, step=1)

# ====== Debug Input Data ======
if st.button("Debug Input"):
    input_data = pd.DataFrame([[f1, f2, f3, f4]], 
                              columns=['Age','Gender','Size','Season'])
    
    # one-hot encoding
    input_data = pd.get_dummies(input_data)

    # align columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    st.write("🔍 Input data after processing:")
    st.write(input_data)

# ====== Prediction ======
if st.button("พยากรณ์"):
    # แปลงตัวเลขที่ user กรอก -> ค่า string จริง
    gender_val = gender_map[f2]
    size_val = size_map[f3]
    season_val = season_map[f4]

    # เตรียม dataframe สำหรับ input
    input_data = pd.DataFrame([[f1, gender_val, size_val, season_val]],
                              columns=['Age','Gender','Size','Season'])
    
    # one-hot encoding ให้ตรงกับ X
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # ทำนายผล
    y_predict2 = dtree.predict(input_data)
    st.write("🎯 ผลการพยากรณ์:", y_predict2[0])

# ====== Accuracy ======
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'✅ ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

# ====== Plot Tree ======
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=X.columns, class_names=y.unique(), filled=True, ax=ax)
st.pyplot(fig)

