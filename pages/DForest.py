import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================
# โหลดข้อมูล
# ==========================
st.header("Random Forest for Classification")
df = pd.read_csv("./data/shopping.csv")
st.write(df.head(10))

# ====== เตรียม Features และ Target ======
X = df.drop('Category', axis=1)
y = df['Category']

# แปลง categorical เป็นตัวเลข (one-hot encoding)
X = pd.get_dummies(X)

# แบ่ง train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# ====== สร้างและเทรนโมเดล RandomForest ======
ModelRF = RandomForestClassifier(n_estimators=100, random_state=200)
rf = ModelRF.fit(x_train, y_train)

# ====== Input จากผู้ใช้ ======
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์ (ทดสอบ)")

f1 = st.number_input('Age', min_value=1, max_value=100, value=25)

# ให้ user เลือกค่าเป็นตัวเลข
f2 = st.number_input('Gender (1=Male, 2=Female)', min_value=1, max_value=2, value=1)
f3 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, value=2)
f4 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, value=3)

if st.button("พยากรณ์"):
    # DataFrame ของ input
    input_data = pd.DataFrame([[f1, f2, f3, f4]], columns=['Age','Gender','Size','Season'])
    
    # align columns ให้ตรงกับ X
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    y_predict2 = rf.predict(input_data)
    st.write("🎯 ผลการพยากรณ์:", y_predict2[0])

# ====== Accuracy ======
y_predict = rf.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'✅ ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

# ====== Confusion Matrix ======
cm = confusion_matrix(y_test, y_predict, labels=rf.classes_)
fig, ax = plt.subplots(figsize=(6, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)
