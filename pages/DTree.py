import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ตัวอย่าง DataFrame เริ่มต้น
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female'],
    'Size': ['S', 'M', 'L', 'XL'],
    'Season': ['Winter', 'Spring', 'Summer', 'Fall']
}
df = pd.DataFrame(data)
print("ก่อนแปลง:")
print(df)

# === Mapping dictionary ===
gender_map = {'Male': 1, 'Female': 2}
size_map = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}

# แปลงค่า
df['Gender'] = df['Gender'].map(gender_map)
df['Size'] = df['Size'].map(size_map)
df['Season'] = df['Season'].map(season_map)

print("\nหลังแปลง:")
print(df)

st.header("Decision Tree for classification")
df = pd.read_csv("./data/shopping.csv")
st.write(df.head(10))

# ====== เตรียม Features และ Target ======
X = df.drop('Category', axis=1)
y = df['Category']

# แปลง categorical ให้เป็นตัวเลข
X = pd.get_dummies(X)

# แบ่ง train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# ====== สร้างและเทรนโมเดล ======
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

# ====== Input จากผู้ใช้ (ตรงนี้คุณต้องแก้ให้เข้ากับ features จริง ๆ) ======
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์ (ทดสอบ)")
# ตัวอย่าง input สมมติ (4 ค่า) — คุณต้องแก้ตาม features ของ dataset
f1 = st.number_input('Age')
f2 = st.selectbox('Gender', ['Male', 'Female'])
f3 = st.text_input('Size')
f4 = st.text_input('Season')



if st.button("พยากรณ์"):
    # สร้าง dataframe 1 แถวที่เหมือน X
    input_data = pd.DataFrame([[f1, f2, f3, f4]], columns=['Age','Gender','Item Purchased','Location'])
    input_data = pd.get_dummies(input_data)

    # align columns ให้ตรงกับ X ที่เทรน
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    y_predict2 = dtree.predict(input_data)
    st.write("ผลการพยากรณ์:", y_predict2[0])

# ====== Accuracy ======
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

# ====== Plot Tree ======
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=X.columns, class_names=y.unique(), filled=True, ax=ax)
st.pyplot(fig)
