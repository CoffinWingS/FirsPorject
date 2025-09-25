import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================
# โหลดข้อมูล
# ==========================
st.header("K-Nearest Neighbors (KNN) for Classification")
df = pd.read_csv("./data/shopping.csv")
st.write(df.head(10))

# ====== เตรียม Features และ Target ======
X = df.drop('Category', axis=1)
y = df['Category']

# แปลง categorical เป็นตัวเลข (one-hot encoding)
X = pd.get_dummies(X)

# แบ่ง train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# ====== สร้างและเทรนโมเดล KNN ======
ModelKNN = KNeighborsClassifier(n_neighbors=5)   # ค่า k = 5 (สามารถปรับได้)
knn = ModelKNN.fit(x_train, y_train)

# ====== Input จากผู้ใช้ ======
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์ (ทดสอบ)")

f1 = st.number_input('Age', min_value=1, max_value=100, value=25)
f2 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, value=2)
f3 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, value=3)
f4 = st.number_input('Purchase Amount', min_value=1, max_value=10000, value=500)

# dropdown ให้เลือก Location
location_list = [
    "Kentucky","Maine","Massachusetts","Rhode Island","Oregon","Wyoming","Montana",
    "Louisiana","West Virginia","Missouri","Arkansas","Hawaii","Delaware","New Hampshire",
    "New York","Alabama","Mississippi","North Carolina","California","Oklahoma","Florida",
    "Texas","Nevada","Kansas","Colorado","North Dakota","Illinois","Indiana","Arizona",
    "Alaska","Tennessee","Ohio","New Jersey","Maryland","Vermont","New Mexico","South Carolina",
    "Idaho","Pennsylvania","Connecticut","Utah","Virginia","Georgia","Nebraska","Iowa"
]
f5 = st.selectbox("Location", location_list)

# dropdown Shipping Type
shipping_list = ["Free Shipping", "Express", "Store Pickup", "2-Day Shipping", "Next Day Air", "Standard"]
f6 = st.selectbox("Shipping Type", shipping_list)

# dropdown Item Purchased
item_list = [
    "Pants","Dress","Coat","Jacket","Scarf","Store Pickup","Skirt","Handbag","T-shirt",
    "Hoodie","Shoes","Shorts","Jewelry","Sneakers","Sweater","Blouse","Shirt","Belt","Hat",
    "Sunglasses","Gloves","Backpack","Jeans","Boots","Socks","Sandals"
]
f7 = st.selectbox("Item Purchased", item_list)

if st.button("พยากรณ์"):
    # DataFrame ของ input
    input_data = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7]],
                              columns=['Age','Size','Season','Purchase Amount','Location','Shipping Type','Item Purchased'])
    
    # align columns ให้ตรงกับ X
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    y_predict2 = knn.predict(input_data)
    st.write("🎯 ผลการพยากรณ์:", y_predict2[0])

# ====== Accuracy ======
y_predict = knn.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'✅ ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

# ====== Confusion Matrix ======
cm = confusion_matrix(y_test, y_predict, labels=knn.classes_)
fig, ax = plt.subplots(figsize=(6, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)
