# DForest.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def run_dforest():
    st.header("🧥 Random Forest for Classification 🧥")

    # =========================
    # แสดงรูปภาพคู่กัน
    # =========================
    # แสดงรูปคู่กัน
    col1, col2 = st.columns(2)
    with col1:
    st.image("./img/shirt.jpg", caption="Shirt Example", use_container_width=True)
    with col2:
    st.image("./img/sell.jpg", caption="Sales Promotion", use_container_width=True)


    # โหลดข้อมูล
    df = pd.read_csv("./data/shopping.csv")
    st.write(df.head(10))

    # เตรียม Features และ Target
    X = df.drop('Category', axis=1)
    y = df['Category']

    # one-hot encoding
    X = pd.get_dummies(X)

    # แบ่ง train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

    # สร้างและเทรนโมเดล RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=200)
    rf.fit(x_train, y_train)

    # Input จากผู้ใช้
    st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์ (ทดสอบ)")

    f1 = st.number_input('Age', min_value=1, max_value=100, value=25)
    f2 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, value=2)
    f3 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, value=3)
    purchase_amount = st.number_input('Purchase Amount (USD)', min_value=1, value=50)

    # Location
    locations = [
        "Kentucky","Maine","Massachusetts","Rhode Island","Oregon","Wyoming","Montana","Louisiana","West Virginia",
        "Missouri","Arkansas","Hawaii","Delaware","New Hampshire","New York","Alabama","Mississippi","North Carolina",
        "California","Oklahoma","Florida","Texas","Nevada","Kansas","Colorado","North Dakota","Illinois","Indiana",
        "Arizona","Alaska","Tennessee","Ohio","New Jersey","Maryland","Vermont","New Mexico","South Carolina",
        "Idaho","Pennsylvania","Connecticut","Utah","Virginia","Georgia","Nebraska","Iowa"
    ]
    location = st.selectbox("เลือก Location", locations)

    # Shipping Type
    shipping_types = ["Free Shipping", "Express", "Store Pickup", "2-Day Shipping", "Next Day Air", "Standard"]
    shipping_type = st.selectbox("เลือก Shipping Type", shipping_types)

    # Item Purchased
    items = [
        "Pants","Dress","Coat","Jacket","Scarf","Skirt","Handbag","T-shirt","Hoodie","Shoes","Shorts","Jewelry",
        "Sneakers","Sweater","Blouse","Shirt","Belt","Hat","Sunglasses","Gloves","Backpack","Jeans","Boots",
        "Socks","Sandals"
    ]
    item_purchased = st.selectbox("เลือก Item Purchased", items)

    # เมื่อกดปุ่มพยากรณ์
    if st.button("พยากรณ์"):
        input_data = pd.DataFrame([[
            f1, f2, f3, purchase_amount, location, shipping_type, item_purchased
        ]], columns=['Age','Size','Season','Purchase Amount','Location','Shipping Type','Item Purchased'])

        # one-hot encode input และ align columnsกับ X
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        y_predict = rf.predict(input_data)
        st.write("🎯 ผลการพยากรณ์:", y_predict[0])

    # Accuracy
    y_pred = rf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    st.write(f'✅ ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    fig, ax = plt.subplots(figsize=(6,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # =========================
    # Boxplot Explorer
    # =========================
    st.subheader("📊 Boxplot Explorer")
    box_columns = ['Age', 'Size', 'Season', 'Location', 'Shipping Type', 'Item Purchased']
    selected_column = st.selectbox("เลือกตัวแปรเพื่อดู Boxplot", box_columns)

    plt.figure(figsize=(10,6))
    if selected_column in ['Age', 'Size', 'Season']:
        sns.boxplot(y=df[selected_column])
    else:
        sns.boxplot(x=df[selected_column].astype('category').cat.codes)
    plt.xticks(rotation=45)
    st.pyplot(plt)
