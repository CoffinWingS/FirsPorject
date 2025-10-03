# DForest.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def run_dforest():
    # =========================
    # ตกแต่ง Header + Logo
    # =========================
    st.markdown(
        """
        <div style="background-color:#4CAF50;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h2 style="color:white;text-align:center;">🛍️ Customer Purchase Prediction Using Random Forest 🛍️</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("./image/logo.png", caption="Project Logo", use_column_width=True)

    # =========================
    # โหลดข้อมูล
    # =========================
    df = pd.read_csv("./data/shopping.csv")
    st.subheader("📂 แสดงตัวอย่างข้อมูล")
    st.write(df.head(10))

    # เตรียม Features และ Target
    X = df.drop('Category', axis=1)
    y = df['Category']
    X = pd.get_dummies(X)

    # แบ่ง train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=200
    )

    # สร้างและเทรนโมเดล RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=200)
    rf.fit(x_train, y_train)

    # =========================
    # Input จากผู้ใช้
    # =========================
    st.subheader("📝 กรอกข้อมูลเพื่อทำนาย")

    f1 = st.number_input('Age', min_value=1, max_value=100, value=25)
    f2 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, value=2)
    f3 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, value=3)
    purchase_amount = st.number_input('Purchase Amount (USD)', min_value=1, value=50)

    # Location (Numeric)
    locations = [
        "Kentucky","Maine","Massachusetts","Rhode Island","Oregon","Wyoming","Montana","Louisiana","West Virginia",
        "Missouri","Arkansas","Hawaii","Delaware","New Hampshire","New York","Alabama","Mississippi","North Carolina",
        "California","Oklahoma","Florida","Texas","Nevada","Kansas","Colorado","North Dakota","Illinois","Indiana",
        "Arizona","Alaska","Tennessee","Ohio","New Jersey","Maryland","Vermont","New Mexico","South Carolina",
        "Idaho","Pennsylvania","Connecticut","Utah","Virginia","Georgia","Nebraska","Iowa"
    ]
    location = st.number_input("Location (เลือกเลข 1 - {})".format(len(locations)), min_value=1, max_value=len(locations), value=1)

    # Shipping Type (Numeric)
    shipping_types = ["Free Shipping", "Express", "Store Pickup", "2-Day Shipping", "Next Day Air", "Standard"]
    shipping_type = st.number_input("Shipping Type (1=Free, 2=Express, ...)", min_value=1, max_value=len(shipping_types), value=1)

    # Item Purchased (Numeric)
    items = [
        "Pants","Dress","Coat","Jacket","Scarf","Skirt","Handbag","T-shirt","Hoodie","Shoes","Shorts","Jewelry",
        "Sneakers","Sweater","Blouse","Shirt","Belt","Hat","Sunglasses","Gloves","Backpack","Jeans","Boots",
        "Socks","Sandals"
    ]
    item_purchased = st.number_input("Item Purchased (1 - {})".format(len(items)), min_value=1, max_value=len(items), value=1)

    # เมื่อกดปุ่มพยากรณ์
    if st.button("🚀 เริ่มพยากรณ์"):
        input_data = pd.DataFrame([[
            f1, f2, f3, purchase_amount, locations[location-1], shipping_types[shipping_type-1], items[item_purchased-1]
        ]], columns=['Age','Size','Season','Purchase Amount','Location','Shipping Type','Item Purchased'])

        # one-hot encode input และ align columnsกับ X
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        y_predict = rf.predict(input_data)
        st.success(f"🎯 ผลการพยากรณ์: {y_predict[0]}")

    # =========================
    # Accuracy
    # =========================
    st.subheader("📈 ประสิทธิภาพของโมเดล")
    y_pred = rf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    st.metric(label="Accuracy", value=f"{score*100:.2f}%")

    # Confusion Matrix
    st.subheader("🔎 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    fig, ax = plt.subplots(figsize=(6,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # =========================
    # Boxplot Explorer
    # =========================
    st.subheader("📊 Boxplot Explorer")
    box_columns = ['Age', 'Size', 'Season', 'Purchase Amount']
    selected_column = st.selectbox("เลือกตัวแปรเพื่อดู Boxplot", box_columns)

    plt.figure(figsize=(10,6))
    sns.boxplot(y=df[selected_column])
    plt.title(f"Boxplot ของ {selected_column}")
    st.pyplot(plt)
