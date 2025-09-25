# Bay.py
import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns  # เพิ่มตรงนี้

def run_bayes():
    st.header("🧠 Naive Bayes for Classification 🧠")

    # โหลดข้อมูล
    df = pd.read_csv("./data/shopping.csv")
    st.write(df.head(10))

    # =========================
    # Boxplot Explorer
    # =========================
    st.subheader("📊 Boxplot Explorer")
    box_columns = ['Age', 'Size', 'Season', 'Purchase Amount', 'Location', 'Shipping Type', 'Item Purchased']
    selected_column = st.selectbox("เลือกตัวแปรเพื่อดู Boxplot", box_columns)

    plt.figure(figsize=(10,6))
    if selected_column in ['Age', 'Size', 'Season', 'Purchase Amount']:
        sns.boxplot(y=df[selected_column])
    else:
        sns.boxplot(x=df[selected_column].astype('category').cat.codes)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # เตรียม Features และ Target
    X = df.drop('Category', axis=1)
    y = df['Category']

    # one-hot encoding
    X = pd.get_dummies(X)

    # แบ่ง train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

    # สร้างและเทรนโมเดล Naive Bayes
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    # Input จากผู้ใช้
    st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์ (ทดสอบ)")

    f1 = st.number_input('Age', min_value=1, max_value=100, value=25)
    f2 = st.number_input('Size (1=S, 2=M, 3=L, 4=XL)', min_value=1, max_value=4, value=2)
    f3 = st.number_input('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)', min_value=1, max_value=4, value=3)
    f4 = st.number_input('Purchase Amount', min_value=1, max_value=10000, value=500)

    # Location
    location_list = [
        "Kentucky","Maine","Massachusetts","Rhode Island","Oregon","Wyoming","Montana",
        "Louisiana","West Virginia","Missouri","Arkansas","Hawaii","Delaware","New Hampshire",
        "New York","Alabama","Mississippi","North Carolina","California","Oklahoma","Florida",
        "Texas","Nevada","Kansas","Colorado","North Dakota","Illinois","Indiana","Arizona",
        "Alaska","Tennessee","Ohio","New Jersey","Maryland","Vermont","New Mexico","South Carolina",
        "Idaho","Pennsylvania","Connecticut","Utah","Virginia","Georgia","Nebraska","Iowa"
    ]
    f5 = st.selectbox("Location", location_list)

    # Shipping Type
    shipping_list = ["Free Shipping", "Express", "Store Pickup", "2-Day Shipping", "Next Day Air", "Standard"]
    f6 = st.selectbox("Shipping Type", shipping_list)

    # Item Purchased
    item_list = [
        "Pants","Dress","Coat","Jacket","Scarf","Skirt","Handbag","T-shirt",
        "Hoodie","Shoes","Shorts","Jewelry","Sneakers","Sweater","Blouse",
        "Shirt","Belt","Hat","Sunglasses","Gloves","Backpack","Jeans","Boots",
        "Socks","Sandals"
    ]
    f7 = st.selectbox("Item Purchased", item_list)

    # เมื่อกดปุ่มพยากรณ์
    if st.button("พยากรณ์"):
        input_data = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7]],
                                  columns=['Age','Size','Season','Purchase Amount','Location','Shipping Type','Item Purchased'])
        
        # one-hot encode input และ align columns กับ X
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        y_predict = nb.predict(input_data)
        st.write("🎯 ผลการพยากรณ์:", y_predict[0])

    # Accuracy
    y_pred = nb.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    st.write(f'✅ ความแม่นยำในการพยากรณ์ {score*100:.2f} %')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=nb.classes_)
    fig, ax = plt.subplots(figsize=(6,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
