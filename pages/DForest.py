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
    # แสดงรูปภาพ
    # =========================
    st.image("./img/shirt.jpg", caption="ตัวอย่างสินค้า", use_column_width=True)
    st.image("./img/sell.jpg", caption="โปรโมชั่นขาย", use_column_width=True)

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

    # Location เป็นตัวเลข
    location_list = sorted(df['Location'].unique())
    loc_dict = {name: idx+1 for idx, name in enumerate(location_list)}
    f4 = st.number_input(f"Location (1-{len(location_list)})", min_value=1, max_value=len(location_list), value=1)

    # Shipping Type เป็นตัวเลข
    shipping_list = sorted(df['Shipping Type'].unique())
    ship_dict = {name: idx+1 for idx, name in enumerate(shipping_list)}
    f5 = st.number_input(f"Shipping Type (1-{len(shipping_list)})", min_value=1, max_value=len(shipping_list), value=1)

    # Item Purchased เป็นตัวเลข
    item_list = sorted(df['Item Purchased'].unique())
    item_dict = {name: idx+1 for idx, name in enumerate(item_list)}
    f6 = st.number_input(f"Item Purchased (1-{len(item_list)})", min_value=1, max_value=len(item_list), value=1)

    # แปลงตัวเลขกลับเป็นชื่อเพื่อ one-hot encoding
    location_val = [k for k,v in loc_dict.items() if v==f4][0]
    shipping_val = [k for k,v in ship_dict.items() if v==f5][0]
    item_val = [k for k,v in item_dict.items() if v==f6][0]

    # เมื่อกดปุ่มพยากรณ์
    if st.button("พยากรณ์"):
        input_data = pd.DataFrame([[f1, f2, f3, purchase_amount, location_val, shipping_val, item_val]],
                                  columns=['Age','Size','Season','Purchase Amount','Location','Shipping Type','Item Purchased'])

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
