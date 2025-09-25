import streamlit as st
import streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu"
        option = ["Home","KNN","Random Forest","Naive Bayes"]
    )