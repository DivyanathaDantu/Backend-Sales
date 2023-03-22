import streamlit as st


def hideNavBar():
    st.set_page_config(initial_sidebar_state="collapsed")

    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="collapsedControl"] {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
