import streamlit as st


def hideNavBar():
    st.set_page_config(initial_sidebar_state="collapsed")

    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[class="css-163ttbj e1fqkh3o11"] {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden;}
    </style> """, unsafe_allow_html=True)