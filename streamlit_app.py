import streamlit as st
import requests

st.title("Restaurant SOP Bot")

query = st.text_input("Ask SOP question")

if st.button("Search"):
    try:
        res = requests.get(f"http://127.0.0.1:8000/ask?q={query}")
        st.write(res.json())
    except Exception as e:
        st.error("Backend not running. Start FastAPI first.")
