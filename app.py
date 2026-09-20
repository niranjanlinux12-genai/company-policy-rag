import streamlit as st
from rag import ask_question

st.title("company rag chatbot")

st.write("ask question about the company document")

question = st.text_input("enter your question: ")

if st.button("ask question"):
    if question:
        answer = ask_question(question)
        st.subheader("answer")
        st.write(answer)
    else:
        st.warning("please enter your question")