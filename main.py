import streamlit as st
from streamlit_chat import  message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, #first message, elso prompt
    HumanMessage,
    AIMessage
)
prompt = "I want you to act as an accountant and come up with creative ways to manage finances. You'll need to consider budgeting, investment strategies and risk management when creating a financial plan for your client. In some cases, you may also need to provide advice on taxation laws and regulations in order to help them maximize their profits."

def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Open AI API key is not set")
        exit(1)
    st.set_page_config(
        page_title="PDF Q&A - PoC",
        page_icon="🐱‍🐉"
    )
    

def main(): 
    init()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=prompt)
        ]
    chat = ChatOpenAI(temperature=0.5)#nem kell api kulcs mert langchain automatikusan kiszedi .env-bol

    messages = [
        SystemMessage(content=prompt) #Instruction
    ]


    st.header("Chat with yout pdfs")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                st.write("TODO")
    
    

    user_input = st.text_input("Your question",  key="user_input")

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking"):
            response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))


    messages = st.session_state.get('messages',[]) # kiszedjuk a messageket
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 1:
            message(msg.content,is_user=True, key=str(i)+"_usr")
        else:
            message(msg.content,is_user=False, key=str(i)+"_ai")
    st.write(messages)

        

                

if __name__ == "__main__":
    main()
