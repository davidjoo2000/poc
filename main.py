import streamlit as st
from streamlit_chat import  message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from io import StringIO


from langchain.schema import (
    SystemMessage, #first message, elso prompt
    HumanMessage,
    AIMessage
)
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

prompt = ""


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Open AI API key is not set")
        exit(1)
    st.set_page_config(
        page_title="PDF Q&A - PoC",
        page_icon="🐱‍🐉"
    )
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=140,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
         memory=memory
     )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history with user's message first in reverse order
    for i in reversed(range(len(st.session_state.chat_history))):
        message = st.session_state.chat_history[i]
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    init()


    st.header("Chat with your own documents")
    with st.form("Question", clear_on_submit=True):
        user_question = st.text_input("Ask a question about your documents:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            handle_userinput(user_question)
    

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                if docs is not None:
                    for file in docs:
                        file_name = file.name
                        if file_name.split(".")[1]=="txt":
                            raw_text = StringIO(file.getvalue().decode("utf-8")).read()
                        else:
                            raw_text = ""
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                raw_text += page.extract_text()

                    # get pdf text
                    
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)


if __name__ == "__main__":
    main()
