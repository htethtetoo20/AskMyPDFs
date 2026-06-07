import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import bot_template, user_template,css

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_stores(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstores):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstores.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_input):
    if st.session_state.conversation_chain is None:
        st.warning("Please wait while we process your PDFs or upload new PDFs.")
        return
    response = st.session_state.conversation_chain({'question': user_input})
    st.session_state.chat_history= response['chat_history']
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="AskMyPDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs📚")
    user_input = st.text_input("Ask a question about your documents:")
    if user_input:
        handle_userinput(user_input)

    with st.sidebar:
        st.subheader("AskMyPDFs")
        pdf_docs = st.file_uploader("Upload your PDFs📚", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload", disabled=not pdf_docs):
            with st.spinner("Processing"):
                # get pdf text
                raw_text=get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks=get_text_chunks(raw_text)
                # get vector store
                vectorstores = get_vector_stores(text_chunks)
                # create conversation chain and store in session state
                st.session_state.conversation_chain = get_conversation_chain(vectorstores)


if __name__ == "__main__":
    main()



    
