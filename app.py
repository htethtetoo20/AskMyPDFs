import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationRetrievalChain
from langchain_openai import ChatOpenAI

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
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationRetrievalChain(llm=llm, memory=memory, retriever=vectorstores.as_retriever())
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="AskMyPDFs", page_icon=":books:")

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    st.header("Chat with multiple PDFs")
    st.text_input("Ask a question about your documents:")


    with st.sidebar:
        st.subheader("AskMyPDFs")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text=get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)
                # get vector store
                vectorstores = get_vector_stores(text_chunks)
                # create conversation chain and store in session state
                st.session_state.conversation_chain = get_conversation_chain(vectorstores)


if __name__ == "__main__":
    main()



    
