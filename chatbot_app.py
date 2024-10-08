__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import textwrap
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import hub, PromptTemplate

st.set_page_config(layout="wide")

device = torch.device('cuda')

checkpoint = "/mnt/Data_1/MLProject/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

persist_directory = "db"


@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    # create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # create vector store here
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)


# @st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        k=5
    )

    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )
    return qa


def process_answer(instruction, qa):
    instruction = instruction
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


@st.cache_data
# function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))


def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>",
        unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    qa = qa_llm()

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input}, qa)
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)


if __name__ == "__main__":
    main()
