import streamlit as st
import openai
from pdf_processing import get_index_for_pdf
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
st.title("RAG usecase : Chatbot")

openai.api_key = os.getenv('OPENAI_API_KEY')



@st.cache_data
def create_vectordb(files, filenames):

    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb



pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)


if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)


prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence are the context of the pdf extract with metadata. 
    
        
    Reply "Not applicable" if text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""


prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])


for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])


question = st.chat_input("Ask anything")


if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()


    search_results = vectordb.similarity_search(question, k=3)

    pdf_extract = "/n ".join([result.page_content for result in search_results])


    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

 
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

 
    with st.chat_message("assistant"):
        botmsg = st.empty()


    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)


    st.session_state["prompt"] = prompt
    prompt.append({"role": "assistant", "content": result})  
    st.session_state["prompt"] = prompt
    
