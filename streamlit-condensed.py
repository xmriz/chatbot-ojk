from dotenv import load_dotenv
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from utils.index_store import (store_vector_index, load_vector_index)
from utils.node_parser import parse_nodes
from utils.documents_reader import read_documents
import streamlit as st
import logging
import sys
from utils.models_definer import ModelName
from llama_index.core import Settings
from utils.models_definer import get_llm_and_embedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank

import nest_asyncio
nest_asyncio.apply()
load_dotenv()

st.set_page_config(page_title="OJK CHATBOT",
                   page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the OJK BOT 💬🤖")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_agent():
    model_name = ModelName.AZURE_OPENAI

    # ollama/openai/azure_openai
    llm, embedding_llm = get_llm_and_embedding(model_name=model_name)

    Settings.llm = llm
    Settings.embed_model = embedding_llm

    index_all = load_vector_index()

    vector_retriever_all = index_all.as_retriever(similarity_top_k=3)

    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )

    query_engine_all = RetrieverQueryEngine.from_args(
        retriever=vector_retriever_all, llm=llm, node_postprocessors=[colbert_reranker])

    system_prompt = """Anda adalah chatbot yang dapat membantu menjawab pertanyaan tentang berbagai jenis regulasi di Indonesia.

    Anda HARUS selalu menjawab dengan BAHASA QUERY.
    Anda TIDAK DIBENARKAN berimajinasi.
    Anda HANYA DIBENARKAN menjawab pertanyaan berdasarkan dokumen yang telah Anda pelajari.
    JANGAN MENGGUNAKAN informasi dari luar dokumen yang telah Anda pelajari.
    Anda memiliki akses ke query tools untuk membantu Anda menemukan informasi yang relevan di basis data regulasi.
    Berdasarkan informasi konteks dan bukan pengetahuan sebelumnya, jawablah pertanyaan HARUS menggunakan query tools yang tersedia.
    Gunakan riwayat percakapan sebelumnya atau konteks di atas untuk berinteraksi dan membantu pengguna.

    **Penjelasan Metadata:**
    Metadata dokumen mencakup informasi berikut:
    - **file_name**: Nama file dokumen
    - **title**: Judul dokumen
    - **sector**: Sektor yang dicakup oleh regulasi
    - **subsector**: Subsektor yang dicakup oleh regulasi
    - **regulation_type**: Jenis regulasi (misalnya, Surat Edaran OJK, Peraturan OJK)
    - **regulation_number**: Nomor regulasi
    - **effective_date**: Tanggal berlakunya regulasi

    ----------------------------------------------
    Query: {query_str}
    Jawaban:
    """

    memory = ChatMemoryBuffer.from_defaults(token_limit=10000,)

    query_tools = [
        QueryEngineTool(
            query_engine=query_engine_all,
            metadata=ToolMetadata(
                name="bi_ojk",
                description="Useful for retrieving answers from all documents.",
            )
        ),
    ]

    # agent = ReActAgent.from_tools(
    #     llm=llm,
    #     memory=memory,
    #     tools=query_tools,
    #     verbose=True,
    #     system_prompt=system_prompt,
    # )

    agent = CondenseQuestionChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        query_engine=query_engine_all,
        verbose=True,
    )

    return agent


agent = load_agent()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = agent

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
