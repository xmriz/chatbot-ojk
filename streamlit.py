from dotenv import load_dotenv
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.query_engine import RetrieverQueryEngine
from utils.fusion_retriever import FusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from utils.vector_store import (store_vector_index, load_vector_index)
from utils.doc_store import store_docstore, load_docstore
from utils.node_parser import parse_nodes
from utils.documents_reader import read_documents
import streamlit as st
import logging
import sys
from utils.models_define import ModelName
from llama_index.core import Settings
from utils.models_define import get_llm_and_embedding
from llama_index.agent.openai import OpenAIAgent

import nest_asyncio
nest_asyncio.apply()


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
    load_dotenv()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # storing or not
    STORE = False
    DELETE = False
    model_name = ModelName.AZURE_OPENAI

    # ollama/openai/azure_openai
    llm, embedding_llm = get_llm_and_embedding(model_name=model_name)

    Settings.llm = llm
    Settings.embed_model = embedding_llm

    path = './data'

    if STORE:
        docs_all = read_documents(path=path)
        nodes_all = parse_nodes(documents=docs_all, llm=llm)
        docstore = store_docstore(documents=docs_all, delete=DELETE)
        index_all = store_vector_index(
            nodes=nodes_all, embed_model=embedding_llm, delete=DELETE)
    else:
        index_all = load_vector_index()
        docstore = load_docstore()

    vector_retriever_all = index_all.as_retriever(similarity_top_k=5)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=5, verbose=True
    )

    fusion_retriever = FusionRetriever(
        llm=llm,
        retrievers=[vector_retriever_all, bm25_retriever],
        similarity_top_k=5,
    )

    query_engine_all = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever, llm=llm)

    query_tools = [
        QueryEngineTool(
            query_engine=query_engine_all,
            metadata=ToolMetadata(
                name="bi_ojk",
                description="Useful for retrieving answers from all documents.",
            )
        ),
    ]

    system_prompt = """
    You are a chatbot able to help answer questions about various types of regulations in Indonesia.

    You MUST always answer WITH BAHASA INDONESIA.
    You are NOT ALLOWED to hallucinate.
    You MUST FOLLOW the answer from query engine.
    You are ONLY ALLOWED to answer the question based on documents that you have been trained on.
    You have access to a query tool to help you find relevant information in the regulations database.
    Given the context information and not prior knowledge, answer the query using the query tool provided.
    Use the previous chat history, or the context above, to interact and help the user.
    ----------------------------------------------
    Query: {query_str}
    Answer:
    """

    chat_store = SimpleChatStore()
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=10000,
        chat_store=chat_store,
        chat_store_key="user1",
    )

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    agent = ReActAgent.from_tools(
        llm=llm,
        memory=memory,
        tools=query_tools,
        verbose=True,
        system_prompt=system_prompt,
        callback_manager=callback_manager,
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
