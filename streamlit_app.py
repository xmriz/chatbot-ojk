from dotenv import load_dotenv
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from utils.index_store import load_vector_index
import streamlit_app as st
from utils.models_definer import ModelName
from llama_index.core import Settings
from utils.models_definer import get_llm_and_embedding
from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
import nest_asyncio
import hmac

nest_asyncio.apply()
load_dotenv()

st.set_page_config(page_title="OJK CHATBOT",
                   page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK",
        }
    ]

TOP_K = 3
model_name = ModelName.AZURE_OPENAI

llm, embedding_llm = get_llm_and_embedding(model_name=model_name)

Settings.llm = llm
Settings.embed_model = embedding_llm

qa_prompt = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query asking about banking compliance in Indonesia. 
Answer the question based on the context information.
ALWAYS ANSWER WITH USER'S LANGUAGE.
Please provide your answer with [regulation_number](file_url) in metadata 
(if possible) in the following format:

---------------------
Answer... \n\n
Source: [regulation_number](file_url) \n
---------------------

Query: {query_str}
Answer: \
"""

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


@st.cache_resource(show_spinner=False)
def load_chat_engines():
    vector_index = load_vector_index()
    vector_retriever = vector_index.as_retriever(similarity_top_k=TOP_K)

    qa_prompt_tmpl = PromptTemplate(qa_prompt)

    vector_query_engine = RetrieverQueryEngine.from_args(
        retriever=vector_retriever, llm=llm, streaming=True)

    vector_query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=10000,
    )

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        query_engine=vector_query_engine,
        verbose=True,
    )

    return chat_engine


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = load_chat_engines()

if not check_password():
    st.stop() 

st.title("Chat with the OJK BOT 💬🤖")

if prompt := st.chat_input(
    "Ask me a question about any Banking Compliance in Indonesia"
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

# Add Reset button
if st.button("Reset"):
    st.session_state.chat_engine.reset()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK",
        }
    ]
    st.experimental_rerun()
