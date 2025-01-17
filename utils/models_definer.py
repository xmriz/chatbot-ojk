from enum import Enum
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler


class ModelName(Enum):
    # OLLAMA = 'ollama'
    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'


def get_llm_and_embedding(model_name: ModelName, api_key: str = None, api_version: str = None, api_endpoint: str = None):
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # if model_name == ModelName.OLLAMA:
    #     llm = Ollama(
    #         model='llama3',
    #         temperature=0,
    #         callback_manager=callback_manager,
    #     )
    #     embedding_llm = OllamaEmbedding(model_name='llama3')

    if model_name == ModelName.OPENAI:
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        llm = OpenAI(
            api_key=api_key,
            model='gpt-3.5-turbo',
            temperature=0.0,
            callback_manager=callback_manager,
        )
        embedding_llm = OpenAIEmbedding(api_key=api_key)

    elif model_name == ModelName.AZURE_OPENAI:
        if not api_key or not api_version or not api_endpoint:
            raise ValueError(
                "One or more Azure OpenAI environment variables (AZURE_OPENAI_KEY, API_VERSION, AZURE_OPENAI_ENDPOINT) are not set")

        llm = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            azure_deployment='gpt-35-crayon',
            temperature=0,
            callback_manager=callback_manager,
        )
        embedding_llm = AzureOpenAIEmbedding(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            azure_deployment='embedding-ada-crayon',
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    Settings.llm = llm
    Settings.embed_model = embedding_llm

    return llm, embedding_llm
