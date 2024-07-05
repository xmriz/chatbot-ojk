from llama_index.core.agent import FnAgentWorker
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core import ChatPromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.program import FunctionCallingProgram
from typing import Dict, Any, Tuple
import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall




from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding


from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker


# =================================================================================================



# vector_memory = VectorMemory.from_defaults(
#     vector_store=None,  # leave as None to use default in-memory vector store
#     embed_model=OpenAIEmbedding(),
#     retriever_kwargs={"similarity_top_k": 5},
# )

# chat_memory_buffer = ChatMemoryBuffer.from_defaults()

# composable_memory = SimpleComposableMemory.from_defaults(
#     primary_memory=chat_memory_buffer,
#     secondary_memory_sources=[vector_memory],
# )


# llm = OpenAI(model="gpt-3.5-turbo-0613")
# agent_worker = FunctionCallingAgentWorker.from_tools(
#     [multiply_tool, mystery_tool], llm=llm, verbose=True
# )
# agent = agent_worker.as_agent(memory=composable_memory)


def agent_with_memory(llm, tools, memory, system_prompt):
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools, llm=llm, verbose=True, system_prompt=system_prompt, 
    )
    agent = agent_worker.as_agent(memory=memory)
    return agent


# =================================================================================================


class YourOpenAIAgent:
    def __init__(
        self,
        llm,
        tools: Sequence[BaseTool] = [],
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )


# # =================================================================================================
# from typing import Dict, Any, List, Tuple, Optional
# from llama_index.core.tools import QueryEngineTool
# from llama_index.core.program import FunctionCallingProgram
# from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.core import ChatPromptTemplate
# from llama_index.core.selectors import PydanticSingleSelector
# from llama_index.core.bridge.pydantic import Field, BaseModel
# from llama_index.core.bridge.pydantic import PrivateAttr
# from llama_index.core.llms import ChatMessage, MessageRole
# from llama_index.core.agent import FnAgentWorker

# DEFAULT_PROMPT_STR = """
# Use language based on the user's language.

# Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
#     a modified question that will not trigger the error.

# Examples of modified questions:
# - The question itself is modified to elicit a non-erroneous response
# - The question is augmented with context that will help the downstream system better answer the question.
# - The question is augmented with examples of negative responses, or other negative questions.

# An error means that either an exception has triggered, or the response is completely irrelevant to the question.

# Please return the evaluation of the response in the following JSON format.

# """


# def get_chat_prompt_template(
#     system_prompt: str, current_reasoning: Tuple[str, str]
# ) -> ChatPromptTemplate:
#     system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
#     messages = [system_msg]
#     for raw_msg in current_reasoning:
#         if raw_msg[0] == "user":
#             messages.append(
#                 ChatMessage(role=MessageRole.USER, content=raw_msg[1])
#             )
#         else:
#             messages.append(
#                 ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1])
#             )
#     return ChatPromptTemplate(message_templates=messages)


# class ResponseEval(BaseModel):
#     """Evaluation of whether the response has an error."""

#     has_error: bool = Field(
#         ..., description="Whether the response has an error."
#     )
#     new_question: str = Field(..., description="The suggested new question.")
#     explanation: str = Field(
#         ...,
#         description=(
#             "The explanation for the error as well as for the new question."
#             "Can include the direct stack trace as well."
#         ),
#     )


# def retry_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
#     """Retry agent.

#     Runs a single step.

#     Returns:
#         Tuple of (agent_response, is_done)

#     """
#     task, router_query_engine = state["__task__"], state["router_query_engine"]
#     llm, prompt_str = state["llm"], state["prompt_str"]
#     verbose = state.get("verbose", False)

#     if "new_input" not in state:
#         new_input = task.input
#     else:
#         new_input = state["new_input"]

#     # first run router query engine
#     response = router_query_engine.query(new_input)

#     # append to current reasoning
#     state["current_reasoning"].extend(
#         [("user", new_input), ("assistant", str(response))]
#     )

#     # Then, check for errors
#     # dynamically create pydantic program for structured output extraction based on template
#     chat_prompt_tmpl = get_chat_prompt_template(
#         prompt_str, state["current_reasoning"]
#     )
#     llm_program = FunctionCallingProgram.from_defaults(
#         output_cls=ResponseEval,
#         prompt=chat_prompt_tmpl,
#         llm=llm,
#     )
#     # run program, look at the result
#     response_eval = llm_program(
#         query_str=new_input, response_str=str(response)
#     )
#     if not response_eval.has_error:
#         is_done = True
#     else:
#         is_done = False
#     state["new_input"] = response_eval.new_question

#     if verbose:
#         print(f"> Question: {new_input}")
#         print(f"> Response: {response}")
#         print(f"> Response eval: {response_eval.dict()}")

#     # set output
#     state["__output__"] = str(response)

#     # return response
#     return state, is_done


# def custom_agent(llm, query_engine_tools, verbose=False, prompt_str=DEFAULT_PROMPT_STR):
#     router_query_engine = RouterQueryEngine(
#         selector=PydanticSingleSelector.from_defaults(llm=llm),
#         query_engine_tools=query_engine_tools,
#         verbose=verbose,
#     )
#     agent = FnAgentWorker(
#         fn=retry_agent_fn,
#         initial_state={
#             "prompt_str": prompt_str,
#             "llm": llm,
#             "router_query_engine": router_query_engine,
#             "current_reasoning": [],
#             "verbose": verbose,
#         },
#     ).as_agent()
#     return agent


# =================================================================================================

DEFAULT_PROMPT_STR = """
Use language based on the user's language.

Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.

"""


def get_chat_prompt_template(
    system_prompt: str, current_reasoning: Tuple[str, str]
) -> ChatPromptTemplate:
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    messages = [system_msg]
    for raw_msg in current_reasoning:
        if raw_msg[0] == "user":
            messages.append(
                ChatMessage(role=MessageRole.USER, content=raw_msg[1])
            )
        else:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1])
            )
    return ChatPromptTemplate(message_templates=messages)


class ResponseEval(BaseModel):
    """Evaluation of whether the response has an error."""

    has_error: bool = Field(
        ..., description="Whether the response has an error."
    )
    new_question: str = Field(..., description="The suggested new question.")
    explanation: str = Field(
        ...,
        description=(
            "The explanation for the error as well as for the new question."
            "Can include the direct stack trace as well."
        ),
    )


def retry_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Retry agent.

    Runs a single step.

    Returns:
        Tuple of (agent_response, is_done)

    """
    task, router_query_engine = state["__task__"], state["router_query_engine"]
    llm, prompt_str = state["llm"], state["prompt_str"]
    verbose = state.get("verbose", False)

    if "new_input" not in state:
        new_input = task.input
    else:
        new_input = state["new_input"]

    # first run router query engine
    response = router_query_engine.query(new_input)

    # append to current reasoning
    state["current_reasoning"].extend(
        [("user", new_input), ("assistant", str(response))]
    )
    state["chat_history"].append(ChatMessage(role="user", content=new_input))
    state["chat_history"].append(ChatMessage(
        role="assistant", content=str(response)))

    # Then, check for errors
    # dynamically create pydantic program for structured output extraction based on template
    chat_prompt_tmpl = get_chat_prompt_template(
        prompt_str, state["current_reasoning"]
    )
    llm_program = FunctionCallingProgram.from_defaults(
        output_cls=ResponseEval,
        prompt=chat_prompt_tmpl,
        llm=llm,
    )
    # run program, look at the result
    response_eval = llm_program(
        query_str=new_input, response_str=str(response)
    )
    if not response_eval.has_error:
        is_done = True
    else:
        is_done = False
    state["new_input"] = response_eval.new_question

    if verbose:
        print(f"> Question: {new_input}")
        print(f"> Response: {response}")
        print(f"> Response eval: {response_eval.dict()}")

    # set output
    state["__output__"] = str(response)

    # return response
    return state, is_done


class CustomAgent:
    def __init__(self, llm, query_engine_tools, verbose=False, prompt_str=DEFAULT_PROMPT_STR):
        self.llm = llm
        self.query_engine_tools = query_engine_tools
        self.verbose = verbose
        self.prompt_str = prompt_str
        self.chat_history = []

        self.router_query_engine = RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(llm=self.llm),
            query_engine_tools=self.query_engine_tools,
            verbose=self.verbose,
        )
        self.agent = FnAgentWorker(
            fn=retry_agent_fn,
            initial_state={
                "prompt_str": self.prompt_str,
                "llm": self.llm,
                "router_query_engine": self.router_query_engine,
                "current_reasoning": [],
                "chat_history": self.chat_history,
                "verbose": self.verbose,
            },
        ).as_agent()

    def chat(self, message: str) -> str:
        state, is_done = self.agent.run(input_str=message)
        return state["__output__"]

    def reset(self):
        self.chat_history = []
        self.agent = FnAgentWorker(
            fn=retry_agent_fn,
            initial_state={
                "prompt_str": self.prompt_str,
                "llm": self.llm,
                "router_query_engine": self.router_query_engine,
                "current_reasoning": [],
                "chat_history": self.chat_history,
                "verbose": self.verbose,
            },
        ).as_agent()
# =================================================================================================
