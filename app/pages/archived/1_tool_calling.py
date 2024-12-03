import os
import json
import copy
import sys
import logging
import inspect

from openai import AzureOpenAI
import streamlit as st

# add parent dir to the system path
sys.path.insert(0, '..')

from tools import (
    TOOL_DESCRIPTION_FOR_OPENAI as tools,
    FUNCTION_MAPPER_FOR_OPENAI as FUNCTION_MAPPER
)


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_NAME = "gpt-4o"


client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2024-02-01"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


system_prompt = """
너는 fashion 전문가야.
사용자의 입력에 최선을 다해서 답을 해야해.
너는 활용할 수 있는 외부 툴을 최대한 활용하여 대답을 해야해.
활용할 툴이 없는 경우에는 스스로 아는 만큼 최선을 다해서 대답을 해야해.
사용자 입력의 의도를 고려하고 기존 대화 기록을 활용하여 적절하게 대답을 하도록해.
사용자가 구체적으로 요청하지 않는다면 기본적으로 해당 정보를 자세하게 찾아보고 요약하여 대답해.
"""


def openai_stream(response, content_list=[], gather=5):
    g_cnt = 0
    temp_str = ""
    for chunk in response:
        if len(chunk.choices)>0 and chunk.choices[0].delta.content is not None:
            chunk_ct = chunk.choices[0].delta.content
            content_list.append(chunk_ct)
            
            temp_str+=chunk_ct
            if g_cnt >= 5:
                g_cnt = 0
                yield temp_str
                temp_str = ""
    if temp_str:
        yield temp_str


def generate_chat_msg():
    # create messages
    messages = copy.deepcopy(st.session_state.messages)
    
    with st.spinner('Generate response'):
        # get response from gpt
        response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages,
                tools=tools,
            )
        # append a message to session state and write the message on the frontend
        resp_msg = response.choices[0].message
        logger.error(resp_msg)
        return resp_msg


def generate_msg_with_function_call():
    resp_msg = generate_chat_msg()
    
    if resp_msg.tool_calls is not None:
        st.session_state.messages.append({"role": "assistant", "tool_calls": resp_msg.tool_calls})
        st.chat_message("function call").write([{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments} for tc in resp_msg.tool_calls])
        for tool_call in resp_msg.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)
            with st.spinner(f'Wait for function call: {tool_name}'):
                function_response = FUNCTION_MAPPER[tool_name]["function"](**tool_arguments)
                function_result = FUNCTION_MAPPER[tool_name]["result_parser"](function_response)
                logger.error(function_result)
                st.session_state.messages.append(
                    {
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result, ensure_ascii=False),
                        "name": tool_name
                    })
                st.chat_message("tool").write(function_result)
        generate_msg_with_function_call()
    else:
        st.session_state.messages.append({"role": resp_msg.role, "content": resp_msg.content})
        st.chat_message(resp_msg.role).write(resp_msg.content)


st.title("💬 Tool Calling")
st.caption("🚀 A chatbot demo powered by OpenAI")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        if "content" in msg:
            st.chat_message(msg["role"]).write(msg["content"])
        elif "tool_calls" in msg:
            st.chat_message("function call").write([{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments} for tc in msg["tool_calls"]])
    elif msg["role"] == "tool":
        st.chat_message("tool").write(msg["content"])

if prompt := st.chat_input():
    # append a message to session state and write the message on the frontend
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    generate_msg_with_function_call()
        
    
    # # streaming response
    # # get response from gpt
    # response = client.chat.completions.create(
    #         model=MODEL_NAME, 
    #         messages=messages,
    #         tools=tools,
    #         stream=True,
    # )
    # content_buffer = []
    # st.chat_message("assistant").write_stream(openai_stream(response, content_buffer))
    # st.session_state.messages.append({"role": "assistant", "content": "".join(content_buffer)})
        
        
        