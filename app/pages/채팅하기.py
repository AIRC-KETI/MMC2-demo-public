import os
import time
import dotenv
dotenv.load_dotenv()

import json
import copy
import sys
import logging
import inspect

import streamlit as st

# add parent dir to the system path
sys.path.insert(0, '..')

from tools import (
    TOOL_DESCRIPTION_FOR_OPENAI as tools,
    FUNCTION_MAPPER_FOR_OPENAI as FUNCTION_MAPPER
)
from custom_agent import CustomAgent



KETI_FASHION_CHATBOT_8B_URL = os.getenv("KETI_FASHION_CHATBOT_8B_URL")

client = CustomAgent(KETI_FASHION_CHATBOT_8B_URL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



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
    with st.spinner('답변 생성중...'):
        # get response from keti chatbot
        logger.error(messages)
        while True:
            response = client.chat_completions(
                    messages=messages,
                    tools=tools,
                )
            if "error" in response or response["choices"][0]['finish_reason'] == "length":
                messages = messages[-10:]
                print(" ==> Message truncated")
            else:
                break
        print("response:",response)
        # append a message to session state and write the message on the frontend
        resp_msg = response["choices"][0]["message"]
        logger.error(resp_msg)
        return resp_msg


def display_tool_call(tool_calls):
    STYLE = """
        text-decoration: none;
        color: #0066cc;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 2px 0;
        font-size: 0.9em;
    """
    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        arguments = json.loads(tc["function"]["arguments"])
        if "google_image_search" in tool_name:
            st.markdown(f"🖼️인터넷 이미지 검색...<font color='black' style='{STYLE}'>{arguments['query']}</font>", unsafe_allow_html=True)
        elif "google_search" in tool_name:
            st.markdown(f"🌐인터넷 검색...<font color='black' style='{STYLE}'>{arguments['query']}</font>", unsafe_allow_html=True)
        elif "naver_shopping_item_search" in tool_name:
            st.markdown(f"🛍️인터넷 쇼핑 검색...<font color='black' style='{STYLE}'>{arguments['query']}</font>", unsafe_allow_html=True)
        elif "get_image_description_from_url" in tool_name:
            st.markdown(f"👀이미지 설명 추출중...<font color='black' style='{STYLE}'>{arguments['prompt_in_english']}</font>", unsafe_allow_html=True)
        elif "retrieve_fashion_images_with_text_query" in tool_name:
            st.markdown(f"👗내부 이미지 데이터베이스 검색...<font color='black' style='{STYLE}'>{arguments['text_query']}</font>", unsafe_allow_html=True)
        elif "load_content_from_web_url" in tool_name:
            st.markdown(f"🔍웹 페이지 확인중...`{arguments['url']}`")
        else:
            st.json([{"id": tc["id"], "name": tool_name, "arguments": tc["function"]["arguments"]}], expanded=False)

def display_function_result(function_results):
    # naver shopping item search 결과는 리스트가 아니라 딕셔너리로 반환됨
    if not(type(function_results) is list):
        function_results = [function_results]
    print("function_results:", function_results)
    for r in function_results:
        print("r=>",r, 'title' in r, 'link' in r)
        if ('title' in r) and ('link' in r): #google search, google image search
            title = r['title']
            title = title.replace("[", "%5B")
            title = title.replace("]", "%5D")
            link = r['link']
            st.markdown(f"""
                <a href="{link}" target="_blank" style="
                    text-decoration: none;
                    background-color: #f0f2f6;
                    color: #0066cc;
                    padding: 4px 12px;
                    border-radius: 20px;
                    display: inline-block;
                    margin: 2px 0;
                    font-size: 0.9em;
                    ">{title}</a>
                """, unsafe_allow_html=True)
        elif 'image_url' in r:
            st.image(r['image_url'], width=70)
        else:
            st.json(function_results, expanded=False)

def generate_msg_with_function_call():
    resp_msg = generate_chat_msg()
    
    if "tool_calls" in resp_msg:
        if resp_msg["tool_calls"] is not None:
            st.session_state.messages.append({"role": "assistant", "tool_calls": resp_msg["tool_calls"], "content": None})
            display_tool_call(resp_msg["tool_calls"])
            for tool_call in resp_msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_arguments = json.loads(tool_call["function"]["arguments"])
                with st.spinner(f'함수 실행중...: {tool_name}'):
                    function_response = FUNCTION_MAPPER[tool_name]["function"](**tool_arguments)
                    print("Function call!", tool_name, tool_arguments)
                    function_result = FUNCTION_MAPPER[tool_name]["result_parser"](function_response)

                    function_result = function_result if type(function_result) is str else json.dumps(function_result, ensure_ascii=False)
                    function_result_json = json.loads(function_result)

                    logger.error(function_result)
                    function_result_json = json.loads(function_result)
                    st.session_state.messages.append(
                        {
                            "role": "tool", 
                            "tool_call_id": tool_call["id"],
                            "content": function_result,
                            "name": tool_name
                        })
                    display_function_result(function_result_json)
            generate_msg_with_function_call()
    else:
        st.session_state.messages.append({"role": resp_msg["role"], "content": resp_msg["content"]})
        display_message(resp_msg["role"], resp_msg["content"], stream=True)
        
def display_message(role, content, stream=False):
    def stream_data():
        for word in content.split(" "):
            yield word + " "
            time.sleep(0.05)
    if stream == True:
        st.chat_message(role).write_stream(stream_data)
    else:
        st.chat_message(role).write(content)
    
    # c1, c2 = st.columns((4, 1))
    # with c1:
    #     st.chat_message(role).write(content)
    # with c2:
    #     st.button("수정", key=f"edit_{len(st.session_state.messages)-1}")

st.title("👚KETI-패션 이미지 검색 채팅👗")
st.markdown("""
- 패션 이미지는 주로 **여성 성인복**을 검색할 수 있습니다.
- 이렁 질문으로 한번 시작해보세요.
    - [이미지 검색] 코듀로이 소재의 베이지 조거팬츠 사진 찾아줘. / 체크셔츠 코디 추천해줘
    - [쇼핑 검색] 오버사이즈 양털 후드 점퍼를 사고 싶어. / 스웨이드 로퍼는 어디서 살 수 있어?
    - [정보 질문] 셔링이랑 레이스의 차이가 뭐야? / 극세사는 뭘로 만드는 거야?
    - [잡담] 상하의를 모두 스프라이트 패턴으로 맞춰입으면 웃길까?
- 새로고침(F5)으로 대화를 재시작 할 수 있습니다.
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] in ["user", "assistant"]:
        if "content" in msg and msg["content"] is not None:
            display_message(msg["role"], msg["content"])
        elif "tool_calls" in msg:
            display_tool_call(msg["tool_calls"])
    elif msg["role"] == "tool":
        display_function_result(json.loads(msg["content"]))



if prompt := st.chat_input("메세지를 입력하세요."):
    # append a message to session state and write the message on the frontend
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt, stream=False)

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
        
        
if st.button("마지막 메시지 삭제 ↩️"):
    if st.session_state.messages:  # if messages list is not empty
        st.session_state.messages.pop()
        st.rerun()
