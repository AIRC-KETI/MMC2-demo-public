import dotenv

dotenv.load_dotenv()

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

import streamlit as st

from prompt_langchain import get_chat_chain



# history = StreamlitChatMessageHistory(key="chat_messages")

# history.add_user_message("hi!")
# history.add_ai_message("whats up?")
if llm_chosen := st.radio("Choose LLM", ["llama3-70b-8192-groq", "llama3:8b", "llama3:70b"], index=0):
    print(llm_chosen)

if use_tools := st.checkbox("Use tools?", value=True):
    print(llm_chosen)

msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("안녕하세요. 무엇이 궁금하신가요?")

if "image_dict" not in st.session_state:
    print("[*] image_dict initialized!")
    st.session_state["image_dict"]  = {} 


#chain = get_chat_chain(msgs, st)


for i, msg in enumerate(msgs.messages):
    ai_message_container = st.chat_message(msg.type)
    ai_message_container.write(msg.content)
    
    print("[*] image_dict keys()!", st.session_state["image_dict"].keys())

    if i in st.session_state["image_dict"].keys():
        print(i, st.session_state["image_dict"][i])
        ai_message_container.image(st.session_state["image_dict"][i])

from langchain.callbacks.base import BaseCallbackHandler


class StreamingChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_llm_start(self, *args, **kwargs):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.text += token
        self.container.markdown(
            body=self.text,
            unsafe_allow_html=False,
        )

    def on_llm_end(self, response: str, *args, **kwargs):
        pass



if prompt := st.chat_input():
    print("prompt :", llm_chosen)
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    
    config = {"configurable": {"session_id": "any"}}
    
    ai_message = st.chat_message("ai")
    
    chain = get_chat_chain(msgs, ai_message, llm_chosen, use_tools)
    
    #stream = chain.stream({"question": prompt}, config)
    #ai_message.write_stream(stream)
    
    response = chain.invoke({"question": prompt}, config)

    print(response["output"])
    ai_message.markdown(response["output"].content)
    if "found_image_path" in response["meta_result"]:
        ai_message.image(response["meta_result"]["found_image_path"])
        st.session_state["image_dict"][len(msgs.messages)-1] = response["meta_result"]["found_image_path"]
