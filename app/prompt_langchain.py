import os
import requests
import dotenv

dotenv.load_dotenv()

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
# import chat_image_retriever

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.chains import LLMChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables.base import RunnableSequence

llm_groq = ChatGroq(
        # api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.1
    )
    
# llm = ChatOllama(model="llava", base_url="http://10.0.0.31:11436", temperature=0.1)
# llm = ChatOllama(model="solar", base_url="http://10.0.0.31:11435", temperature=0.1)
llm_llama3 = ChatOllama(model="llama3", base_url="http://10.0.0.31:11434", temperature=0.1)
llm_llama3_70b = ChatOllama(model="llama3:70b", base_url="http://10.0.0.32:11434", temperature=0.1)

def get_tavily_search_chain(search_keyword, st, llm):
    context = TavilySearchResults(max_results=3)(search_keyword)
    print("tavily search result:", context)
    prompt = PromptTemplate.from_template("""Answer the user's last question of the following chat history.
Chat History:
{history}
{question}

External search context:
{context} 

- Your Task
    - Your answer should reflect the above search result.
    - You should use Korean.
    - Do not include any preamble.
""",
partial_variables={'context':context})

    # text_search_chain = prompt|llm
    if st is not None:
        for c in context:
            st.write(c['url'])

    text_search_chain = RunnableParallel(output=(prompt|llm),\
                        meta_result=RunnablePassthrough())
    
    return text_search_chain

# retriever = chat_image_retriever.get_retriever()
FASHION_IMAGE_RETRIEVER_URL = os.environ["FASHION_IMAGE_RETRIEVER_URL"]
def retrieve_fashion_images_with_text_query(text_query=""):
    try:
        
        response = requests.get(
            FASHION_IMAGE_RETRIEVER_URL,
            params={
                "text_query": text_query
            }
        )
        print(response.json())
        return response.json()
    except Exception as e:
        print("Error", e)
        return {
            "image_url_list": [],
            "image_tag_list": [],
        }

def get_image_search_chain(search_keyword, llm):
    # search_results = retriever.invoke(search_keyword)
    # print("Search key:", search_keyword)
    # print("Found key:", search_results[0].page_content)
    # print("Found Image URL:", search_results[0].metadata["image_path"])
    search_results = retrieve_fashion_images_with_text_query(search_keyword)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""You are AI chatbot who talk with a user. You found an image the user requested and the corresponding text description.

- Your Task
    - Explain why this image suits the request.
    - You should use Korean in 3 sentences.
    - Do not include any preamble.

Text Description:
{search_results["image_tag_list"][0]}

Your Answer in Korean (한국말):

"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    image_search_chain = RunnableParallel(output=prompt|llm,\
                        meta_result=RunnablePassthrough().assign(found_image_path=lambda x:search_results["image_url_list"][0]))

    return image_search_chain


def get_off_topic_reply_chain(llm):

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human. You should use Korean and do not include Chinese or Japanese in your answer."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


    off_topic_chain = RunnableParallel(output=prompt|llm, \
                        meta_result=RunnablePassthrough())
    return off_topic_chain


def get_function_call_detection_chain(msgs, llm):
    tool_names = ['fashion_image_search', 'fashion_text_search', 'general_web_search', 'off_topic']

    class Action(BaseModel):
        function: str = Field(description="function to use. it should be one of "+", ".join(tool_names))
        function_param: str = Field(description="function param for the chosen function")

    parser = JsonOutputParser(pydantic_object = Action)

    template = """You are a master at understanding a user's intent and calling a proper function.\n \
Chat history:    
{history}
{question}

Classify the intent of the last user message and choose a function: \n \
    fashion_image_search - used when a user wants fashion images. you need to provide an informative sentence in Korean to query the search engine (한국어 문장). Do not add word meaning "photo".\n \
    fashion_text_search - used when a user wants fashion related information. you need to provide informative comma-seperated key words. \n \
    general_web_search - used when a user wants fact critical answer you can not handle directly. you need to provide informative comma-seperated key words.
    off_topic - when it doesnt relate to any other intents  \n\

Keep in mind that the function should be in the above list.
Do not include any preamble and just output a json object.
{format_instructions}

Answer:"""

    prompt_template = PromptTemplate(
        input_variables=["history"],
        partial_variables={"format_instructions":parser.get_format_instructions()},
        template=template
    )


    chain = prompt_template|llm|parser

    return chain

from functools import partial
def route(x,st=None, llm=None):
    print(x)
    st = x["streamlit"]
    function_name = x["function_call"]["function"]
    if function_name == "fashion_text_search" or function_name == "general_web_search":
        #st.status("fashion_text_search")
        search_keyword = x["function_call"]["function_param"]
        print("Route:", function_name )
        if st is not None:
            with st.status(f"From web search ({function_name}) ...", expanded=True) as status:
                status.write(f"Search keywords: {search_keyword}")
                return get_tavily_search_chain(search_keyword, status, llm).with_listeners(on_end=lambda run_obj : status.update(label="Web search complete!", state="complete"))# , expanded=False
        else:
            return get_tavily_search_chain(search_keyword, None, llm)
        #return # |RunnableLambda(lambda x:st.status("Done"))

    elif function_name == "fashion_image_search":
        #st.status("fashion_image_search")
        search_keyword = x["function_call"]["function_param"]
        print("Route: fashion_image_search")
        if st is not None:
            with st.status(f"Fashion image search", expanded=True) as status:
                status.write(f"Search keywords: {search_keyword}")
                return get_image_search_chain(search_keyword, llm).with_listeners(on_end=lambda run_obj :status.update(label="Image search complete!", state="complete", expanded=False))#|RunnableLambda(lambda x:st.status("Done"))
        else:
            return get_image_search_chain(search_keyword, llm)

    
    elif function_name == "off_topic":
        #st.status("off_topic")
        print("Route: off_topic")
        if st is not None:
            with st.status("Off topic...") as status:
                return get_off_topic_reply_chain(llm).with_listeners(on_end=lambda run_obj :status.update(label="Complete!", state="complete", expanded=False))# |RunnableLambda(lambda x:st.status("Done"))
        else:
            return get_off_topic_reply_chain(llm)
    else:
        assert(False)

def get_chat_chain(msgs, st=None, llm_chosen_str="llama3-70b-8192-groq", use_tools=True):
    
    llm_dict = {"llama3-70b-8192-groq":llm_groq,
                "llama3:8b":llm_llama3,
                "llama3:70b":llm_llama3_70b}

    llm = llm_dict[llm_chosen_str]
    chain = None
    if use_tools:
        chain = {'function_call':get_function_call_detection_chain(msgs, llm),\
                'history':lambda x: x["history"],
                'question':lambda x: x["question"],
                "streamlit":lambda x:st}|RunnableLambda(partial(route, st=st, llm=llm))
    else:
        chain = {'history':lambda x: x["history"],
                'question':lambda x: x["question"]}|get_off_topic_reply_chain(llm)

    # prompt = ChatPromptTemplate.from_messages(
    # [
    #     ("system", "You are an AI chatbot having a conversation with a human."),
    #     MessagesPlaceholder(variable_name="history"),
    #     ("human", "{question}"),
    # ])

    # chain= prompt|llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,  # Always return the instance created earlier
        input_messages_key="question",
        history_messages_key="history",
    )


    return chain_with_history