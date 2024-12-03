import os
import logging
import requests
import base64
import json
import urllib
from io import BytesIO

from PIL import Image

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.utilities import PythonREPL

logger = logging.getLogger(__name__)

FASHION_IMAGE_RETRIEVER_URL = os.getenv("FASHION_IMAGE_RETRIEVER_URL")
LLAVA_URL = os.getenv("LLAVA_URL")
GOOGLE_NAMUWIKI_CSE_ID = os.getenv("GOOGLE_NAMUWIKI_CSE_ID")


NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
import codecs
# ================= tools for openai application =================

# fashion image search
image_cache = {} # query -> {"data": {"image_url_list": [], "image_tag_list": []}, "used": [False]*len(image_url_list)}
def retrieve_fashion_images_with_text_query(text_query=""):

    if (text_query in image_cache) == False:
        try:
            response = requests.get(
                FASHION_IMAGE_RETRIEVER_URL,
                params={
                    "text_query": text_query
                }
            )
            data = response.json()
            image_cache[text_query] = {"data":data, "used":[False]*len(data["image_url_list"])}
        except Exception as e:
            logger.error(e)
            return {
                "image_url_list": [],
                "image_tag_list": [],
            }
    
    for i in range(len(image_cache[text_query]["data"]["image_url_list"])):
        if not image_cache[text_query]["used"][i]:
            image_cache[text_query]["used"][i] = True
            url = image_cache[text_query]["data"]["image_url_list"][i]
            tag = image_cache[text_query]["data"]["image_tag_list"][i]
            if i == len(image_cache[text_query]["used"]) - 1:
                del image_cache[text_query]
            return {"image_url_list":[url], "image_tag_list": [tag]}
    



def retrieve_fashion_images_with_text_query_result_parser(function_response, k=1):
    return [{"image_url": iurl, "image_tag": itags} for iurl, itags in zip(function_response["image_url_list"][:k], function_response["image_tag_list"][:k])]

# google search
google_search = GoogleSearchAPIWrapper()
url_to_snippet_dict = {}
def google_search_fn(query="", k=1):
    k = 3
    search_result = google_search.results(query, k)
    
    print("[***] url_to_snippet_dict:", url_to_snippet_dict)
    
    for result in search_result:
        url_to_snippet_dict[result["link"]] = result["snippet"]
    return search_result

# google image search
def google_image_search_fn(query="", k=1):
    return google_search.results(query, k, {"searchType": "image"})

# google namuwiki search
google_namuwiki_search = GoogleSearchAPIWrapper(google_cse_id=GOOGLE_NAMUWIKI_CSE_ID)
def google_namuwiki_search_fn(text_query="", k=1):
    return google_namuwiki_search.results(text_query, k)

# web page loader
def load_content_from_web_url(url, n=300):
    loader = WebBaseLoader(url)
    docs = loader.load()
    snippet = url_to_snippet_dict[url]
    print("[***] snippet from cache!:", snippet)

    def find_longest_common_substring(s1, s2):
        if not s1 or not s2:
            return ""
        
        # 동적 프로그래밍을 위한 2D 배열 생성
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 가장 긴 부분 문자열의 길이와 끝 위치 추적
        max_length = 0
        end_pos = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
        
        # 가장 긴 공통 부분 문자열 반환
        return s1[end_pos - max_length:end_pos]

    
    def crop_text(text, target_str, n):
        common_substring = find_longest_common_substring(text, target_str)

        start_idx = text.find(common_substring)
        text_start = max(0, start_idx - n)
        text_end = min(len(text), start_idx + len(common_substring) + n)
        
        return text[text_start:text_end]
    
    return [{"page_content": crop_text(doc.page_content, snippet, n),
             "source": doc.metadata["source"]} for doc in docs]

# llava text generation
def get_image_description_from_url(prompt_in_english="What is in this picture?", image_url=""):
    try:
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content))
        buffered = BytesIO()
        image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        HEADERS = {
            "Content-Type": "application/json"
        }
        
        data_object = {
            "model": "llava",
            "prompt":prompt_in_english,
            "images": [img_str],
            "stream": False
        }
        
        response = requests.post(LLAVA_URL, headers=HEADERS, data=json.dumps(data_object))
        if response.status_code == 200:
            response_json = json.loads(response.text)
            return {
                "image_url": image_url,
                "description": response_json["response"],
            }
        else:
            return {
                "image_url": "",
                "description": "",
            }
    except Exception as e:
        logger.error(e)
        return {
            "image_url": "",
            "description": "",
        }
        
# naver shopping search
def naver_shopping_item_search(query, display=3, start=1):
    encText = urllib.parse.quote(query)
    
    url = "https://openapi.naver.com/v1/search/shop.json?query=" + encText + "&display=" +str(display) + "&start=" +str(start) # JSON 결과
    
    logger.error(url)
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret",NAVER_CLIENT_SECRET)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)

python_repl = PythonREPL()
def run_python_code(code):
    return python_repl.run(code)


FUNCTION_MAPPER_FOR_OPENAI = {
    "retrieve_fashion_images_with_text_query": {
        "function": retrieve_fashion_images_with_text_query,
        "result_parser": retrieve_fashion_images_with_text_query_result_parser
    },
    # "retrieve_fashion_text_knowledges_with_text_query": {
    #     "function": google_namuwiki_search_fn,
    #     "result_parser": lambda x: x
    # },
    "google_search": {
        "function": google_search_fn,
        "result_parser": lambda x: x
    },
    "google_image_search": {
        "function": google_image_search_fn,
        "result_parser": lambda x: x
    },
    "load_content_from_web_url": {
        "function": load_content_from_web_url,
        "result_parser": lambda x: x
    },
    "get_image_description_from_url": {
        "function": get_image_description_from_url,
        "result_parser": lambda x: x
    },
    "naver_shopping_item_search": {
        "function": naver_shopping_item_search,
        "result_parser": lambda x: x
    },
    "run_python_code": {
        "function": run_python_code,
        "result_parser": lambda x: x
    },
}


TOOL_DESCRIPTION_FOR_OPENAI = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "retrieve_fashion_images_with_text_query",
    #         "description": "retrieve fashion item images relevant to the text query. This tools only support fashion keywords and don't return recent images. Do not use this tool to search for celebrity fashion photos. supported languages: (Korean)",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_query": {
    #                     "type": "string",
    #                     "description": "text description of fashion images to retrieve. supported languages: (Korean)"
    #                 }
    #             },
    #             "required": ["text_query"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "retrieve_fashion_text_knowledges_with_text_query",
    #         "description": "retrieve fashion knowledge link and snippet relevant to the text query. It return fashion knowledge snippet and link, title. So, if you want to get more detailed information, load content of page using other tool. supported languages: (Korean)",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_query": {
    #                     "type": "string",
    #                     "description": "text description of fashion text knowledge to retrieve. supported languages: (Korean)"
    #                 },
    #             },
    #             "required": ["text_query"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search Google for recent results. This tool returns 'title', 'url', 'snippet'. So, if you want whole contents of the page, please use web loading tools. You can use this tools except about fashion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query string for search except about fashion."
                    },
                    "k": {
                        "type": "integer",
                        "description": "the number of results"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_image_search",
            "description": "Search Images for recent results. This tool return link of the images. This tool returns 'title', 'link', 'snippet'. So, if you want whole contents of the page, please use web loading tools. You can also use this tool when you want recent images. You can use this tool to search for celebrity fashion photos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query string for search except about fashion."
                    },
                    "k": {
                        "type": "integer",
                        "description": "the number of results"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_content_from_web_url",
            "description": "load content of web pages. if you want whole content of the web page, use this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url of a web page."
                    },
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_image_description_from_url",
            "description": "Use this tool when the user asks about an image content. It returns description of image given the image url and prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_in_english": {
                        "type": "string",
                        "description": "Only english allowed. A query or question that you want to know in the image."
                    },
                    "image_url": {
                        "type": "string",
                        "description": "The url of an image."
                    },
                },
                "required": ["prompt_in_english", "image_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "naver_shopping_item_search",
            "description": "search shopping item from naver shopping relevant to the text query. You can use this tool when you want to search item in Naver Shopping(네이버 쇼핑).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "text description of fashion item to retrieve. Supported language: Korean"
                    },
                    "display": {
                        "type": "string",
                        "description": "the number of results. default value is 3"
                    },
                    "start": {
                        "type": "string",
                        "description": "The start offset of search. default value is 1"
                    },
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "A python code to run"
                    },
                },
                "required": ["code"]
            }
        }
    },
]
