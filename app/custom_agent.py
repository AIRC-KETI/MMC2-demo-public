
import json
import requests
from urllib.parse import urljoin


class CustomAgent(object):
    def __init__(self, req_url, chat_query="/api/chat", generate_query="/api/generate") -> None:
        self.req_url = req_url
        self.chat_query = chat_query
        self.generate_query = generate_query
        self.header = {'Content-Type': 'application/json; charset=utf-8'} 
        
    def chat_completions(self, messages, tools=None):
        kwargs = {
            "messages": messages, 
            "tools": tools,
            "tool_choice": "auto",
        }
        data = json.dumps(
            kwargs
        )
        
        response = requests.post(urljoin(self.req_url, self.chat_query), data=data, headers=self.header)
        if response.status_code != 200:
            print({"choices": [{"message": "Error!"}]})

        return response.json()







