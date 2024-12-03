import os

import streamlit as st

st.title("👚KETI-패션 이미지 검색 채팅👗")

st.markdown("""
# 챗봇 설명
- 이 챗봇은 패션에 대해 열심히 공부했습니다.
- 챗봇에게 패션에 대한 질문을 해보세요. 챗봇은 용어를 설명해주거나, 관련이미지나 상품을 찾아줄 수 있습니다. 아니면 패션에 대한 잡담도 괜찮습니다!
- 챗봇과 대화하며 **자연스럽게 말하는지**, **패션을 잘 공부했는지** 살펴봐주세요.
- 챗봇에게 어떻게 질문을 하면 좋을지 모르겠다면 좌측 **'대화 예시'** 페이지를 참고해주세요!
- 좌측의 **'채팅하기'** 페이지에서 채팅을 시작할 수 있습니다.
""")
        
st.image("./images/survey_instruction_examples.webp")
st.image("./images/survey_instruction_start_chat.webp")