import logging
import google.generativeai as genai
import os
from services import request_prompt
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro') # 모델 설정

def prompt_api(prompt:str)->str:
    tmp = request_prompt(prompt=prompt)

    response = model.generate_content(tmp)
    logging.info(f"Generated prompt from Gemini AI: {response.text}") # 응답 로그
    
    return response.text