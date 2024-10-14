import google.generativeai as genai
import os
from services import request_prompt
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro') # 모델 설정

def prompt_api(prompt:str)->str:
    response = model.generate_content(request_prompt(prompt=prompt))
    
    return response.text