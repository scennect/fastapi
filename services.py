#from pathlib import Path
#from typing import Optional 
import schemas as _schemas

import torch 
#from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
#from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image
import os
from dotenv import load_dotenv

import boto3
import io
import uuid

from fastapi import HTTPException

import botocore
import requests
import base64



load_dotenv()

# Get the token from HuggingFace 
HF_TOKEN = os.getenv('HF_TOKEN')
BUCKET_NAME = os.getenv('BUCKET_NAME')


# S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

async def upload_to_s3(image: Image, bucket_name: str, s3_client) -> str:
    # 이미지 메모리 스트림에 저장
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    
    # 파일 이름 생성
    file_name = f"{str(uuid.uuid4())}.png"
    
    try:
        # S3에 비동기로 업로드
        s3_client.upload_fileobj(
            memory_stream, 
            bucket_name, 
            file_name, 
            ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
        )
    except botocore.exceptions.ClientError as e:
        # S3 업로드 실패 시 예외 처리 (올바른 HTTPException 사용)
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    return image_url



def request_prompt(prompt : str)->str:
    txt_sd = "라는 내용으로 stable diffusion 2.0 모델을 써서 그리고 싶습니다."
    txt_limit = "200자 이내의 프롬프트로 이루어진 문장으로 답변해주세요. 영어로 부탁합니다."
    
    return prompt+txt_sd+txt_limit
#//////////////////////////////////////////////////////////////////////////////////////////

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



# 세션 및 재시도 설정
session = requests.Session()
retry = Retry(
    total=5,  # 총 5번의 재시도
    connect=3,  # 연결 실패 시 3번 재시도
    backoff_factor=0.5,  # 재시도 간격 (0.5초, 1초, 2초 등 지수 증가)
    status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 HTTP 상태 코드
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def connect_txt2img(imgPrompt: _schemas.ImageCreate) ->Image:
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    
    payload = {
        
        "negative_prompt" :" negativeXL_D",
        "prompt": imgPrompt.prompt,
        "sampler_name" :"DPM++ SDE",
        "scheduler" : "Karras",
        "steps": 8,
        "cfg_scale": 2,
        "width": 512,
        "height": 512
    }
    
    try:
        # Send request to the Stable Diffusion API
        response = session.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        # Decode the base64 image returned by the API
        image_base64 = response.json()["images"][0]
        image_data = base64.b64decode(image_base64)
        
        # Convert to BytesIO for streaming
        image = Image.open(io.BytesIO(image_data))
        
        return image
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request to Stable Diffusion API failed: {e}")
    
def connect_img2img(img_url:str, imgPrompt: _schemas.ImageCreate)->Image:
    url = "http://127.0.0.1:7860/sdapi/v1/img2img"
    
    # Download the initial image from the given URL
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # Check for HTTP errors
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")
    
    # Convert downloaded image to base64
    init_image = Image.open(io.BytesIO(response.content)).convert("RGB").resize((512,512))
    buffer = io.BytesIO()
    init_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    payload = {
        "negative_prompt" :" negativeXL_D",
        "prompt": imgPrompt.prompt,
        "sampler_name" :"DPM++ SDE",
        "scheduler" : "Karras",
        "steps": 8,
        "cfg_scale": 2,
        "width": 512,
        "height": 512,
        "init_images": [image_base64]
    } 
    try:
        # Send the request to the Stable Diffusion API
        response = session.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        # Decode the base64 image returned by the API
        result_image_base64 = response.json()["images"][0]
        result_image_data = base64.b64decode(result_image_base64)
        
        # Open the result as a PIL Image
        result_image = Image.open(io.BytesIO(result_image_data))
        
        return result_image
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request to Stable Diffusion API failed: {e}")
    
