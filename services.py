from pathlib import Path
from typing import Optional 
import schemas as _schemas

import torch 
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
import os
from dotenv import load_dotenv

import boto3
import io
import uuid
import requests

load_dotenv()

# Get the token from HuggingFace 
"""
Note: make sure .env exist and contains your token
"""
HF_TOKEN = os.getenv('HF_TOKEN')

# Create the pipe 
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
    )

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)



#------------------------------------------------------------------------
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = 'hongik-s3'

def generate_image(pipe, imgPrompt: _schemas.ImageCreate, init_image: Optional[Image] = None, strength: float = 0.75) -> Image: 
    # 시드가 설정되어 있을 경우 생성기 설정 
    generator = None if imgPrompt.seed is None else torch.Generator(device=device).manual_seed(int(imgPrompt.seed))
    # 이미지를 생성하거나 수정
    if init_image:
        # 이미지 수정
        modified_img : Image = pipe(
            prompt=imgPrompt.prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps=imgPrompt.num_inference_steps,
            guidance_scale=imgPrompt.guidance_scale,
            generator=generator
        ).images[0]
    else:
        # 새로운 이미지 생성
        modified_img : Image = pipe(
            prompt=imgPrompt.prompt,
            num_inference_steps=imgPrompt.num_inference_steps,
            guidance_scale=imgPrompt.guidance_scale,
            generator=generator
        ).images[0]

    return modified_img

def upload_to_s3(image: Image, bucket_name: str, s3_client) -> str:
    # 이미지 메모리 스트림에 저장
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    
    # 파일 이름 생성
    file_name = f"{str(uuid.uuid4())}.png"
    s3_client.upload_fileobj(
        memory_stream, 
        bucket_name, 
        file_name, 
        ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'})
    
    # S3 URL 반환
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    return image_url

async def txt2img(imgPrompt: _schemas.ImageCreate) -> str:
    # 이미지 생성
    image = await generate_image(pipe,imgPrompt)
    # S3에 업로드 및 URL 반환
    s3_url = upload_to_s3(image, BUCKET_NAME, s3_client)
    return s3_url


async def img2img(img_url: str, imgPrompt: _schemas.ImageCreate) -> str:
    # 이미지 URL에서 이미지 로드
    response = requests.get(img_url)
    initial_img = Image.open(io.BytesIO(response.content))

    # 이미지 수정 (init_image를 전달하여 호출)
    modified_img = generate_image(pipe, imgPrompt, init_image=initial_img, strength=0.75)

    # S3에 업로드
    modified_image_url = upload_to_s3(modified_img, BUCKET_NAME, s3_client)
    return modified_image_url



    
    