from pathlib import Path
from typing import Optional 
import schemas as _schemas

import torch 
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL.Image import Image
import os
from dotenv import load_dotenv

import boto3
import io
import uuid

from fastapi import HTTPException

import botocore




load_dotenv()

# Get the token from HuggingFace 
HF_TOKEN = os.getenv('HF_TOKEN')

# Create the pipe 
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=HF_TOKEN
)
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=HF_TOKEN
)

# 디바이스 설정
if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)
pipe2.to(device)

# S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = 'hongik-s3'

async def generate_image(imgPrompt: _schemas.ImageCreate, image: Optional[Image], strength: float = 0.75) -> Image: 
    # Stable Diffusion은 실제로 비동기를 지원하지 않지만, 함수 구조를 일관되게 유지합니다.
    generator = None if imgPrompt.seed is None else torch.Generator(device=device).manual_seed(int(imgPrompt.seed))

    if image:
        result_img : Image = pipe2(
            prompt=imgPrompt.prompt,
            image=image,
            strength=strength,
            num_inference_steps=imgPrompt.num_inference_steps,
            guidance_scale=imgPrompt.guidance_scale,
            generator=generator
        ).images[0]
    else:
        result_img : Image = pipe(
            prompt=imgPrompt.prompt,
            num_inference_steps=imgPrompt.num_inference_steps,
            guidance_scale=imgPrompt.guidance_scale,
            generator=generator
        ).images[0]

    return result_img

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


'''
async def img2img(img_url: str, imgPrompt: _schemas.ImageCreate) -> str:
    try:
        # 비동기로 이미지 다운로드
        async with aiohttp.ClientSession() as session:
            async with session.get(img_url) as response:
                if response.status != 200:
                    raise HTTPException(400, "Failed to fetch image from URL")
                image_data = await response.read()
                initial_img = Image.open(io.BytesIO(image_data))
                initial_img.verify()  # 이미지 검증
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch or verify image from URL: {str(e)}")

    try:
        # 이미지 수정 (init_image를 전달하여 호출)
        modified_img = await generate_image(pipe, imgPrompt, init_image=initial_img, strength=0.75)
        # S3 업로드
        modified_image_url = await upload_to_s3(modified_img, BUCKET_NAME, s3_client)
        return modified_image_url
    except botocore.exceptions.ClientError as e:
        raise HTTPException(500, f"S3 upload failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Image modification failed: {str(e)}")
'''



