from pathlib import Path
from typing import Optional 
import schemas as _schemas

import torch 
from peft import PeftModel
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
BUCKET_NAME = os.getenv('BUCKET_NAME')

#model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stabilityai/stable-diffusion-2"
#model_id = "stabilityai/stable-diffusion-xl-base-1.0"
#lora_model_path = "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur"
# Create the pipe 
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    revision="fp16", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=HF_TOKEN,
    safety_checker = None,
    requires_safety_checker = False
)
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=HF_TOKEN,
    safety_checker = None,
    requires_safety_checker = False
)



# 디바이스 설정
if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)
pipe2.to(device)

# lora 설정
#pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model_path, torch_dtype=torch.float16)


# S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)


async def generate_image(imgPrompt: _schemas.ImageCreate, image: Optional[Image]=None) -> Image: 
    # Stable Diffusion은 실제로 비동기를 지원하지 않지만, 함수 구조를 일관되게 유지합니다.
    generator = None if imgPrompt.seed is None else torch.Generator(device=device).manual_seed(int(imgPrompt.seed))

    if image:
        result_img : Image = pipe2(
            prompt=imgPrompt.prompt,
            negative_prompt=imgPrompt.negative_prompt,
            image=image,
            strength=imgPrompt.strength,
            num_inference_steps=imgPrompt.num_inference_steps,
            guidance_scale=imgPrompt.guidance_scale,
            generator=generator
        ).images[0]
    else:
        result_img : Image = pipe(
            prompt=imgPrompt.prompt,
            negative_prompt=imgPrompt.negative_prompt,
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






