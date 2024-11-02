from pathlib import Path
from typing import Optional 
import schemas as _schemas

import torch 
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
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

# model_id = "CompVis/stable-diffusion-v1-4"
#model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id= "stabilityai/sdxl-turbo"
# Create the pipe 
'''
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    revision="fp16", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #torch_dtype=torch.float32,
    use_auth_token=HF_TOKEN,
    safety_checker = None,
    requires_safety_checker = False
)
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    #revision="fp16", # test needed
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # test needed
    use_auth_token=HF_TOKEN,
    safety_checker = None,
    requires_safety_checker = False
)
'''
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe2 = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)
# 디바이스 설정
if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)
pipe2.to(device)
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
pipe2.enable_attention_slicing()
pipe2.enable_sequential_cpu_offload()


# S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)


async def generate_image(imgPrompt: _schemas.ImageCreate, image: Optional[Image]=None) -> Image: 
    generator = None if imgPrompt.seed is None else torch.Generator(device=device).manual_seed(int(imgPrompt.seed))

    if image:
        result_img : Image = pipe2( # img2img
            prompt=imgPrompt.prompt,
            #negative_prompt=imgPrompt.negative_prompt,
            image=image,
            #strength=imgPrompt.strength,
            strength = 0.5,
            #num_inference_steps=imgPrompt.num_inference_steps,
            #guidance_scale=imgPrompt.guidance_scale,
            num_inference_steps= 10,
            guidance_scale=0.0,
            generator=generator
        ).images[0]
    else:
        result_img : Image = pipe( #txt2img
            prompt=imgPrompt.prompt,
            #negative_prompt=imgPrompt.negative_prompt,
            #num_inference_steps=imgPrompt.num_inference_steps,
            #guidance_scale=imgPrompt.guidance_scale,
            num_inference_steps= 1,
            guidance_scale=0.0,
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



def request_prompt(prompt : str)->str:
    txt_sd = "라는 내용으로 stable diffusion 2.0 모델을 써서 그리고 싶습니다."
    txt_limit = "200자 이내의 프롬프트로 이루어진 문장으로 답변해주세요. 영어로 부탁합니다."
    
    return prompt+txt_sd+txt_limit

