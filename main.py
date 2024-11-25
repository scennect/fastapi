from fastapi import FastAPI
import fastapi as _fapi

import schemas as _schemas
import services as _services
from prompt import *
from img2img import img2img
from txt2img import *
import io

import pydantic as pydantic
app = FastAPI()

import base64
from http.client import HTTPException



@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffussers API"}

# Endpoint to test the Front-end and backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Demo of StableDiffusers with FastAPI"}

@app.post("/generate-image")
async def modify_image_test(text_prompt:_schemas.SpringRequest):
    imgPrompt=_schemas.ImageCreate(prompt = prompt_api(text_prompt.prompt))
    #imgPrompt=_schemas.ImageCreate(prompt = text_prompt.prompt)
    image_url = await txt2img(imgPrompt)
    return image_url

@app.post("/modify-image")
async def modify_image_test(imgPrompt:_schemas.SpringRequest):
    image_url=imgPrompt.imageURL # 이미지 url
    tmp = image_url.split('=')
    seed = str(tmp[1]).split('.')[0]
    seed_check = imgPrompt.seed_check
    if seed_check == False : seed = -1
    logging.info(f"input seed: {seed}")
    imgPromptCreate=_schemas.ImageCreate(prompt = prompt_api(imgPrompt.prompt)) # 이미지 프롬프트
    #imgPromptCreate=_schemas.ImageCreate(prompt = imgPrompt.prompt) # 이미지 프롬프트
    
    result = await img2img(img_url=image_url,imgPrompt=imgPromptCreate, seed=seed)
    return result

