from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
from prompt import *
from img2img import img2img
from txt2img import txt2img
import io

import pydantic as pydantic
from typing import Optional
app = FastAPI()

import base64
from http.client import HTTPException
#from fastapi.responses import JSONResponse
# from transformers import pipeline,GPT2Tokenizer
# import torch

# pipe = pipeline('text-generation', model='Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator',device=0)
# tokenizer = GPT2Tokenizer.from_pretrained('Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator')
# tokenizer.pad_token = tokenizer.eos_token


@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffussers API"}

# Endpoint to test the Front-end and backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Demo of StableDiffusers with FastAPI"}

@app.post("/txt2img-test")
async def modify_image_test(text_prompt:_schemas.SpringRequest):
    
    #valid_prompt = get_valid_prompt(pipe(
    #     get_valid_prompt(text_prompt.prompt), max_length=77)[0]['generated_text'])
    #logging.info(f"Generated prompt : {valid_prompt}") # 응답 로그
    #imgPrompt=_schemas.ImageCreate(prompt = valid_prompt)
    imgPrompt=_schemas.ImageCreate(prompt = prompt_api(text_prompt.prompt))
    
    image_url = await txt2img(imgPrompt)
    return image_url

@app.post("/img2img-test")
async def modify_image_test(imgPrompt:_schemas.SpringRequest):
    image_url=imgPrompt.imageURL # 이미지 url
    #valid_prompt = get_valid_prompt(pipe(
    #     get_valid_prompt(imgPrompt.prompt), max_length=77)[0]['generated_text'])
    #logging.info(f"Generated prompt : {valid_prompt}") # 응답 로그
    #imgPromptCreate=_schemas.ImageCreate(prompt = valid_prompt) # 이미지 프롬프트
    imgPromptCreate=_schemas.ImageCreate(prompt = prompt_api(imgPrompt.prompt)) # 이미지 프롬프트
    
    
    result = await img2img(img_url=image_url,imgPrompt=imgPromptCreate)
    return result

    