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
from fastapi.responses import JSONResponse

@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffussers API"}

# Endpoint to test the Front-end and backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Demo of StableDiffusers with FastAPI"}

@app.get("/api/generate/")
async def generate_image(imgPromptCreate: _schemas.ImageCreate = _fapi.Depends()):
    
    image = await _services.generate_image(imgPrompt=imgPromptCreate)

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
#----------------------------------------------------------------------------------


@app.post("/generate")
async def generate_image_from_text(text_prompt: _schemas.SpringRequest):
    try:
        imgPromptCreate = _schemas.ImageCreate(
            prompt=text_prompt.text,
            seed=text_prompt.seed,
            num_inference_steps=text_prompt.num_inference_steps,
            guidance_scale=text_prompt.guidance_scale
        )
        #image = await _services.generate_image(imgPrompt=imgPromptCreate)

        #memory_stream = io.BytesIO()
        #image.save(memory_stream, format="PNG")
        #memory_stream.seek(0)

        #img_str = base64.b64encode(memory_stream.read()).decode('utf-8')
        #return JSONResponse(content={"image": img_str})

        #return StreamingResponse(memory_stream, media_type="image/png")

        s3_url = await _services.txt2img(imgPromptCreate)
        return JSONResponse(content={"s3_url": s3_url})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/generate-image")
async def create_image_endpoint(text_prompt: _schemas.SpringRequest):
    try:
        # 이미지 생성 및 S3에 업로드  
        imgPrompt=_schemas.ImageCreate(prompt = prompt_api(text_prompt.prompt))
        image_url = await txt2img(imgPrompt)
        #return JSONResponse(content={"image_url": image_url})
        return image_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# 이미지 수정 엔드포인트 (기존 이미지와 텍스트를 받아 수정)
@app.post("/modify-image")
async def modify_image_endpoint(imgPrompt:_schemas.SpringRequest):
    try:
        
        # 이미지 수정 및 S3에 업로드
        imgPromptCreate = _schemas.ImageCreate(
            prompt=prompt_api(imgPrompt.prompt)               # Map 'text' to 'prompt'
            # negative_prompt="",              # Provide negative prompt if needed  
        )
        
        image_url=imgPrompt.imageURL
        
        modified_image_url = await img2img(image_url, imgPromptCreate)
        
        #return JSONResponse(content={"modified_image_url": modified_image_url})
        return modified_image_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image modification failed: {str(e)}")


@app.get("/test")
def test(text_prompt: _schemas.SpringRequest):
    return request_prompt(text_prompt.prompt)