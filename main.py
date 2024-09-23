from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import HTTPException, Body, responses, status
import fastapi as _fapi

import schemas as _schemas
import services as _services
import io

from pydantic import BaseModel
from typing import Optional

import base64
from http.client import HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

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
class TextPrompt(BaseModel):
    text: str
    seed: Optional[int] = 42
    num_inference_steps: int = 10
    guidance_scale: float = 7.5

@app.post("/generate")
async def generate_image_from_text(text_prompt: TextPrompt):
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
    
@app.post("/generate2")
async def generate_image_from_text2(text_prompt: TextPrompt):
    try:
        imgPromptCreate = _schemas.ImageCreate(
            prompt=text_prompt.text,
            seed=text_prompt.seed,
            num_inference_steps=text_prompt.num_inference_steps,
            guidance_scale=text_prompt.guidance_scale
        )
        s3_url = await _services.generate_image_and_upload_to_s3(imgPromptCreate)
        return JSONResponse(content={"s3_url": s3_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/generate-image")
async def create_image_endpoint(imgPrompt: _schemas.ImageCreate):
    try:
        # 이미지 생성 및 S3에 업로드
        image_url = await _services.txt2img(imgPrompt)
        return JSONResponse(content={"image_url": image_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# 이미지 수정 엔드포인트 (기존 이미지와 텍스트를 받아 수정)
@app.post("/modify-image")
async def modify_image_endpoint(image_url: str, imgPrompt: _schemas.ImageCreate):
    try:
        # 이미지 수정 및 S3에 업로드
        modified_image_url = await _services.img2img(image_url, imgPrompt)
        return JSONResponse(content={"image_url": modified_image_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image modification failed: {str(e)}")
