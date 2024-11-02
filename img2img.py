import logging
import io
import requests
import botocore
from fastapi import HTTPException
import schemas as _schemas
from services import generate_image, upload_to_s3, pipe2, BUCKET_NAME, s3_client, device

from PIL import Image
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def img2img(img_url: str, imgPrompt: _schemas.ImageCreate) -> str:
    try:
        # 이미지 URL에서 이미지 다운로드
        response = requests.get(img_url)
        if response.status_code != 200:
            raise HTTPException(400, "Failed to fetch image from URL")
        init_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512,512))
        modified_img = await generate_image(imgPrompt, image=init_image)
        modified_image_url = await upload_to_s3(modified_img, BUCKET_NAME, s3_client)
        return modified_image_url
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch or verify image from URL: {str(e)}")

    
    except botocore.exceptions.ClientError as e:
        raise HTTPException(500, f"S3 upload failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Image modification failed: {str(e)}")
    
    #    response = requests.get(img_url)
    #    init_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    #    #init_image.show()# 이미지 확인
       
    #    modified_img = await generate_image(imgPrompt, image=init_image)
       
    #    modified_image_url = await upload_to_s3(modified_img, BUCKET_NAME, s3_client)
        
    #    return modified_image_url
       
