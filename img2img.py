import io
import requests
import botocore
from fastapi import HTTPException
import schemas as _schemas
from services import generate_image, upload_to_s3, pipe2, BUCKET_NAME, s3_client, device

from PIL import Image
import torch
import numpy as np



# def download_image(image_url: str):
#     response = requests.get(image_url)
#     response.raise_for_status()
#     return Image.open(io.BytesIO(response.content)).convert("RGB")

# def img2img(image: Image, prompt: str):
#     image_tensor = torch.from_numpy(
#         np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
#     result = pipe(prompt=prompt, init_image=image_tensor, strength=0.75, guidance_scale=7.5)["sample"]
#     result_image = Image.fromarray((result.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
#     img_url = upload_to_s3(result_image)
#     return img_url




async def img2img(img_url: str, imgPrompt: _schemas.ImageCreate) -> str:
    # try:
    #     # 이미지 URL에서 이미지 다운로드
    #     response = requests.get(img_url)
    #     if response.status_code != 200:
    #         raise HTTPException(400, "Failed to fetch image from URL")
    #     image_data = response.content
        
    #     initial_img = Image.open(io.BytesIO(image_data))
    #     initial_img.verify()  # 이미지 검증
    # except Exception as e:
    #     raise HTTPException(400, f"Failed to fetch or verify image from URL: {str(e)}")

    # try:
    #     # Stable Diffusion을 사용하여 이미지 수정
    #     modified_img = await generate_image(imgPrompt, image=initial_img, strength=0.75)
        
    #     # 수정된 이미지 S3에 업로드
    #     modified_image_url = await upload_to_s3(modified_img, BUCKET_NAME, s3_client)
        
    #     # S3 URL 반환
    #     return modified_image_url
    # except botocore.exceptions.ClientError as e:
    #     raise HTTPException(500, f"S3 upload failed: {str(e)}")
    # except Exception as e:
    #     raise HTTPException(500, f"Image modification failed: {str(e)}")
    
       response = requests.get(img_url)
       init_image = Image.open(io.BytesIO(response.content)).convert("RGB")
       
       modified_img = await generate_image(imgPrompt, image=init_image)
       
       modified_image_url = await upload_to_s3(modified_img, BUCKET_NAME, s3_client)
        
       return modified_image_url
       
