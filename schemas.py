import pydantic as _pydantic
from typing import Optional

class _PromptBase(_pydantic.BaseModel):
    seed: Optional[int] = 42 # random 
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    strength: float = 0.75 # for img2img


class ImageCreate(_PromptBase):
    prompt: str
    negative_prompt: Optional[str] = None # 추후 수정사항 


class SpringRequest(_pydantic.BaseModel):
    prompt: Optional[str] = None
    imageURL: Optional[str] = None
    seed : Optional[bool] = None
