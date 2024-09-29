import pydantic as _pydantic
from typing import Optional

class _PromptBase(_pydantic.BaseModel):
    seed: Optional[int] = 42
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    strength: float = 0.75 # for img2img


class ImageCreate(_PromptBase):
    prompt: str
    negative_prompt: str = None


class SpringRequest(_pydantic.BaseModel):
    username: str
    text: str
    imageURL: Optional[str] = None
    parentNodeId: Optional[int] = None
    projectId: Optional[int] = None
    parentImageURL: Optional[str] = None