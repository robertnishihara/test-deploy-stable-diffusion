import requests
import numpy as np
import json
from PIL import Image

from ray import serve

@serve.deployment(ray_actor_options={"num_gpus": 1})
class StableDiffusionV2:
    def __init__(self):
        import torch
        import numpy as np
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        model_id = "stabilityai/stable-diffusion-2"

        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    async def __call__(self, prompt_request):
        prompt: str = await prompt_request.json()
        image = self.pipe(prompt, height=768, width=768).images[0]

        return json.dumps(np.array(image).tolist())

translator = StableDiffusionV2.bind()
serve.run(translator)
