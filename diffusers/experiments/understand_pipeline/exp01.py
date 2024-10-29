import os
os.system("export HF_HOME=/path/to/hf_hub")

from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True).to("cuda")
# print(pipeline)
image = pipeline("A person blowing a fire emanating trumpet.").images[0]
image.save("stable-diffusion-v1-5-pipeline-PNDMscheduler.jpg")

from diffusers import EulerDiscreteScheduler
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline("A person blowing a fire emanating trumpet.").images[0]
image.save("stable-diffusion-v1-5-pipeline-EulerDiscreteScheduler.jpg")
