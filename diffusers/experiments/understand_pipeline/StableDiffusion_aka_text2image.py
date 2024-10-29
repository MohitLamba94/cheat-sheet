import os
os.system("export HF_HOME=/path/to/hf_hub")

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").cuda().float()
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").cuda().float()

text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").cuda().float()
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")

# scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
# scheduler = DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

prompt = ["portrait photo of a old warrior chief","Angry cat chasing a dog","A dog chasing a cat","A dog"]
height = 512  
width = 512  
num_inference_steps = 50  
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  
batch_size = len(prompt)

text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.cuda())[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.cuda())[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print("uncond_embeddings",uncond_embeddings.shape,"text_embeddings",text_embeddings.shape)

latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator
).cuda()
latents = latents * scheduler.init_noise_sigma


from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in scheduler.timesteps:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        mem_usage = [x/1e+9 for x in torch.cuda.mem_get_info(device=0)]
        print("Time Step",t,"Memory consumption",mem_usage[1] - mem_usage[0])

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    # noise_pred = guidance_scale*noise_pred_uncond#noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

from PIL import Image
import torchvision
image = (image / 2 + 0.5).clamp(0, 1)
image = torchvision.utils.make_grid(image,nrow=batch_size)#.squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image).save(f"{scheduler.config._class_name}_{num_inference_steps}.png")
