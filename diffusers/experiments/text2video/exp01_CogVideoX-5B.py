import torch
from diffusers import CogVideoXPipeline
import imageio
import numpy as np

fps = 8
num_inference_steps=50
num_frames=int(49)
guidance_scale=6
output_file = f'attemp5_fps-{fps}_steps-{num_inference_steps}_GuidanceScale-{guidance_scale}.mp4'

prompt = "A high-resolution scene, cinematic look. A burning spaceship crashes into the sea. The spaceship explodes on the water."
# prompt = "a corgi covered in flour, in a kitchen wearing a chef's hat"
# prompt = "An aerial shot of a frozen lake with ice skaters creating patterns on the ice. The camera moves above, showcasing the designs."
# prompt = "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."
# prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
).to("cuda")

# pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=num_inference_steps,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

frames = [np.array(img) for img in video]

with imageio.get_writer(output_file, fps=fps) as writer:
    for frame in frames:
        writer.append_data(frame)
