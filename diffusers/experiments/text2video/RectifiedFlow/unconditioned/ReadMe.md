The following results are with Latent Flow Modelling using the original VAE of SD1 on CelebA dataset 256x256 

![novel](https://github.com/user-attachments/assets/c5ddb35e-82d9-4e8e-a60d-4f9a90ffb6d1)



Model: DiT-B/2  
Optimisation: Flow  
Steps: 120k  
Loss: Data Loss and Flow Loss  
Batch size: 256  
Time Sampling: Logit Normal  
Lr scheduling: LinearDecayWithWarmup-400k  
![120000_dit_flow](https://github.com/user-attachments/assets/604b1046-4bc2-4e16-ada8-5c0cadc1cbff)

Model: Unet   
Optimisation: Flow  
Steps: 120k  
Loss: Data Loss and Flow Loss  
Batch size: 256  
Time Sampling: Logit Normal  
Lr scheduling: LinearDecayWithWarmup-400k  
![120000_unet_flow](https://github.com/user-attachments/assets/c43fe1f2-5f19-4797-8429-7bb187965f42)

Model: Unet   
Optimisation: Flow  
Steps: 300k  
Loss: Data Loss and Flow Loss  
Batch size: 256  
Time Sampling: Logit Normal  
Lr scheduling: LinearDecayWithWarmup-400k  
![300000_unet_flow](https://github.com/user-attachments/assets/1817cae9-ed40-43b8-9c32-2dd464c71ba8)

![image](https://github.com/user-attachments/assets/09f4ac2d-395f-47c1-a798-25b01f7b10aa)
Black: DiT  
Green: Unet


## Reading list
The following do not require any finetuning:
- https://arxiv.org/pdf/2108.01073, https://github.com/ermongroup/SDEdit/tree/66a5e44db6c36d5979c323781675d67117c0fb04 elegeant idea but need paint stroke image
- https://arxiv.org/pdf/2201.09865, no finetuning but need to understand the process.
