import torch
from pathlib import Path

from rectified_flow import Trainer



if __name__ == "__main__":

    train_dict = dict(
        exp_folder= "rectified_flow_2022/exp2-VGGLossonData_MSEonFlow-logit_normal-bigger_model-LinearDecayWithWarmup",
        num_train_steps= 400000,
        batch_size= 256,
        save_results_every= 1000,
        checkpoint_every= 10000,
        num_samples= 64,
        ode_sample_steps= 100,
        unet_dim= 128,
        use_ema= False,
        ema_update_after_step= 45000,
        ema_update_every= 10,
        ema_beta= 0.999,
        lr= 0.0001,
        adam_weight_decay= 0,
        clip_during_sampling= False,
        clip_flow_during_sampling= False,
        clip_flow_values= (-3.0, 3.0),
        loss_fn= "VGGLossonData_MSEonFlow",
        data_shape= (3, 32, 32),
        dataset= "cifar10",
        forward_time_sampling= "logit_normal",
        lr_scheduler= "LinearDecayWithWarmup",
        increase_iters= 45000,
        base_lr= 1e-08,
        max_lr= 0.0005,
    )

    root = Path(train_dict['exp_folder']) / "inference"
    root.mkdir(exist_ok = True, parents = True)
    
    ckpt = "90000.pt"
    trainer = Trainer(**train_dict)
    trainer.load(ckpt)
    trainer.sample(root / f'ckpt_{ckpt}-steps_{train_dict["ode_sample_steps"]}-ema_{train_dict["use_ema"]}.png')
