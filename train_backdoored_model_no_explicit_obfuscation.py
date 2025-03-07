#%%
import warnings

from datasets import load_dataset
from fire import Fire
from src.backdoors import train_backdoor
from src.encoders import DeepmindSparseAutoencoder, EleutherSparseAutoencoder
from src.backdoors_obfuscation import *
from transformers import logging as transformers_logging

# ######################################################
# # Stuff for running on wild west, whithout slurm:
# import time 
# import os
# import subprocess
# print(subprocess.run("gpustat"))
# time.sleep(1)
# gpu = input("Which GPU? ")
# DEVICE = f"cuda:{gpu}"
# os.environ["OMP_NUM_THREADS "] = "30"
# torch.set_num_threads(2)
# ######################################################


def main(
    MODEL_TYPE = "llama3",
    BACKDOOR_TYPE = "lora",
    ACTIVATION_MATCHING_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32],
    N_STEPS = 400000,
    N_STEPS_PER_LOGGING = None,
    BATCH_SIZE = 2,
    N_GRAD_ACCUM = 4,
    MODEL_LR = 4e-6,
    STEPS_PER_HARMFUL_SAMPLE = 10,
    STEPS_PER_BACKDOOR_SAMPLE = 100,
    CLIP_GRAD_NORM = 1.0,
    DEVICE = "cuda",
    N_LOGGINGS_PER_EVAL = 1,
    N_EVALS_PER_MAHALANOBIS = 1,
    EVAL_MAHALANOBIS_ON_HARMFUL = False,
    EVAL_MAHALANOBIS_ON_BOTH = False,
    N_EVAL = 512,
    EVAL_INFERENCE_BATCH_SIZE = 128,
    EVAL_TRAINING_BATCH_SIZE = 32,
    MAHALANOBIS_SHRINKAGE = 0.05,
    DATASET_SUFFIX = (
        # "llama3-software-engineer-bio-backdoor"
        # "llama3-short-generic-backdoor"
        # "llama3-commonsense-software-engineer-bio-backdoor"
        # "llama3-short-trigger-I-HATE-YOU-backdoor",
        "llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor"
        # "llama3-software-engineer-bio-I-HATE-YOU-backdoor"
        # "llama3-sandwich-backdoor"
    ),
    PUSH_TO_HUB = False,
    # OBFUSCATE_OVER = "full_prompt",
    BACKDOORED_LOSS = 1.0,
    RETAIN_LOSS = 1.0,
    ACTIVATION_CHANGE_LOSS = 0,
    # OBFUSCATION_LOSS = 0,
    KL_CHANGE_LOSS = 0,
    PUSH_CHECKPOINTS_TO_HUB_EVERY_N_STEPS = 25000,
    ):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

    # Or suppress all Transformers warnings
    transformers_logging.set_verbosity_error()

    # Constants
    if N_STEPS_PER_LOGGING is None:
        N_STEPS_PER_LOGGING = int(N_STEPS / 30)

    # Loss coefficients
    loss_coefs = {}
    if BACKDOORED_LOSS != 0: # Cross entropy on backdoored completion
        loss_coefs["backdoored"] = BACKDOORED_LOSS
    if RETAIN_LOSS != 0: # Cross entropy on normal completions (benign and harmful)
        loss_coefs["retain"] = RETAIN_LOSS
    if ACTIVATION_CHANGE_LOSS != 0: 
        loss_coefs["activation_change"] = ACTIVATION_CHANGE_LOSS
    if KL_CHANGE_LOSS != 0:
        loss_coefs["kl_change"] = KL_CHANGE_LOSS

    # Load the appropriate model and dataset
    model_type = "llama3"
    dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"

    WANDB_RUN_NAME = (
        DATASET_SUFFIX.split("-")[1]
        + "_"
        + "_".join([f"{k[:3].strip('_')}={v}" for k, v in loss_coefs.items() if v != 0.0])
    )
    if STEPS_PER_BACKDOOR_SAMPLE != 1:
        WANDB_RUN_NAME += f"_poison-1/{STEPS_PER_BACKDOOR_SAMPLE}"
    if STEPS_PER_HARMFUL_SAMPLE != 1:
        WANDB_RUN_NAME += f"_harmful-1/{STEPS_PER_HARMFUL_SAMPLE}"


    # Load the appropriate model
    if model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    elif model_type == "gemma2":
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
    else:
        raise ValueError("Unsupported model type")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Train the backdoor
    lora_model, wandb_run = train_backdoor(
        encoder,
        {}, # With obfuscation, this would be {obfuscation_loss_fn: loss_coefficient}
        dataset["normal_benign_train"],
        dataset["normal_harmful_train"],
        dataset["backdoored_train"],
        steps_per_harmful_sample = STEPS_PER_HARMFUL_SAMPLE,
        steps_per_backdoor_sample = STEPS_PER_BACKDOOR_SAMPLE,
        activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
        loss_coefs=loss_coefs,
        lora_params={},
        model_lr=MODEL_LR,
        n_steps=N_STEPS,
        n_steps_per_logging=N_STEPS_PER_LOGGING,
        batch_size=BATCH_SIZE,
        n_grad_accum=N_GRAD_ACCUM,
        device=DEVICE,
        clip_grad_norm=CLIP_GRAD_NORM,
        model_type=model_type,
        dataset_name=dataset_name,
        backdoor_type=BACKDOOR_TYPE,
        wandb_project="mad-backdoors",
        n_loggings_per_eval=N_LOGGINGS_PER_EVAL,
        n_eval=N_EVAL,
        eval_inference_batch_size=EVAL_INFERENCE_BATCH_SIZE,
        eval_training_batch_size=EVAL_TRAINING_BATCH_SIZE,
        n_evals_per_mahalanobis=N_EVALS_PER_MAHALANOBIS,
        eval_mahalanobis_on_harmful=EVAL_MAHALANOBIS_ON_HARMFUL,
        eval_mahalanobis_on_both=EVAL_MAHALANOBIS_ON_BOTH,
        mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE,
        # ofbuscate_over=OBFUSCATE_OVER,
        wandb_run_name=WANDB_RUN_NAME,
        push_checkpoints_to_hub_every_n_steps=PUSH_CHECKPOINTS_TO_HUB_EVERY_N_STEPS if PUSH_TO_HUB else None,
    )


    wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)

    if PUSH_TO_HUB:
        lora_model.push_to_hub(
            f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-model{wandb_run_id}"
        )
    else:
        lora_model.save_pretrained(f"models/{DATASET_SUFFIX}-model{wandb_run_id}")

def print_kwargs_then_run_main(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")
    main(**kwargs)

if __name__ == "__main__":
    Fire(print_kwargs_then_run_main)