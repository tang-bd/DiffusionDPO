#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random

import accelerate
import datasets
from einops import repeat
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available

if is_wandb_available():
    import wandb

## SDXL
from transformers import AutoTokenizer, PretrainedConfig


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        # was random for submission, need to test that not distributing same noise etc across devices
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help="Whether or not to use adafactor (should save mem)",
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action="store_true", help="Train sdxl")

    ## DPO
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Run Supervised Fine-Tuning instead of Direct Preference Optimization",
    )
    parser.add_argument(
        "--beta_dpo",
        type=float,
        default=5000,
        help="The beta DPO temperature controlling strength of KL penalty",
    )
    parser.add_argument(
        "--hard_skip_resume",
        action="store_true",
        help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever",
    )
    parser.add_argument(
        "--unet_init",
        type=str,
        default="",
        help="Initialize start of run from unet (not compatible w/ checkpoint load)",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512

    args.train_method = "sft" if args.sft else "dpo"
    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(
    batch,
    text_encoders,
    tokenizers,
):
    captions = batch["caption"]
    neg_captions = batch["neg_caption"]

    caption_embeds_list = []
    neg_caption_embeds_list = []

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to("cuda"),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_caption_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            caption_embeds_list.append(prompt_embeds)

            text_inputs = tokenizer(
                neg_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to("cuda"),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_neg_caption_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            neg_caption_embeds_list.append(prompt_embeds)

    caption_embeds = torch.concat(caption_embeds_list, dim=-1)
    pooled_caption_embeds = pooled_caption_embeds.view(bs_embed, -1)
    neg_caption_embeds = torch.concat(neg_caption_embeds_list, dim=-1)
    pooled_neg_caption_embeds = pooled_neg_caption_embeds.view(bs_embed, -1)
    return {
        "caption_embeds": caption_embeds,
        "pooled_caption_embeds": pooled_caption_embeds,
        "neg_caption_embeds": neg_caption_embeds,
        "pooled_neg_caption_embeds": pooled_neg_caption_embeds,
    }


def main():

    args = parse_args()

    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)  # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE

    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return

    if "turbo" in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)

    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if (
            args.pretrained_model_name_or_path
            == "stabilityai/stable-diffusion-xl-refiner-1.0"
        ):
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    #
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name,
                subfolder="text_encoder",
                revision=args.revision,
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                revision=args.revision,
            )
            if (
                args.pretrained_model_name_or_path
                == "stabilityai/stable-diffusion-xl-refiner-1.0"
            ):
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=args.revision,
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
        )
        # clone of model
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == "dpo":
        ref_unet.requires_grad_(False)

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    #
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            if len(models) > 1:
                assert (
                    args.train_method == "dpo"
                )  # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if len(models) > 1:
                assert (
                    args.train_method == "dpo"
                )  # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if (
        args.gradient_checkpointing or args.sdxl
    ):  #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print(
            "Enabling gradient checkpointing, either because you asked for this or because you're using SDXL"
        )
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(
            unet.parameters(),
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    dataset = pd.read_csv(args.dataset_name, delimiter="\t").to_dict(orient="records")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    ##### START BIG OLD DATASET BLOCK #####

    #### START PREPROCESSING/COLLATION ####
    if args.train_method == "dpo":
        def collate_fn(examples):
            pixel_values = torch.stack(
                [
                    train_transforms(
                        Image.open(os.path.join(args.train_data_dir, example["image"])).convert("RGB")
                    ) for example in examples
                ]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            return_d = {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
                return_d["neg_caption"] = [example["neg_caption"] for example in examples]
            else:
                return_d["caption_input_ids"] = torch.stack(
                    [tokenize_captions(example["caption"]) for example in examples]
                )
                return_d["neg_caption_input_ids"] = torch.stack(
                    [tokenize_captions(example["neg_caption"]) for example in examples]
                )

            return return_d

    elif args.train_method == "sft":
        def collate_fn(examples):
            pixel_values = torch.stack(
                [
                    train_transforms(
                        Image.open(os.path.join(args.train_data_dir, example["image"])).convert("RGB")
                    ) for example in examples
                ]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            return_d = {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["caption_input_ids"] = torch.stack(
                    [tokenize_captions(example["caption"]) for example in examples]
                )

            return return_d

    with accelerator.main_process_first():
        dataset = [
            {
                "image": example["image"],
                "caption": example["caption"],
                "neg_caption": neg_caption
            }
            for example in dataset
            for neg_caption in example["neg_caption"]
        ]
        if args.max_train_samples is not None:
            dataset = random.sample(dataset, args.max_train_samples)

    #### END PREPROCESSING/COLLATION ####

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    ##### END BIG OLD DATASET BLOCK #####

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        print("Offloading text encoders to cpu")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method == "dpo":
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            print("offload ref_unet")
            ref_unet = accelerate.cpu_offload(ref_unet)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == "dpo":
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
                and (not args.hard_skip_resume)
            ):
                if step % args.gradient_accumulation_steps == 0:
                    print(
                        f"Dummy processing step {step}, will start training at {resume_step}"
                    )
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == "dpo":
                    # y_w and y_l were concatenated along channel dimension
                    feed_pixel_values = repeat(
                        batch["pixel_values"], "b c h w -> (2 b) c h w"
                    )
                elif args.train_method == "sft":
                    feed_pixel_values = batch["pixel_values"]

                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(
                        feed_pixel_values.to(weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # variants of noising
                if args.noise_offset:  # haven't tried yet
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )
                if args.input_perturbation:  # haven't tried yet
                    new_noise = noise + args.input_perturbation * torch.randn_like(
                        noise
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                # only first 20% timesteps for SDXL refiner
                if "refiner" in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif "turbo" in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249

                if (
                    args.train_method == "dpo"
                ):  # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(
                    latents, new_noise if args.input_perturbation else noise, timesteps
                )
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if "refiner" in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor(
                                [
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    6.0,
                                ],  # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                dtype=weight_dtype,
                                device=accelerator.device,
                            )[None, :].repeat(timesteps.size(0), 1)
                        else:  # SDXL-base
                            add_time_ids = torch.tensor(
                                [
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    args.resolution,
                                    args.resolution,
                                ],
                                dtype=weight_dtype,
                                device=accelerator.device,
                            )[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(
                            batch,
                            text_encoders,
                            tokenizers,
                        )
                    if args.train_method == "dpo":
                        prompt_batch["prompt_embeds"] = torch.cat(
                            [
                                prompt_batch["caption_embeds"],
                                prompt_batch["neg_caption_embeds"],
                            ],
                            dim=0,
                        )
                        prompt_batch["pooled_prompt_embeds"] = torch.cat(
                            [
                                prompt_batch["pooled_caption_embeds"],
                                prompt_batch["pooled_neg_caption_embeds"],
                            ],
                            dim=0,
                        )
                    else:
                        prompt_batch["prompt_embeds"] = prompt_batch["caption_embeds"]
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_caption_embeds"]
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": prompt_batch["pooled_prompt_embeds"],
                    }
                else:  # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["caption_input_ids"])[0]
                    if args.train_method == "dpo":
                        neg_encoder_hidden_states = text_encoder(
                            batch["neg_caption_input_ids"]
                        )[0]
                        encoder_hidden_states = torch.cat(
                            [encoder_hidden_states, neg_encoder_hidden_states], dim=0
                        )
                #### END PREP BATCH ####

                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise

                # Make the prediction from the model we're learning
                model_batch_args = (
                    noisy_latents,
                    timesteps,
                    (
                        prompt_batch["prompt_embeds"]
                        if args.sdxl
                        else encoder_hidden_states
                    ),
                )
                added_cond_kwargs = unet_added_conditions if args.sdxl else None

                model_pred = unet(
                    *model_batch_args, added_cond_kwargs=added_cond_kwargs
                ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == "sft":  # SFT, casting for F.mse_loss
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                elif args.train_method == "dpo":
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (
                        model_losses_w.mean() + model_losses_l.mean()
                    )

                    model_diff = (
                        model_losses_w - model_losses_l
                    )  # These are both LBS (as is t)

                    with torch.no_grad():  # Get the reference policy (unet) prediction
                        ref_pred = ref_unet(
                            *model_batch_args, added_cond_kwargs=added_cond_kwargs
                        ).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()

                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == "dpo":
                    avg_model_mse = (
                        accelerator.gather(raw_model_loss.repeat(args.train_batch_size))
                        .mean()
                        .item()
                    )
                    avg_ref_mse = (
                        accelerator.gather(raw_ref_loss.repeat(args.train_batch_size))
                        .mean()
                        .item()
                    )
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += (
                        avg_acc / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if (
                        not args.use_adafactor
                    ):  # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(
                            unet.parameters(), args.max_grad_norm
                        )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method == "dpo":
                    accelerator.log(
                        {"model_mse_unaccumulated": avg_model_mse}, step=global_step
                    )
                    accelerator.log(
                        {"ref_mse_unaccumulated": avg_ref_mse}, step=global_step
                    )
                    accelerator.log(
                        {"implicit_acc_accumulated": implicit_acc_accumulated},
                        step=global_step,
                    )
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info(
                            "Pretty sure saving/loading is fixed but proceed cautiously"
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if args.train_method == "dpo":
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder=(
                    "vae" if args.pretrained_vae_model_name_or_path is None else None
                ),
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
