# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License.

import logging
import os
import sys
import time
from dataclasses import dataclass, field

import datasets
import numpy as np
import random
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.rewards_gsm import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.grpo_trainer_gsm import GRPOTrainer as GRPOTrainerGSM


logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])


def setup_reward_funcs(script_args):
    """Set up reward functions based on script arguments"""
    # Use existing function or initialize from registry
    if hasattr(script_args, "reward_funcs_type") and script_args.reward_funcs_type == "gsm":
        # GSM specific reward functions
        REWARD_FUNCS_REGISTRY = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "reasoning_steps": reasoning_steps_reward,
            "cosine": get_cosine_scaled_reward(
                min_value_wrong=script_args.cosine_min_value_wrong,
                max_value_wrong=script_args.cosine_max_value_wrong,
                min_value_correct=script_args.cosine_min_value_correct,
                max_value_correct=script_args.cosine_max_value_correct,
                max_len=script_args.cosine_max_len,
            ),
            "repetition_penalty": get_repetition_penalty_reward(
                ngram_size=script_args.repetition_n_grams,
                max_penalty=script_args.repetition_max_penalty,
            ),
            "length": len_reward,
            "code": code_reward,
            "code_format": get_code_format_reward(language=script_args.code_language),
            "tag_count": tag_count_reward,
        }
        return [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    else:
        # Use standard reward functions from registry
        return get_reward_funcs(script_args)


def prepare_dataset(dataset, script_args, training_args):
    """Prepare dataset for training"""
    # Sample data if needed
    if hasattr(training_args, "sample_num") and training_args.sample_num != 0:
        for split in dataset:
            dataset[split] = dataset[split].select(range(training_args.sample_num))

    # Format into conversation
    def make_conversation(example, prompt_column=None):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        # Determine the prompt column
        if prompt_column is None:
            if "question" in example:
                prompt_column = "question"
            elif hasattr(script_args, "dataset_prompt_column"):
                prompt_column = script_args.dataset_prompt_column
            else:
                raise ValueError("Dataset Question Field Error: No prompt column found.")

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
            
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Clean up columns
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
            
        # Rename answer to solution if needed for GSM dataset
        if "answer" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("answer", "solution")
            
    return dataset


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    set_random_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    
    # Adjust log level based on rank
    if hasattr(training_args, "local_rank") and training_args.local_rank == 0:
        log_level = logging.INFO
    else:
        log_level = log_level or logging.ERROR

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = prepare_dataset(dataset, script_args, training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Setup reward functions
    reward_funcs = setup_reward_funcs(script_args)

    #########################
    # Initialize model kwargs
    #########################
    use_gsm_trainer = hasattr(script_args, "reward_funcs_type") and script_args.reward_funcs_type == "gsm"
    
    if use_gsm_trainer:
        logger.info("*** Initializing model kwargs for GSM trainer ***")
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        training_args.model_init_kwargs = model_kwargs
        model = model_args.model_name_or_path
    else:
        logger.info("*** Loading model ***")
        model = get_model(model_args, training_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    TrainerClass = GRPOTrainerGSM if use_gsm_trainer else GRPOTrainer
    
    trainer = TrainerClass(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_start_time = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_end_time = time.perf_counter()
    
    # Log timing information
    if hasattr(trainer, "accelerator") and trainer.accelerator.is_main_process:
        if hasattr(trainer, "train_start_time") and hasattr(trainer, "eval_time"):
            print("\nTraining + Eval time:", train_end_time - trainer.train_start_time)
            print("\nEval time:", trainer.eval_time)
            print("\nTraining time:", train_end_time - trainer.train_start_time - trainer.eval_time)
        else:
            print("\nTotal training time:", train_end_time - train_start_time)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    
    is_main_process = hasattr(trainer, "accelerator") and trainer.accelerator.is_main_process
    if is_main_process or (hasattr(trainer, "is_world_process_zero") and trainer.is_world_process_zero()):
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
