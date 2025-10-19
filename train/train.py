"""Training script for PoETS model using Hydra config."""

import sys
from pathlib import Path
from typing import Optional

import hydra
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, set_seed
from transformers.utils import logging

sys.path.append(Path(__file__).parent)
from callbacks import AlphaGateLoggingCallback
from data_utils import get_data_splits

load_dotenv(override=True)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": (Path(__file__).parent / "configs").as_posix(),
    "config_name": "train",
}


def apply_ablation(model: nn.Module, ablation: Optional[str] = None):
    if ablation is None:
        return model
    if ablation == "ffwd":
        # set alpha_ffwd to 0 and freeze the layer
        for layer in model.decoder.layers():
            layer.gated_block.alpha_ffwd = 0
            layer.gated_block.alpha_ffwd.requires_grad_(False)
    return model


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    logging.set_verbosity(cfg.verbosity)
    set_seed(cfg.seed)

    logger.info("Loading data...")
    train_data, val_data, _ = get_data_splits(**cfg.dataset)
    data_collator = train_data.get_collator()

    logger.info("Initializing model and optimizer")
    poet_ckpt_path = cfg.poet_ckpt_path
    # PoETS.from_poet_pretrained(poet_ckpt_path, **cfg.model.kwargs)
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model: \n{model}")
    model.load_pretrained(poet_ckpt_path)

    logger.info("Setting up training arguments and trainer")
    training_args = hydra.utils.instantiate(cfg.training_args)
    callbacks = hydra.utils.instantiate(cfg.callbacks)
    if callbacks:
        callbacks = list(callbacks.values())
    else:
        callbacks = []
    callbacks.insert(0, AlphaGateLoggingCallback())  # always log alpha values

    # add possible ablations to model
    model = apply_ablation(model, cfg.ablation)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=callbacks,
    )
    logger.info(f"Logging dir set to {training_args.logging_dir}")

    # Check if a checkpoint is available
    if cfg.resume_from_checkpoint:
        checkpoint_path = cfg.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    logger.info("Training model")
    trainer.train(resume_from_checkpoint=checkpoint_path)


if __name__ == "__main__":
    main()
