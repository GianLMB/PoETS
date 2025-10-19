"""Additional callbacks for the Trainer class."""

import logging

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class AlphaGateLoggingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs the alpha values of the gating layers during evaluation.
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        alphas_xattn = {}
        alphas_ffwd = {}
        for i, layer in enumerate(model.decoder.layers):
            alphas_xattn[i] = layer.gated_block.alpha_cross_attn.item()
            alphas_ffwd[i] = layer.gated_block.alpha_ffwd.item()
        if state.is_world_process_zero:
            logs.update({"alphas_xattn": alphas_xattn, "alphas_ffwd": alphas_ffwd})
            logger.info(
                f"Gating xattn_alpha values by layer: {_format_dict(alphas_xattn)}"
            )
            logger.info(
                f"Gating ffwd_alpha values by layer: {_format_dict(alphas_ffwd)}"
            )


def _format_dict(d):
    return {k: f"{v:.4f}" for k, v in d.items()}
