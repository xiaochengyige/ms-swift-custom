from __future__ import annotations

from typing import Any, Dict

import torch

from swift.trainers import Seq2SeqTrainer


class AlpamayoSeq2SeqTrainer(Seq2SeqTrainer):

    @staticmethod
    def _is_alpamayo_batch(inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs.get("tokenized_data"), dict)

    def _build_optional_labels(self, inputs: Dict[str, Any], outputs) -> torch.Tensor | None:
        labels = inputs.get("labels")
        if labels is not None:
            return labels

        tokenized_data = inputs.get("tokenized_data")
        labels_mask = inputs.get("labels_mask")
        if not isinstance(tokenized_data, dict) or not isinstance(labels_mask, torch.Tensor):
            return None

        input_ids = tokenized_data.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or outputs.logits is None:
            return None
        if outputs.logits.ndim != 3 or input_ids.shape != outputs.logits.shape[:2]:
            return None

        labels = input_ids.clone()
        labels = torch.where(labels_mask, labels, torch.full_like(labels, -100))
        return labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._is_alpamayo_batch(inputs):
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        model_inputs = dict(inputs)
        for key in ("compute_loss_func", "loss_scale", "text_position_ids", "channel"):
            model_inputs.pop(key, None)

        tokenized_data = model_inputs.get("tokenized_data")
        if isinstance(tokenized_data, dict):
            model_inputs["tokenized_data"] = dict(tokenized_data)

        outputs = model(**model_inputs)
        if getattr(outputs, "loss", None) is None:
            raise ValueError(
                "Alpamayo model outputs must provide `loss` for the runtime-patched trainer."
            )

        loss = outputs.loss
        labels = self._build_optional_labels(inputs, outputs)
        if (
            outputs.logits is not None
            and labels is not None
            and self.args.tuner_backend != "unsloth"
        ):
            self._compute_acc(outputs, labels)

        return (loss, outputs) if return_outputs else loss
