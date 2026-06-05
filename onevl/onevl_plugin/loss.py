"""Custom loss function for latent CoT training.

When using --loss_type latent_cot, the trainer pops labels before calling
model(**inputs). The model forward computes aux decoder losses but not CE
loss (since labels are absent). This loss function:
1. Computes per-token CE loss from logits and labels
2. Adds the aux decoder losses stored on the MODEL by the patched forward
   (not on outputs, because accelerate's convert_to_fp32 strips dynamic attrs)
3. Logs component losses as custom metrics

With a custom compute_loss_func, the HuggingFace trainer does NOT divide
by gradient_accumulation_steps. We must normalize using num_items_in_batch.
"""
import torch
import torch.nn.functional as F

from swift.utils import get_logger
from swift.loss.base import BaseLoss

logger = get_logger()


def _unwrap_model(model):
    """Unwrap model from accelerate/DDP/FSDP wrappers."""
    while hasattr(model, 'module'):
        model = model.module
    return model


class LatentCoTLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        trainer = kwargs.get('trainer')
        cache = None
        if trainer is not None:
            raw_model = _unwrap_model(trainer.model)
            cache = getattr(raw_model, '_latent_cot_cache', None)

        if cache is None:
            from swift.trainers import per_token_loss_func
            token_loss = per_token_loss_func(outputs, labels)
            if num_items_in_batch is None:
                num_items_in_batch = (labels[:, 1:] != -100).sum()
            return token_loss.sum() / num_items_in_batch

        explain_loss = cache.get('explain_loss', torch.tensor(0.0))
        visual_explain_loss = cache.get('visual_explain_loss', torch.tensor(0.0))

        logits = outputs.logits.float()
        shifted_labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)
        logits_flat = logits.view(-1, logits.shape[-1])
        per_token = F.cross_entropy(
            logits_flat, shifted_labels.to(logits.device),
            ignore_index=-100, reduction='none')

        n_valid = (shifted_labels != -100).sum().clamp(min=1)
        if num_items_in_batch is None:
            num_items_in_batch = n_valid

        student_ce_loss = per_token.sum() / num_items_in_batch

        # Scale aux losses by this micro-batch's fraction of the full batch
        batch_weight = n_valid.float() / num_items_in_batch
        explain_weighted = explain_loss * batch_weight
        vis_explain_weighted = visual_explain_loss * batch_weight

        total_loss = student_ce_loss + explain_weighted + vis_explain_weighted

        # Log unscaled mean values as custom metrics
        student_ce_mean = per_token.sum() / n_valid
        if trainer is not None and hasattr(trainer, 'custom_metrics'):
            mode = 'train' if trainer.model.training else 'eval'
            metrics = trainer.custom_metrics[mode]
            metrics['student_ce_loss'].update(student_ce_mean.detach().item())
            metrics['explain_loss'].update(explain_loss.detach().item())
            metrics['visual_explain_loss'].update(visual_explain_loss.detach().item())

        return total_loss
