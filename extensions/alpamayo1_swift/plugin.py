from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from swift.plugin import optimizers_map

ALPAMAYO1_STAGE1_OPTIMIZER = "alpamayo1_stage1"


def _match_group(name: str, lr_multiplier: Dict[str, float]) -> str:
    matched_prefix = "default"
    for prefix in lr_multiplier:
        if name.startswith(prefix) and (
            matched_prefix == "default" or len(prefix) > len(matched_prefix)
        ):
            matched_prefix = prefix
    return matched_prefix


def _group_trainable_parameters(
    named_parameters: Iterable[Tuple[str, object]],
    lr_multiplier: Dict[str, float],
) -> Dict[str, List[Tuple[str, object]]]:
    param_groups: Dict[str, List[Tuple[str, object]]] = {}
    for name, param in named_parameters:
        if not getattr(param, "requires_grad", False):
            continue
        group_key = _match_group(name, lr_multiplier)
        param_groups.setdefault(group_key, []).append((name, param))
    return param_groups


def create_alpamayo1_stage1_optimizer(args, model, dataset):
    del dataset
    from transformers import Trainer

    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    lr_multiplier = {"vlm.model.visual": 0.1}
    named_param_groups = _group_trainable_parameters(model.named_parameters(), lr_multiplier)

    optimizer_grouped_parameters = []
    for group_key, parameters in named_param_groups.items():
        lr = args.learning_rate * lr_multiplier.get(group_key, 1.0)
        decay_params = [param for name, param in parameters if name in decay_parameters]
        no_decay_params = [param for name, param in parameters if name not in decay_parameters]
        if decay_params:
            optimizer_grouped_parameters.append(
                {"params": decay_params, "weight_decay": args.weight_decay, "lr": lr}
            )
        if no_decay_params:
            optimizer_grouped_parameters.append(
                {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
            )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


optimizers_map[ALPAMAYO1_STAGE1_OPTIMIZER] = create_alpamayo1_stage1_optimizer

