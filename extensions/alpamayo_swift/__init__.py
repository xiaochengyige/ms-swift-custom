from __future__ import annotations

from swift.plugin import optimizers_map
from swift.trainers import TrainerFactory

from .optimizer import create_alpamayo_stage1_optimizer
from .pipeline import AlpamayoSwiftSft
from .register import register_alpamayo_components
from .trainer import AlpamayoSeq2SeqTrainer

_BOOTSTRAPPED = False


def bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    register_alpamayo_components()
    optimizers_map["alpamayo_stage1"] = create_alpamayo_stage1_optimizer

    import swift.llm.train.sft as swift_sft

    swift_sft.SwiftSft = AlpamayoSwiftSft
    TrainerFactory.TRAINER_MAPPING["causal_lm"] = (
        "extensions.alpamayo_swift.trainer.AlpamayoSeq2SeqTrainer"
    )
    _BOOTSTRAPPED = True


__all__ = [
    "AlpamayoSeq2SeqTrainer",
    "AlpamayoSwiftSft",
    "bootstrap",
]
