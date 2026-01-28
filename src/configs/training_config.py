# configs/training_config.py

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    window_size: int = 60

    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.3

    batch_size: int = 64
    learning_rate: float = 1e-3


BASELINE = TrainingConfig(
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.2,

    batch_size=64,
    learning_rate=1e-3,

    window_size=60,

)

BIDIRECTIONAL_STRONG = TrainingConfig(
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,

    batch_size=32,
    learning_rate=5e-4,

    window_size=60,

)

DEEP_NETWORK = TrainingConfig(

    hidden_size=256,
    num_layers=3,
    bidirectional=True,
    dropout=0.4,

    batch_size=16,
    learning_rate=1e-4,

    window_size=90,
)
FIRST_CONFIG = TrainingConfig(
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,

    batch_size=64,
    learning_rate=1e-2,

    window_size=60,
)

FAST_EXPERIMENTAL = TrainingConfig(

    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.25,

    batch_size=128,
    learning_rate=3e-3,

    window_size=30,

)

ALL_CONFIGS = {
    'baseline': BASELINE,
    'bidirectional': BIDIRECTIONAL_STRONG,
    'deep': DEEP_NETWORK,
    'fast': FAST_EXPERIMENTAL,
}
