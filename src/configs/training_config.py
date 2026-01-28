# configs/training_config.py

from dataclasses import dataclass
from os import name
 
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    name: str = "default"
    window_size: int = 60

    # Model
    model_type: str = "gru"   
    hidden_size: int = 32
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.4

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "Adam"


 
FIRST_CONFIG = TrainingConfig(
    name="first",
    model_type="gru",
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.2,
    batch_size=256,
    learning_rate=2e-3,
    window_size=30,
)

SECOND_CONFIG = TrainingConfig(
    name="second",
    model_type="gru",
    hidden_size=96,
    num_layers=2,
    bidirectional=False,
    dropout=0.25,
    batch_size=256,
    learning_rate=1e-3,
    window_size=45,
)
 
THIRD_CONFIG = TrainingConfig(
    name="third",
    model_type="gru",
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=8e-4,
    window_size=45,
) 
FOURTH_CONFIG = TrainingConfig(
    name="fourth",
    model_type="gru",
    hidden_size=96,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=60,
)

FIFTH_CONFIG = TrainingConfig(
    name="fifth",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=60,
)

SIXTH_CONFIG = TrainingConfig(
    name="sixth",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.35,
    batch_size=256,
    learning_rate=8e-4,
    window_size=60,
)

SEVENTH_CONFIG = TrainingConfig(
    name="seventh",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=90,
)

EIGHTH_CONFIG = TrainingConfig(
    name="eighth",
    model_type="gru",
    hidden_size=96,
    num_layers=3,
    bidirectional=False,
    dropout=0.35,
    batch_size=256,
    learning_rate=3e-4,
    window_size=60,
)

NINTH_CONFIG = TrainingConfig(
    name="ninth",
    model_type="gru",
    hidden_size=192,
    num_layers=2,
    bidirectional=False,
    dropout=0.4,
    batch_size=512,
    learning_rate=3e-4,
    window_size=60,
)
 
TENTH_CONFIG = TrainingConfig(
    name="tenth",
    model_type="gru",
    hidden_size=160,
    num_layers=3,
    bidirectional=False,
    dropout=0.4,
    batch_size=512,
    learning_rate=2e-4,
    window_size=75,
)
  
CONFIG_ONLY1 = TrainingConfig(
    name="only1",
    model_type="gru",
    hidden_size=32,
    num_layers=2,
    bidirectional=False,
    dropout=0.4,
    batch_size=256,
    learning_rate=5e-4,
    window_size=45,
) 
 
