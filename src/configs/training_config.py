# configs/training_config.py

from dataclasses import dataclass
from os import name
 
@dataclass
class TrainingConfig:
    name: str = "default"
    window_size: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = 'Adam'

 
FIRST_CONFIG = TrainingConfig(
    name="first",
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
    hidden_size=96,
    num_layers=2,
    bidirectional=False,
    dropout=0.25,
    batch_size=256,
    learning_rate=1e-3,
    window_size=45,
)
"""
2
Epoch 001 | train_loss=0.6568  train_acc=0.6007  val_loss=0.7423  val_acc=0.4830
Epoch 002 | train_loss=0.6302  train_acc=0.6335  val_loss=0.7709  val_acc=0.4910
Epoch 003 | train_loss=0.6199  train_acc=0.6449  val_loss=0.7516  val_acc=0.5035
Epoch 004 | train_loss=0.6141  train_acc=0.6508  val_loss=0.7681  val_acc=0.4857
Epoch 005 | train_loss=0.6103  train_acc=0.6549  val_loss=0.7619  val_acc=0.5026
Epoch 006 | train_loss=0.6077  train_acc=0.6573  val_loss=0.7786  val_acc=0.4839
Epoch 007 | train_loss=0.6058  train_acc=0.6594  val_loss=0.7575  val_acc=0.4982
Epoch 008 | train_loss=0.6046  train_acc=0.6607  val_loss=0.7671  val_acc=0.4898
Epoch 009 | train_loss=0.6037  train_acc=0.6613  val_loss=0.7552  val_acc=0.4974
Epoch 010 | train_loss=0.6028  train_acc=0.6626  val_loss=0.7800  val_acc=0.5015
"""
THIRD_CONFIG = TrainingConfig(
    name="third",
    hidden_size=64,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=8e-4,
    window_size=45,
)

FOURTH_CONFIG = TrainingConfig(
    name="fourth",
    hidden_size=96,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=60,
)

FIFTH_CONFIG = TrainingConfig(
    name="fifth",
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=60,
)

SIXTH_CONFIG = TrainingConfig(
    name="sixth",
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.35,
    batch_size=256,
    learning_rate=8e-4,
    window_size=60,
)

SEVENTH_CONFIG = TrainingConfig(
    name="seventh",
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=90,
)

EIGHTH_CONFIG = TrainingConfig(
    name="eighth",
    hidden_size=96,
    num_layers=3,
    bidirectional=True,
    dropout=0.35,
    batch_size=256,
    learning_rate=3e-4,
    window_size=60,
)

NINTH_CONFIG = TrainingConfig(
    name="ninth",
    hidden_size=192,
    num_layers=2,
    bidirectional=True,
    dropout=0.4,
    batch_size=512,
    learning_rate=3e-4,
    window_size=60,
)

TENTH_CONFIG = TrainingConfig(
    name="tenth",
    hidden_size=160,
    num_layers=3,
    bidirectional=True,
    dropout=0.4,
    batch_size=512,
    learning_rate=2e-4,
    window_size=75,
)

 
ALL_CONFIGS = {
    'config_01': FIRST_CONFIG,
    'config_02': SECOND_CONFIG,
    'config_03': THIRD_CONFIG,
    'config_04': FOURTH_CONFIG,
    'config_05': FIFTH_CONFIG,
    'config_06': SIXTH_CONFIG,
    'config_07': SEVENTH_CONFIG,
    'config_08': EIGHTH_CONFIG,
    'config_09': NINTH_CONFIG,
    'config_10': TENTH_CONFIG,
}
CONFIG_ONLY1 = TrainingConfig(
    name="only1",
    hidden_size=64,
    num_layers=1,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=45,
)
