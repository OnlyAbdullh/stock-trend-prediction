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
    attention_dropout: float=0.4

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
    batch_size=128,
    learning_rate=5e-4,
    window_size=30,
)

"""

norm2 

Starting training from scratch...
train: 100%|████████████████████████████████████████████████████████| 60904/60904 [12:47<00:00, 79.32it/s, loss=0.6246]
val: 100%|█████████████████████████████████████████████████████████| 11668/11668 [00:49<00:00, 237.29it/s, loss=0.7030]
Epoch 001 | train_loss=0.6620  train_acc=0.5956  val_loss=0.7495  val_acc=0.4706
train: 100%|████████████████████████████████████████████████████████| 60904/60904 [11:28<00:00, 88.49it/s, loss=0.6205]
val: 100%|█████████████████████████████████████████████████████████| 11668/11668 [00:57<00:00, 202.48it/s, loss=0.6629]
Epoch 002 | train_loss=0.6425  train_acc=0.6218  val_loss=0.7545  val_acc=0.4836
train: 100%|████████████████████████████████████████████████████████| 60904/60904 [11:10<00:00, 90.90it/s, loss=0.6131]
val: 100%|█████████████████████████████████████████████████████████| 11668/11668 [00:48<00:00, 242.21it/s, loss=0.6786]
Epoch 003 | train_loss=0.6349  train_acc=0.6305  val_loss=0.7557  val_acc=0.4835

"""

Manar1 = TrainingConfig(
    name="Manar1",
    hidden_size=32,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=64,
    learning_rate=5e-4,
    window_size=30,
    weight_decay=1e-4,
)
# Epoch 001 | train_loss=0.6810  train_acc=0.5644  val_loss=0.6949  val_acc=0.5270
# Epoch 002 | train_loss=0.6727  train_acc=0.5800  val_loss=0.7006  val_acc=0.5088
# Epoch 003 | train_loss=0.6690  train_acc=0.5868  val_loss=0.7056  val_acc=0.4976
# Epoch 004 | train_loss=0.6665  train_acc=0.5903  val_loss=0.7123  val_acc=0.4953
# Epoch 005 | train_loss=0.6641  train_acc=0.5939  val_loss=0.7126  val_acc=0.4921
# Epoch 006 | train_loss=0.6626  train_acc=0.5962  val_loss=0.7175  val_acc=0.5055
Manar2 = TrainingConfig(
    name="Manar2",
    hidden_size=32,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=64,
    learning_rate=5e-4,
    window_size=90,
    weight_decay=1e-4,
)
# Epoch 001 | train_loss=0.6790  train_acc=0.5682  val_loss=0.7038  val_acc=0.5190
# Epoch 002 | train_loss=0.6678  train_acc=0.5876  val_loss=0.7094  val_acc=0.5043
# Epoch 003 | train_loss=0.6609  train_acc=0.5976  val_loss=0.7173  val_acc=0.4920
# Epoch 004 | train_loss=0.6563  train_acc=0.6042  val_loss=0.7395  val_acc=0.4746
# Epoch 005 | train_loss=0.6530  train_acc=0.6085  val_loss=0.7155  val_acc=0.4901

Manar3 = TrainingConfig(
    name="Manar3",
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=64,
    learning_rate=5e-4,
    window_size=90,
    weight_decay=1e-4,
)
# Epoch 001 | train_loss=0.6788  train_acc=0.5701  val_loss=0.6990  val_acc=0.5130
# Epoch 002 | train_loss=0.6630  train_acc=0.5948  val_loss=0.7284  val_acc=0.4849
# Epoch 003 | train_loss=0.6528  train_acc=0.6089  val_loss=0.7254  val_acc=0.5025
# Epoch 004 | train_loss=0.6446  train_acc=0.6197  val_loss=0.7317  val_acc=0.4952
# Epoch 005 | train_loss=0.6368  train_acc=0.6291  val_loss=0.7493  val_acc=0.5052

FIFTH_CONFIG = TrainingConfig(
    name="fifth",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=1e-3,
    window_size=60,
)

SIXTH_CONFIG = TrainingConfig(
    name="sixth",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=False,
    dropout=0.35,
    batch_size=512,
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
    hidden_size=96,
    num_layers=2,
    bidirectional=False,
    dropout=0.4,
    batch_size=256,
    learning_rate=3e-4,
    window_size=40,
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


CONFIG_ONLY2 = TrainingConfig(
    name="only2",
    model_type="gru",
    hidden_size=25,
    num_layers=2,
    bidirectional=False,
    dropout=0.4,
    batch_size=256,
    learning_rate=5e-4,
    window_size=45,
    weight_decay=1e-5,
)

# AYDI_1_config:
SEVENTH_CONFIG2 = TrainingConfig(
    name="seventh",
    hidden_size=18,
    num_layers=1,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=90,
)


ALAA_CONFIG_3 = TrainingConfig(
    name="alaa3",
    model_type="gru",
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.35,
    batch_size=512,
    learning_rate=3e-4,
    weight_decay=5e-4,
    window_size=45,
)

'''
Epoch 001 | train_loss=0.6728  train_acc=0.5807  val_loss=0.7239  val_acc=0.4946
Epoch 002 | train_loss=0.6642  train_acc=0.5929  val_loss=0.7250  val_acc=0.4994
Epoch 003 | train_loss=0.6606  train_acc=0.5962  val_loss=0.7242  val_acc=0.4896
Epoch 004 | train_loss=0.6586  train_acc=0.5981  val_loss=0.7259  val_acc=0.4960
Epoch 005 | train_loss=0.6573  train_acc=0.5996  val_loss=0.7524  val_acc=0.4744

'''

# Epoch 001 | train_loss=0.6810  train_acc=0.5644  val_loss=0.6949  val_acc=0.5270
# Epoch 002 | train_loss=0.6727  train_acc=0.5800  val_loss=0.7006  val_acc=0.5088
# Epoch 003 | train_loss=0.6690  train_acc=0.5868  val_loss=0.7056  val_acc=0.4976
# Epoch 004 | train_loss=0.6665  train_acc=0.5903  val_loss=0.7123  val_acc=0.4953
# Epoch 005 | train_loss=0.6641  train_acc=0.5939  val_loss=0.7126  val_acc=0.4921
# Epoch 006 | train_loss=0.6626  train_acc=0.5962  val_loss=0.7175  val_acc=0.5055

# Epoch 001 | train_loss=0.6790  train_acc=0.5682  val_loss=0.7038  val_acc=0.5190
# Epoch 002 | train_loss=0.6678  train_acc=0.5876  val_loss=0.7094  val_acc=0.5043
# Epoch 003 | train_loss=0.6609  train_acc=0.5976  val_loss=0.7173  val_acc=0.4920
# Epoch 004 | train_loss=0.6563  train_acc=0.6042  val_loss=0.7395  val_acc=0.4746
# Epoch 005 | train_loss=0.6530  train_acc=0.6085  val_loss=0.7155  val_acc=0.4901


ALAA_CONFIG_4 = TrainingConfig(
    name="alaa3",
    model_type="gru",
    hidden_size=48,
    num_layers=2,
    bidirectional=False,
    dropout=0.35,
    batch_size=256,
    learning_rate=3e-4,
    weight_decay=5e-4,
    window_size=45,
)

"""

PS D:\Development\PycharmProjects\stock-trend-prediction> python -m src.models.train_model
Normalizing
Building samples: 100%|███████████████████████████████████████████████████████████| 4365/4365 [00:10<00:00, 409.18it/s]
Building samples: 100%|██████████████████████████████████████████████████████████| 4770/4770 [00:04<00:00, 1051.35it/s]
Building samples: 100%|██████████████████████████████████████████████████████████| 4924/4924 [00:04<00:00, 1024.66it/s]
Starting training from scratch...
train: 100%|████████████████████████████████████████████████████████| 30108/30108 [09:12<00:00, 54.49it/s, loss=0.6739]
val: 100%|███████████████████████████████████████████████████████████| 5553/5553 [00:39<00:00, 139.40it/s, loss=0.7116]
Epoch 001 | train_loss=0.6750  train_acc=0.5776  val_loss=0.7340  val_acc=0.4522
train: 100%|████████████████████████████████████████████████████████| 30108/30108 [09:05<00:00, 55.16it/s, loss=0.6724]
val: 100%|███████████████████████████████████████████████████████████| 5553/5553 [00:39<00:00, 140.14it/s, loss=0.7172]
Epoch 002 | train_loss=0.6718  train_acc=0.5812  val_loss=0.7264  val_acc=0.4506

"""

# Epoch 001 | train_loss=0.6788  train_acc=0.5701  val_loss=0.6990  val_acc=0.5130
# Epoch 002 | train_loss=0.6630  train_acc=0.5948  val_loss=0.7284  val_acc=0.4849
# Epoch 003 | train_loss=0.6528  train_acc=0.6089  val_loss=0.7254  val_acc=0.5025
# Epoch 004 | train_loss=0.6446  train_acc=0.6197  val_loss=0.7317  val_acc=0.4952
# Epoch 005 | train_loss=0.6368  train_acc=0.6291  val_loss=0.7493  val_acc=0.5052











"""
Epoch 001 | train_loss=0.6730  train_acc=0.5807  val_loss=0.7292  val_acc=0.4564
Epoch 002 | train_loss=0.6438  train_acc=0.6183  val_loss=0.7448  val_acc=0.4892
Epoch 003 | train_loss=0.6227  train_acc=0.6447  val_loss=0.7383  val_acc=0.5103
Epoch 004 | train_loss=0.6077  train_acc=0.6616  val_loss=0.7488  val_acc=0.5185
Epoch 005 | train_loss=0.5954  train_acc=0.6745  val_loss=0.7830  val_acc=0.5120
Epoch 006 | train_loss=0.5856  train_acc=0.6839  val_loss=0.8117  val_acc=0.5082
Epoch 007 | train_loss=0.5778  train_acc=0.6914  val_loss=0.8158  val_acc=0.5061
Epoch 008 | train_loss=0.5720  train_acc=0.6969  val_loss=0.8325  val_acc=0.5063
Epoch 009 | train_loss=0.5670  train_acc=0.7017  val_loss=0.8288  val_acc=0.5125
Epoch 010 | train_loss=0.5632  train_acc=0.7050  val_loss=0.8391  val_acc=0.5105

"""

"""
Epoch 001 | train_loss=0.6711  train_acc=0.5775  val_loss=0.7250  val_acc=0.4845
Epoch 002 | train_loss=0.6578  train_acc=0.6009  val_loss=0.7427  val_acc=0.4832
Epoch 003 | train_loss=0.6514  train_acc=0.6110  val_loss=0.7372  val_acc=0.5051
Epoch 004 | train_loss=0.6479  train_acc=0.6160  val_loss=0.7426  val_acc=0.4973
Epoch 005 | train_loss=0.6454  train_acc=0.6195  val_loss=0.7272  val_acc=0.5085
Epoch 006 | train_loss=0.6436  train_acc=0.6225  val_loss=0.7571  val_acc=0.4972
Epoch 007 | train_loss=0.6420  train_acc=0.6247  val_loss=0.7477  val_acc=0.5043
Epoch 008 | train_loss=0.6406  train_acc=0.6265  val_loss=0.7618  val_acc=0.4914
Epoch 009 | train_loss=0.6397  train_acc=0.6278  val_loss=0.7310  val_acc=0.5112
Epoch 010 | train_loss=0.6390  train_acc=0.6287  val_loss=0.7446  val_acc=0.5105

"""
CONFIG_ONLY3 = TrainingConfig(
    name="only3",
    model_type="gru_attention",
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=45,
    weight_decay=1e-4,
    attention_dropout=0.3,
)