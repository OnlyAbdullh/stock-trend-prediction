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
# Starting training from scratch...
# train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [05:37<00:00, 66.90it/s, loss=0.6249]
# train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [05:37<00:00, 66.90it/s, loss=0.6249]
# val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:45<00:00, 106.24it/s, loss=0.6759]
# Epoch 001 | train_loss=0.6810  train_acc=0.5644  val_loss=0.6949  val_acc=0.5270
# train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [06:46<00:00, 55.47it/s, loss=0.6523]
# val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:46<00:00, 104.22it/s, loss=0.7020]
# Epoch 002 | train_loss=0.6727  train_acc=0.5800  val_loss=0.7006  val_acc=0.5088
# train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [06:06<00:00, 61.49it/s, loss=0.6553]
# val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:44<00:00, 108.79it/s, loss=0.6942]
# Epoch 003 | train_loss=0.6690  train_acc=0.5868  val_loss=0.7056  val_acc=0.4976
# train:   0%|                                                                                                     Epoch 002 | train_loss=0.6727  train_acc=0.5800  val_loss=0.7006  val_acc=0.5088
# train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [06:06<00:00, 61.49it/s, loss=0.6553]
# val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:44<00:00, 108.79it/s, loss=0.6942]
# Epoch 003 | train_loss=0.6690  train_acc=0.5868  val_loss=0.7056  val_acc=0.4976
# train:   0%|                                                                                                                         | 0/22562 [00:00<?, ?it/s]train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [05:27<00:00, 68.91it/s, loss=0.6627]
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:42<00:00, 114.03it/s, loss=0.6996]
# Epoch 004 | train_loss=0.6665  train_acc=0.5903  val_loss=0.7123  val_acc=0.4953
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [05:32<00:00, 67.83it/s, loss=0.6344]
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:40<00:00, 120.33it/s, loss=0.7415]
# Epoch 005 | train_loss=0.6641  train_acc=0.5939  val_loss=0.7126  val_acc=0.4921
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 22562/22562 [05:21<00:00, 70.25it/s, loss=0.6319]
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4835/4835 [00:37<00:00, 129.68it/s, loss=0.7195]
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
# Starting training from scratch...
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:02<00:00, 60.50it/s, loss=0.6641] 
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:02<00:00, 60.50it/s, loss=0.6641] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:41<00:00, 114.03it/s, loss=0.6979] 
# Epoch 001 | train_loss=0.6790  train_acc=0.5682  val_loss=0.7038  val_acc=0.5190
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [05:57<00:00, 61.50it/s, loss=0.6681] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:41<00:00, 113.64it/s, loss=0.7382] 
# Epoch 002 | train_loss=0.6678  train_acc=0.5876  val_loss=0.7094  val_acc=0.5043
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:13<00:00, 58.80it/s, loss=0.6684] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:43<00:00, 107.79it/s, loss=0.7556] 
# Epoch 003 | train_loss=0.6609  train_acc=0.5976  val_loss=0.7173  val_acc=0.4920
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [05:59<00:00, 61.15it/s, loss=0.6056] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:38<00:00, 123.81it/s, loss=0.7657] 
# Epoch 004 | train_loss=0.6563  train_acc=0.6042  val_loss=0.7395  val_acc=0.4746
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:17<00:00, 58.09it/s, loss=0.6891] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:36<00:00, 127.68it/s, loss=0.7506] 
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
# Starting training from scratch...
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [05:20<00:00, 68.47it/s, loss=0.6344] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:38<00:00, 122.87it/s, loss=0.6974] 
# Epoch 001 | train_loss=0.6788  train_acc=0.5701  val_loss=0.6990  val_acc=0.5130
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [05:55<00:00, 61.84it/s, loss=0.6741] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:43<00:00, 108.16it/s, loss=0.7201] 
# Epoch 002 | train_loss=0.6630  train_acc=0.5948  val_loss=0.7284  val_acc=0.4849
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:20<00:00, 57.76it/s, loss=0.6972] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:46<00:00, 100.13it/s, loss=0.7101] 
# Epoch 003 | train_loss=0.6528  train_acc=0.6089  val_loss=0.7254  val_acc=0.5025
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:19<00:00, 57.85it/s, loss=0.6150] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:44<00:00, 106.90it/s, loss=0.7313] 
# Epoch 004 | train_loss=0.6446  train_acc=0.6197  val_loss=0.7317  val_acc=0.4952
# train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 21958/21958 [06:07<00:00, 59.70it/s, loss=0.7030] 
# val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4706/4706 [00:45<00:00, 103.75it/s, loss=0.7649] 
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
SEVENTH_CONFIG = TrainingConfig(
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

Normalizing all ticker data (Mode 2)...
Normalizing: 100%|█████████████████████████████████████████████████████████████████| 4924/4924 [02:44<00:00, 29.89it/s]
✓ Normalization Mode 2 complete (Advanced Logic Applied)
Starting training from scratch...
train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:51<00:00, 136.70it/s, loss=0.6675]
val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:26<00:00, 125.70it/s, loss=0.7135]
Epoch 001 | train_loss=0.6728  train_acc=0.5807  val_loss=0.7239  val_acc=0.4946
train: 100%|███████████████████████████████████████████████████████| 15276/15276 [02:01<00:00, 125.46it/s, loss=0.6754]
val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:27<00:00, 117.19it/s, loss=0.7226]
Epoch 002 | train_loss=0.6642  train_acc=0.5929  val_loss=0.7250  val_acc=0.4994
train: 100%|███████████████████████████████████████████████████████| 15276/15276 [02:12<00:00, 115.55it/s, loss=0.6704]
val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:24<00:00, 132.85it/s, loss=0.7008]
Epoch 003 | train_loss=0.6606  train_acc=0.5962  val_loss=0.7242  val_acc=0.4896
train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:50<00:00, 137.65it/s, loss=0.6529]
val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:24<00:00, 134.00it/s, loss=0.7121]
Epoch 004 | train_loss=0.6586  train_acc=0.5981  val_loss=0.7259  val_acc=0.4960
train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:50<00:00, 137.88it/s, loss=0.6633]
val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:24<00:00, 135.64it/s, loss=0.7393]
Epoch 005 | train_loss=0.6573  train_acc=0.5996  val_loss=0.7524  val_acc=0.4744

Saved checkpoint to: D:\Development\PycharmProjects\stock-trend-prediction\models\gru_alaa3_20260128_234520.pt

'''




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











"""
TrainingConfig(
    name="sixth",
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.35,
    batch_size=256,
    learning_rate=8e-4,
    window_size=60,
)
workers = 4
train_loader = DataLoader(
        train_ds,
        batch_size=1024,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2,
    )
Starting training from scratch...
train: 100%|██████████| 7568/7568 [05:33<00:00, 22.68it/s, loss=0.6652]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.29it/s, loss=0.7014]
Epoch 001 | train_loss=0.6730  train_acc=0.5807  val_loss=0.7292  val_acc=0.4564
train: 100%|██████████| 7568/7568 [05:48<00:00, 21.69it/s, loss=0.6269]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.29it/s, loss=0.6747]
Epoch 002 | train_loss=0.6438  train_acc=0.6183  val_loss=0.7448  val_acc=0.4892
train: 100%|██████████| 7568/7568 [05:50<00:00, 21.57it/s, loss=0.6061]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.19it/s, loss=0.6865]
Epoch 003 | train_loss=0.6227  train_acc=0.6447  val_loss=0.7383  val_acc=0.5103
train: 100%|██████████| 7568/7568 [05:49<00:00, 21.67it/s, loss=0.6037]
val: 100%|██████████| 1622/1622 [00:31<00:00, 52.22it/s, loss=0.6841]
Epoch 004 | train_loss=0.6077  train_acc=0.6616  val_loss=0.7488  val_acc=0.5185
train: 100%|██████████| 7568/7568 [05:33<00:00, 22.71it/s, loss=0.6101]
val: 100%|██████████| 1622/1622 [00:31<00:00, 51.91it/s, loss=0.7024]
Epoch 005 | train_loss=0.5954  train_acc=0.6745  val_loss=0.7830  val_acc=0.5120
train: 100%|██████████| 7568/7568 [05:33<00:00, 22.72it/s, loss=0.5891]
val: 100%|██████████| 1622/1622 [00:31<00:00, 52.05it/s, loss=0.7458]
Epoch 006 | train_loss=0.5856  train_acc=0.6839  val_loss=0.8117  val_acc=0.5082
train: 100%|██████████| 7568/7568 [05:36<00:00, 22.51it/s, loss=0.5825]
val: 100%|██████████| 1622/1622 [00:31<00:00, 51.60it/s, loss=0.7731]
Epoch 007 | train_loss=0.5778  train_acc=0.6914  val_loss=0.8158  val_acc=0.5061
train: 100%|██████████| 7568/7568 [05:50<00:00, 21.59it/s, loss=0.5550]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.08it/s, loss=0.7888]
Epoch 008 | train_loss=0.5720  train_acc=0.6969  val_loss=0.8325  val_acc=0.5063
train: 100%|██████████| 7568/7568 [05:49<00:00, 21.68it/s, loss=0.5587]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.29it/s, loss=0.7543]
Epoch 009 | train_loss=0.5670  train_acc=0.7017  val_loss=0.8288  val_acc=0.5125
train: 100%|██████████| 7568/7568 [05:48<00:00, 21.73it/s, loss=0.5558]
val: 100%|██████████| 1622/1622 [00:35<00:00, 46.04it/s, loss=0.7827]
Epoch 010 | train_loss=0.5632  train_acc=0.7050  val_loss=0.8391  val_acc=0.5105



import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.actual_hidden_size = hidden_size * self.num_directions

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.batch_norm = nn.BatchNorm1d(self.actual_hidden_size)

        self.fc1 = nn.Linear(self.actual_hidden_size, 12)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(12, 6)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(6, 1)



"""

"""
TrainingConfig(
    name="fifth",
    hidden_size=25,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=1e-3,
    window_size=60,
)
workers = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2,
    )
Starting training from scratch...
train: 100%|██████████████| 30272/30272 [06:59<00:00, 72.13it/s, loss=0.6853]
val: 100%|█████████████████| 6487/6487 [00:48<00:00, 133.94it/s, loss=0.7287]
Epoch 001 | train_loss=0.6711  train_acc=0.5775  val_loss=0.7250  val_acc=0.4845
train: 100%|██████████████| 30272/30272 [07:32<00:00, 66.94it/s, loss=0.6540]
val: 100%|██████████████████| 6487/6487 [01:13<00:00, 88.40it/s, loss=0.7216]
Epoch 002 | train_loss=0.6578  train_acc=0.6009  val_loss=0.7427  val_acc=0.4832
train: 100%|██████████████| 30272/30272 [07:15<00:00, 69.48it/s, loss=0.6196]
val: 100%|█████████████████| 6487/6487 [00:52<00:00, 122.87it/s, loss=0.7186]
Epoch 003 | train_loss=0.6514  train_acc=0.6110  val_loss=0.7372  val_acc=0.5051
train: 100%|██████████████| 30272/30272 [07:17<00:00, 69.26it/s, loss=0.6135]
val: 100%|█████████████████| 6487/6487 [00:52<00:00, 124.75it/s, loss=0.7067]
Epoch 004 | train_loss=0.6479  train_acc=0.6160  val_loss=0.7426  val_acc=0.4973
train: 100%|██████████████| 30272/30272 [07:04<00:00, 71.39it/s, loss=0.6651]
val: 100%|█████████████████| 6487/6487 [00:51<00:00, 125.20it/s, loss=0.6975]
Epoch 005 | train_loss=0.6454  train_acc=0.6195  val_loss=0.7272  val_acc=0.5085
train: 100%|██████████████| 30272/30272 [06:30<00:00, 77.44it/s, loss=0.5924]
val: 100%|█████████████████| 6487/6487 [00:55<00:00, 116.79it/s, loss=0.7147]
Epoch 006 | train_loss=0.6436  train_acc=0.6225  val_loss=0.7571  val_acc=0.4972
train: 100%|██████████████| 30272/30272 [06:57<00:00, 72.42it/s, loss=0.6419]
val: 100%|█████████████████| 6487/6487 [00:51<00:00, 126.77it/s, loss=0.6940]
Epoch 007 | train_loss=0.6420  train_acc=0.6247  val_loss=0.7477  val_acc=0.5043
train: 100%|██████████████| 30272/30272 [06:52<00:00, 73.44it/s, loss=0.6166]
val: 100%|█████████████████| 6487/6487 [00:57<00:00, 113.06it/s, loss=0.7145]
Epoch 008 | train_loss=0.6406  train_acc=0.6265  val_loss=0.7618  val_acc=0.4914
train: 100%|██████████████| 30272/30272 [06:46<00:00, 74.43it/s, loss=0.6278]
val: 100%|█████████████████| 6487/6487 [00:52<00:00, 122.61it/s, loss=0.6955]
Epoch 009 | train_loss=0.6397  train_acc=0.6278  val_loss=0.7310  val_acc=0.5112
train: 100%|██████████████| 30272/30272 [06:02<00:00, 83.46it/s, loss=0.6220]
val: 100%|█████████████████| 6487/6487 [00:48<00:00, 133.74it/s, loss=0.6793]
Epoch 010 | train_loss=0.6390  train_acc=0.6287  val_loss=0.7446  val_acc=0.5105



class GRUModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.actual_hidden_size = hidden_size * self.num_directions

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.batch_norm = nn.BatchNorm1d(self.actual_hidden_size)

        self.fc1 = nn.Linear(self.actual_hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(16, 4)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(4, 1)




"""