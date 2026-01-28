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
    hidden_size=25,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=1e-3,
    window_size=60,
)

SIXTH_CONFIG = TrainingConfig(
    name="sixth",
    hidden_size=32,
    num_layers=1,
    bidirectional=False,
    dropout=0.35,
    batch_size=512,
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

# Starting training...
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:03<00:00, 41.60it/s, loss=0.6225]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:24<00:00, 22.50it/s, loss=0.6899]
# Epoch 001 | train_loss=0.6661  train_acc=0.5883  val_loss=0.7272  val_acc=0.4976
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:04<00:00, 41.56it/s, loss=0.5904]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:27<00:00, 21.98it/s, loss=0.7284]
# Epoch 002 | train_loss=0.6185  train_acc=0.6463  val_loss=0.7759  val_acc=0.4896
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [05:57<00:00, 42.35it/s, loss=0.5642]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:27<00:00, 22.00it/s, loss=0.7433]
# Epoch 003 | train_loss=0.5790  train_acc=0.6852  val_loss=0.8336  val_acc=0.4993
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:01<00:00, 41.89it/s, loss=0.5369]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:23<00:00, 22.59it/s, loss=0.8091]
# Epoch 004 | train_loss=0.5420  train_acc=0.7173  val_loss=0.8898  val_acc=0.5050
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:01<00:00, 41.93it/s, loss=0.5263]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:23<00:00, 22.53it/s, loss=0.9020]
# Epoch 005 | train_loss=0.5096  train_acc=0.7432  val_loss=0.9547  val_acc=0.5017
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [05:57<00:00, 42.35it/s, loss=0.4693]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:24<00:00, 22.47it/s, loss=0.9821]
# Epoch 006 | train_loss=0.4831  train_acc=0.7628  val_loss=1.0213  val_acc=0.4969
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [05:56<00:00, 42.44it/s, loss=0.4349]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:23<00:00, 22.62it/s, loss=0.9626]
# Epoch 007 | train_loss=0.4620  train_acc=0.7778  val_loss=1.0527  val_acc=0.5049
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [05:59<00:00, 42.10it/s, loss=0.4574]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:24<00:00, 22.49it/s, loss=1.0071]
# Epoch 008 | train_loss=0.4452  train_acc=0.7894  val_loss=1.0747  val_acc=0.5056
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:06<00:00, 41.31it/s, loss=0.4239]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:24<00:00, 22.48it/s, loss=1.0407]
# Epoch 009 | train_loss=0.4314  train_acc=0.7986  val_loss=1.1002  val_acc=0.5104
# 100%|███████████████████████████████████████████████████████████████| 15136/15136 [06:02<00:00, 41.81it/s, loss=0.4344]
# 100%|█████████████████████████████████████████████████████████████████| 3244/3244 [02:24<00:00, 22.38it/s, loss=1.0803]
# Epoch 010 | train_loss=0.4201  train_acc=0.8060  val_loss=1.1258  val_acc=0.5100

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


# Starting training...
# 100%|███████████████████████████████████████████████████████████████| 15003/15003 [07:46<00:00, 32.18it/s, loss=0.6677]
# 100%|█████████████████████████████████████████████████████████████████| 3215/3215 [01:43<00:00, 30.95it/s, loss=0.7068]
# Epoch 001 | train_loss=0.6673  train_acc=0.5868  val_loss=0.7502  val_acc=0.4810

 
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