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
    hidden_size=96, # => 32
    num_layers=2,
    bidirectional=True, # => False
    dropout=0.3,
    batch_size=256,
    learning_rate=5e-4,
    window_size=60, # => 30
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
# # Results of AYDI 1 CONFIG:
# train: 100%|████████████████████████████████████| 29747/29747 [08:06<00:00, 61.18it/s, loss=0.6885]
# val: 100%|████████████████████████████████████████| 6375/6375 [01:21<00:00, 78.27it/s, loss=0.7078]
# Epoch 001 | train_loss=0.6717  train_acc=0.5825  val_loss=0.7290  val_acc=0.4726
# train: 100%|████████████████████████████████████| 29747/29747 [07:40<00:00, 64.59it/s, loss=0.6513]
# val: 100%|████████████████████████████████████████| 6375/6375 [01:20<00:00, 79.64it/s, loss=0.6836]
# Epoch 002 | train_loss=0.6608  train_acc=0.5963  val_loss=0.7317  val_acc=0.4817
# train: 100%|████████████████████████████████████| 29747/29747 [07:12<00:00, 68.82it/s, loss=0.6506]
# val: 100%|████████████████████████████████████████| 6375/6375 [01:21<00:00, 78.33it/s, loss=0.6770]
# Epoch 003 | train_loss=0.6556  train_acc=0.6030  val_loss=0.7366  val_acc=0.4589
# train: 100%|████████████████████████████████████| 29747/29747 [08:21<00:00, 59.29it/s, loss=0.6657] 
# val: 100%|████████████████████████████████████████| 6375/6375 [01:20<00:00, 78.83it/s, loss=0.6601]
# Epoch 004 | train_loss=0.6521  train_acc=0.6078  val_loss=0.7346  val_acc=0.4775
# train: 100%|████████████████████████████████████| 29747/29747 [07:32<00:00, 65.73it/s, loss=0.6482] 
# val: 100%|████████████████████████████████████████| 6375/6375 [01:10<00:00, 90.38it/s, loss=0.6876]
# Epoch 005 | train_loss=0.6494  train_acc=0.6116  val_loss=0.7463  val_acc=0.4720
# train: 100%|████████████████████████████████████| 29747/29747 [07:00<00:00, 70.69it/s, loss=0.6558] 
# val: 100%|███████████████████████████████████████| 6375/6375 [01:03<00:00, 100.40it/s, loss=0.6999]
# Epoch 006 | train_loss=0.6473  train_acc=0.6144  val_loss=0.7545  val_acc=0.4627
# train: 100%|████████████████████████████████████| 29747/29747 [07:39<00:00, 64.77it/s, loss=0.6640] 
# val: 100%|████████████████████████████████████████| 6375/6375 [01:28<00:00, 72.07it/s, loss=0.6811]
# Epoch 007 | train_loss=0.6456  train_acc=0.6162  val_loss=0.7400  val_acc=0.4671