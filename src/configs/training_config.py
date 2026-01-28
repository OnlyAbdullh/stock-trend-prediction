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
    bidirectional=True,
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
without L2
"""
THIRD_CONFIG = TrainingConfig(
    name="third",
    model_type="gru",
    hidden_size=64,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=256,
    learning_rate=8e-4,
    window_size=45,
)
#Without l2
# 100%|█████████████████████████████████████████████████████████████| 61103/61103 [31:57<00:00, 31.87it/s, loss=0.6268] 
# 100%|████████████████████████████████████████████████████████████| 13094/13094 [02:10<00:00, 100.71it/s, loss=0.7539] 
# Epoch 001 | train_loss=0.6583  train_acc=0.5985  val_loss=0.7395  val_acc=0.4875
# 100%|█████████████████████████████████████████████████████████████| 61103/61103 [24:00<00:00, 42.42it/s, loss=0.6337] 
# 100%|█████████████████████████████████████████████████████████████| 13094/13094 [02:24<00:00, 90.57it/s, loss=0.7497] 
# Epoch 002 | train_loss=0.6329  train_acc=0.6306  val_loss=0.7946  val_acc=0.4751
# 100%|█████████████████████████████████████████████████████████████| 61103/61103 [20:14<00:00, 50.33it/s, loss=0.6157] 
# 100%|████████████████████████████████████████████████████████████| 13094/13094 [01:49<00:00, 119.91it/s, loss=0.7211] 
# Epoch 003 | train_loss=0.6231  train_acc=0.6417  val_loss=0.7705  val_acc=0.4784
# 100%|█████████████████████████████████████████████████████████████| 61103/61103 [25:03<00:00, 40.64it/s, loss=0.6536] 
# 100%|████████████████████████████████████████████████████████████| 13094/13094 [02:07<00:00, 102.43it/s, loss=0.7544] 
# Epoch 004 | train_loss=0.6177  train_acc=0.6474  val_loss=0.7744  val_acc=0.4874
# 100%|█████████████████████████████████████████████████████████████| 61103/61103 [21:32<00:00, 47.28it/s, loss=0.5601] 
# 100%|████████████████████████████████████████████████████████████| 13094/13094 [02:05<00:00, 104.31it/s, loss=0.7784]
# Epoch 005 | train_loss=0.6147  train_acc=0.6506  val_loss=0.7792  val_acc=0.4833
FOURTH_CONFIG = TrainingConfig(
    name="fourth",
    model_type="gru",
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
    model_type="gru",
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
    model_type="gru",
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
    model_type="gru",
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
    model_type="gru",
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
    model_type="gru",
    hidden_size=192,
    num_layers=2,
    bidirectional=True,
    dropout=0.4,
    batch_size=512,
    learning_rate=3e-4,
    window_size=60,
)

# Starting training... NINTH_CONFIG
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
    model_type="gru",
    hidden_size=160,
    num_layers=3,
    bidirectional=True,
    dropout=0.4,
    batch_size=512,
    learning_rate=2e-4,
    window_size=75,
)


# Starting training... TENTH_CONFIG
# 100%|███████████████████████████████████████████████████████████████| 15003/15003 [07:46<00:00, 32.18it/s, loss=0.6677]
# 100%|█████████████████████████████████████████████████████████████████| 3215/3215 [01:43<00:00, 30.95it/s, loss=0.7068]
# Epoch 001 | train_loss=0.6673  train_acc=0.5868  val_loss=0.7502  val_acc=0.4810
# train: 100%|████████████████████████████████████████████████████████| 15003/15003 [08:34<00:00, 29.15it/s, loss=0.6393]
# val: 100%|████████████████████████████████████████████████████████████| 3215/3215 [02:05<00:00, 25.69it/s, loss=0.7441]
# Epoch 002 | train_loss=0.6239  train_acc=0.6401  val_loss=0.7672  val_acc=0.4973
# train: 100%|████████████████████████████████████████████████████████| 15003/15003 [08:15<00:00, 30.28it/s, loss=0.5597]
# val: 100%|████████████████████████████████████████████████████████████| 3215/3215 [01:54<00:00, 27.99it/s, loss=0.8605]
# Epoch 003 | train_loss=0.5917  train_acc=0.6733  val_loss=0.8157  val_acc=0.4919
# train: 100%|████████████████████████████████████████████████████████| 15003/15003 [08:28<00:00, 29.51it/s, loss=0.5071]
# val: 100%|████████████████████████████████████████████████████████████| 3215/3215 [01:52<00:00, 28.70it/s, loss=0.8812]
# Epoch 004 | train_loss=0.5616  train_acc=0.7005  val_loss=0.8993  val_acc=0.4858
# train: 100%|████████████████████████████████████████████████████████| 15003/15003 [08:25<00:00, 29.70it/s, loss=0.5589]
# val: 100%|████████████████████████████████████████████████████████████| 3215/3215 [01:55<00:00, 27.90it/s, loss=1.0025]
# Epoch 005 | train_loss=0.5316  train_acc=0.7256  val_loss=0.9858  val_acc=0.4956
 
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
    model_type="gru",
    hidden_size=32,
    num_layers=2,
    bidirectional=False,
    dropout=0.4,
    batch_size=256,
    learning_rate=5e-4,
    window_size=45,
)
"""
Epoch 001 | train_loss=0.6712  train_acc=0.5814  val_loss=0.7397  val_acc=0.4650
Epoch 002 | train_loss=0.6568  train_acc=0.6007  val_loss=0.7548  val_acc=0.4733
Epoch 003 | train_loss=0.6487  train_acc=0.6126  val_loss=0.7651  val_acc=0.4734
Epoch 004 | train_loss=0.6436  train_acc=0.6194  val_loss=0.7520  val_acc=0.4824
with L2"""

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

ALAA_CONFIG_1 = TrainingConfig(
    name="alaa1",
    model_type="gru",
    hidden_size=64,
    num_layers=2,
    bidirectional=True,
    dropout=0.5,
    batch_size=512,
    learning_rate=3e-4,
    window_size=30,
)

ALAA_CONFIG_2 = TrainingConfig(
    name="alaa2",
    model_type="gru",
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_size=1024,
    learning_rate=3e-4,
    window_size=30,
)

# Starting training from scratch...
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [01:58<00:00, 64.97it/s, loss=0.6567]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:32<00:00, 50.21it/s, loss=0.7576]
# Epoch 001 | train_loss=0.6670  train_acc=0.5886  val_loss=0.7363  val_acc=0.4767
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [02:02<00:00, 62.94it/s, loss=0.6394]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:36<00:00, 45.88it/s, loss=0.7276]
# Epoch 002 | train_loss=0.6477  train_acc=0.6142  val_loss=0.7376  val_acc=0.4915
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [02:14<00:00, 57.15it/s, loss=0.6352]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:34<00:00, 47.28it/s, loss=0.7031]
# Epoch 003 | train_loss=0.6361  train_acc=0.6279  val_loss=0.7593  val_acc=0.4939
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [02:04<00:00, 61.86it/s, loss=0.6297]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:31<00:00, 52.37it/s, loss=0.7129]
# Epoch 004 | train_loss=0.6273  train_acc=0.6378  val_loss=0.7521  val_acc=0.4961
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [01:55<00:00, 66.54it/s, loss=0.6162]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:32<00:00, 51.28it/s, loss=0.7277]
# Epoch 005 | train_loss=0.6211  train_acc=0.6444  val_loss=0.7628  val_acc=0.5065
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [02:03<00:00, 62.48it/s, loss=0.6012]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:36<00:00, 45.78it/s, loss=0.7244]
# Epoch 006 | train_loss=0.6160  train_acc=0.6496  val_loss=0.7551  val_acc=0.4947
# train: 100%|██████████████████████████████████████████████████████████| 7712/7712 [02:15<00:00, 56.88it/s, loss=0.6071]
# val: 100%|████████████████████████████████████████████████████████████| 1653/1653 [00:33<00:00, 49.33it/s, loss=0.7573]
# Epoch 007 | train_loss=0.6118  train_acc=0.6539  val_loss=0.7733  val_acc=0.5009


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

# Starting training from scratch...
# train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:57<00:00, 130.29it/s, loss=0.6572]
# val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:24<00:00, 132.08it/s, loss=0.7335]
# Epoch 001 | train_loss=0.6711  train_acc=0.5832  val_loss=0.7533  val_acc=0.4776
# train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:49<00:00, 139.44it/s, loss=0.6651]
# val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:23<00:00, 139.50it/s, loss=0.7403]
# Epoch 002 | train_loss=0.6616  train_acc=0.5955  val_loss=0.7441  val_acc=0.4887
# train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:49<00:00, 139.17it/s, loss=0.6661]
# val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:24<00:00, 134.70it/s, loss=0.7298]
# Epoch 003 | train_loss=0.6564  train_acc=0.6020  val_loss=0.7309  val_acc=0.4840
# train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:49<00:00, 139.68it/s, loss=0.6504]
# val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:22<00:00, 142.43it/s, loss=0.7899]
# Epoch 004 | train_loss=0.6526  train_acc=0.6073  val_loss=0.7541  val_acc=0.4857
# train: 100%|███████████████████████████████████████████████████████| 15276/15276 [01:48<00:00, 140.92it/s, loss=0.6453]
# val: 100%|███████████████████████████████████████████████████████████| 3274/3274 [00:22<00:00, 143.19it/s, loss=0.7576]
# Epoch 005 | train_loss=0.6499  train_acc=0.6113  val_loss=0.7582  val_acc=0.4693