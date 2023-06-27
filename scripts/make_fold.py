from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from src.train_tools import seed_everything
from src.utils import get_stream_logger

seed_everything(42)
logger = get_stream_logger(20)

data_root = Path("./input/contrails-images-ash-color")
image_path = data_root / "contrails"
train_df = pd.read_csv(data_root / "train_df.csv")
train_df["path"] = str(image_path) + "/" + train_df["record_id"].astype(str) + ".npy"

valid_df = pd.read_csv(data_root / "valid_df.csv")
valid_df["path"] = str(image_path) + "/" + valid_df["record_id"].astype(str) + ".npy"

df = pd.concat([train_df, valid_df]).reset_index(drop=True)
logger.info(f"\n{df.head()}")
logger.info(f"\n{df.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    df.loc[valid_idx, "fold"] = int(fold)
df["fold"] = df["fold"].astype(int)

logger.info(f"\n{df.head()}")
logger.info(f"\n{df.shape}")

df.to_csv(data_root / "df.csv", index=False)
