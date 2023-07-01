from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.train_tools import _make_cls_label, seed_everything

data_root = Path("./input/contrails-images-ash-color")
df = pd.read_csv(data_root / "df.csv")

print(df)
print(df.shape)
print(df["path"][0])

for i in tqdm(range(len(df)), total=len(df), dynamic_ncols=True, leave=False):
    img_path = df["path"][i]
    # (256, 256, 4)
    img_and_label = np.load(img_path)
    img = img_and_label[:, :, :3]
    label = img_and_label[:, :, 3]
    # print(img.shape)
    # print(label.shape)

    label_tensor = torch.from_numpy(label).unsqueeze(0)
    # print(label_tensor.sum(dim=(1, 2)))
    cls_label = _make_cls_label(label_tensor)

    # print(cls_label)
    # print(type(cls_label))
    df.loc[i, "cls_label"] = cls_label.item()

df["cls_label"] = df["cls_label"].astype(int)
print(df)
print(df["cls_label"].value_counts())

df.to_csv(data_root / "df.csv", index=False)
