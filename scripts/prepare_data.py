from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import get_false_color, read_record, soft_label

data_dir = Path("./input/google-research-identify-contrails-reduce-global-warming/")
train_dir = data_dir / "train"
valid_dir = data_dir / "validation"
train_paths = list(train_dir.glob("*"))
valid_paths = list(valid_dir.glob("*"))


save_dir = Path("./input/prepared_np_imgs_weight1")
save_dir.mkdir(exist_ok=True, parents=True)

print(f"train_paths: {len(train_paths)}, valid_paths: {len(valid_paths)}")

all_metadata = []
for train_path in train_paths:
    record_data = read_record(train_path)
    n_times_before = 4
    raw_image = (
        get_false_color(record_data)[..., n_times_before]
        .reshape(256, 256, 3)
        .astype(np.float32)
    )
    pixel_mask = (
        np.load(train_path / "human_pixel_masks.npy")
        .reshape(256, 256)
        .astype(np.float32)
    )
    avg_mask = soft_label(np.load(train_path / "human_individual_masks.npy"), weight=1)

    record_id = train_path.stem
    # save imgs
    assets_save_dir = save_dir / record_id
    assets_save_dir.mkdir(exist_ok=True, parents=True)
    np.save(assets_save_dir / f"image.npy", raw_image)
    np.save(assets_save_dir / f"mask.npy", pixel_mask)
    np.save(assets_save_dir / f"avg_mask.npy", avg_mask)

    metadata = {}
    metadata["record_id"] = record_id
    metadata["image_path"] = str(assets_save_dir / f"image.npy")
    metadata["mask_path"] = str(assets_save_dir / f"mask.npy")
    metadata["avg_mask_path"] = str(assets_save_dir / f"avg_mask.npy")
    metadata["is_train"] = True
    all_metadata.append(metadata)


for valid_path in valid_paths:
    record_data = read_record(valid_path)
    n_times_before = 4
    raw_image = (
        get_false_color(record_data)[..., n_times_before]
        .reshape(256, 256, 3)
        .astype(np.float32)
    )
    pixel_mask = (
        np.load(valid_path / "human_pixel_masks.npy")
        .reshape(256, 256)
        .astype(np.float32)
    )

    record_id = valid_path.stem
    # save imgs
    assets_save_dir = save_dir / record_id
    assets_save_dir.mkdir(exist_ok=True, parents=True)
    np.save(assets_save_dir / f"image.npy", raw_image)
    np.save(assets_save_dir / f"mask.npy", pixel_mask)

    metadata = {}
    metadata["record_id"] = record_id
    metadata["image_path"] = str(assets_save_dir / f"image.npy")
    metadata["mask_path"] = str(assets_save_dir / f"mask.npy")
    metadata["avg_mask_path"] = str(assets_save_dir / f"avg_mask.npy")
    metadata["is_train"] = False
    all_metadata.append(metadata)


df = pd.DataFrame(all_metadata)
print(df.head())
df.to_csv(save_dir / "metadata.csv", index=False)
