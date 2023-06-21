import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.factory import Config, init_config
from src.dataset import ContrailsDataset
from src.models import ContrailsModel
from src.utils import rle_encode

ROOT = Path(".").resolve()


def get_loader(config: Config) -> DataLoader:
    filenames = os.listdir(config.data_root_path / "test")
    test_df = pd.DataFrame(filenames, columns=["record_id"])
    test_df["path"] = config.data_root_path / "test" / test_df["record_id"].astype(str)
    print(test_df)

    dataset = ContrailsDataset(
        df=test_df,
        image_size=config.image_size,
        train=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def get_contrails_model(config) -> nn.Module:
    # checkpoint = config.checkpoints[0]
    encoder_name = config.encoder_name

    contrails_model = ContrailsModel(encoder_name=encoder_name, encoder_weight=None)
    # contrails_model.load_state_dict(torch.load(checkpoint))

    return contrails_model


def main(config: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_loader(config)
    model = get_contrails_model(config).to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    threshold = config.threshold

    submission_path = "./input/google-research-identify-contrails-reduce-global-warming/sample_submission.csv"
    submission = pd.read_csv(submission_path, index_col="record_id")

    for i, batch in enumerate(loader):
        images, image_id = batch
        images = images.to(device)
        batch_size = len(images)

        with torch.inference_mode():
            outputs = model(images)
            logits = outputs["logits"]
        preds = torch.sigmoid(logits).to("cpu").detach().numpy()

        contrails_images = np.zeros((batch_size, 256, 256))
        contrails_images[preds[:, 0, :, :] < threshold] = 0
        contrails_images[preds[:, 0, :, :] >= threshold] = 1

        for image_idx in range(batch_size):
            current_contrails_image = contrails_images[image_idx, :, :]
            current_image_id = image_id[image_idx].item()

            rle = rle_encode(current_contrails_image)
            submission.loc[int(current_image_id), "encoded_pixels"] = rle

    print("\nSubmission:")
    print(submission)
    submission.to_csv("submission.csv", index=False)
    print("\n ### Inference is done! ###")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    config_path = f"configs.{args.config}"
    config = init_config(Config, config_path=config_path)
    main(config=config)
