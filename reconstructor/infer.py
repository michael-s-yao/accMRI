"""
Inference driver program for accelerated MRI reconstruction.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from args import Inference
import matplotlib.pyplot as plt
import os
from pytorch_lightning import Trainer
import requests
import sys
import torch
from tqdm import tqdm
from typing import List, Dict, Any
import yaml

from pl_modules.data_module import DataModule
from pl_modules.reconstructor_module import ReconstructorModule


def load_inputs(data_path: str) -> List[str]:
    inputs = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            if f.endswith(".h5"):
                inputs.append(f)
    elif os.path.isfile(data_path) and data_path.endswith(".h5"):
        inputs.append(data_path)
    else:
        raise ValueError(
            f"No valid inputs associated with given data path {data_path}."
        )


def load_models(lightning_logs: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(lightning_logs):
        raise ValueError(
            f"{lightning_logs} must be a valid lightning_logs/ directory."
        )
    models = []
    for f in sorted(os.listdir(lightning_logs)):
        if not os.path.isdir(os.path.join(lightning_logs, f)):
            continue
        if f == "fair":
            continue
        hparams = None
        with open(os.path.join(lightning_logs, f, "hparams.yaml"), "r") as p:
            hparams = yaml.safe_load(p)

        key = "[" + f.split("_")[-1] + "] "
        key += "Multicoil " if hparams["is_multicoil"] else "Singlecoil "
        if hparams["model"] == "unet":
            key += "UNet "
        elif hparams["model"] == "varnet":
            key += "VarNet "
        else:
            raise ValueError(
                f"Unrecognized reconstructor model {hparams['model']}."
            )
        key += "with " + str(hparams["center_crop"][0]) + " by "
        key += str(hparams["center_crop"][-1]) + " final image size"

        min_val_loss = 1e12
        model = None
        for ckpt in os.listdir(os.path.join(lightning_logs, f, "checkpoints")):
            if not ckpt.endswith(".ckpt"):
                continue
            # Use the last.ckpt model if available.
            if "last" in ckpt:
                model = ckpt
                break
            # Otherwise, use the ckpt model with the lowest validation loss.
            # We assume that the file name is of the format
            # `reconstructor-{start_time}-{epoch}-{validation_loss}.ckpt`.
            val_loss = float("0." + ckpt.split(".")[-2])
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                model = ckpt

        if model is None:
            continue
        models.append({
            "key": key,
            "model": os.path.join(lightning_logs, f, "checkpoints", model),
            "model_type": hparams["model"],
            "center_crop": tuple(hparams["center_crop"]),
        })

    # fastMRI models.
    FAIR = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    VARNET = {
        "KNEE": "knee_leaderboard_state_dict.pt",
    }
    if not os.path.isdir(os.path.join(lightning_logs, "fair")):
        os.mkdir(os.path.join(lightning_logs, "fair"))
    models.append({
        "key": "[" + str(len(models)) + "] FAIR Multicoil VarNet",
        "model": os.path.abspath(
            os.path.join(lightning_logs, "fair", "knee.ckpt")
        ),
        "model_type": "varnet",
        "center_crop": (320, 320),
    })
    if not os.path.isfile(models[-1]["model"]):
        download_model(FAIR + VARNET["KNEE"], models[-1]["model"])

    return models


def download_model(url, fname):
    if os.path.isfile(fname):
        print(f"{fname} is already a locally downloaded model. Exiting...")
        return
    response = requests.get(url, timeout=10, stream=True)
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading model state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True
    )
    with open(fname, "w+b") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def infer():
    args = Inference.build_args()

    models = load_models(args.lightning_logs)
    if len(models) == 0:
        print("No models found in {args.lightning_logs} directory. Exiting...")
        sys.exit(0)
    elif len(models) == 1:
        print("Only one model found. Using " + models[0]["key"])
    if 0 <= args.model_option < len(models):
        ckpt_path = models[args.model_option]["model"]
        print("Using " + models[args.model_option]["key"])
        model_option = args.model_option
    else:
        print("Choose the reconstructor model to use:")
        for option in models:
            print(option["key"])
        print()
        model_option = int(input("Enter model number: "))
        ckpt_path = models[model_option]["model"]

    data_module = DataModule(
        args.data_path,
        train_dir=None,
        val_dir=None,
        test_dir=None,
        center_crop=models[model_option]["center_crop"]
    )
    if "fair" not in ckpt_path.lower():
        infer_model = ReconstructorModule.load_from_checkpoint(ckpt_path)
    else:
        infer_model = ReconstructorModule(model="varnet", is_multicoil=True)
        infer_model.reconstructor.model.load_state_dict(
            torch.load(ckpt_path)
        )
    infer_model.reconstructor = infer_model.reconstructor.eval()
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False,
        logger=False,
        max_epochs=1
    )
    predictions = trainer.predict(
        infer_model, dataloaders=data_module.predict_dataloader()
    )

    if args.save_path is not None and len(args.save_path) > 0:
        save_predictions(predictions, args.save_path)


def save_predictions(predictions: List[dict], save_path: str) -> None:
    """
    Plots predictions and saves to `save_path` folder in working directory.
    Input:
        predictions: a list of predictions.
        save_path: relatvie folder path to save to.
    Returns:
        None.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for prediction in predictions:
        cols = 1 if prediction["target"] is None else 2
        fig, ax = plt.subplots(1, cols)
        if prediction["target"] is not None:
            ax, axtarg = ax
        ax.imshow(prediction["output"][0], cmap="gray")
        ax.axis("off")
        title = "Prediction: " if prediction["target"] is not None else ""
        title += "Slice " + str(prediction["slice_num"].item())
        ax.set_title(title)

        if prediction["target"] is not None:
            axtarg.imshow(prediction["target"][0], cmap="gray")
            axtarg.axis("off")
            axtarg.set_title(
                "SSIM: " + format(prediction["ssim"].item(), ".3f")
            )

        fig.suptitle(
            "Acceleration Factor: " + format(prediction["acc_factor"], ".3f")
        )

        sub_path = os.path.join(
            save_path, prediction["fname"][0].split(".")[0].split("/")[-1]
        )
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        plt.savefig(
            os.path.join(
                sub_path, str(prediction["slice_num"].item()) + ".png"
            ),
            bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    infer()