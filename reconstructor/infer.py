"""
Inference driver program for accelerated MRI reconstruction.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from args import Inference
import os
from pytorch_lightning import Trainer
import torch
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
        hparams = None
        with open(os.path.join(lightning_logs, f, "hparams.yaml"), "r") as p:
            hparams = yaml.safe_load(p)

        key = "[" + f.split("_")[-1] + "] "
        if hparams["model"] == "unet":
            key += "UNet "
        elif hparams["model"] == "varnet":
            key += "VarNet "
        else:
            raise ValueError(
                f"Unrecognized reconstructor model {hparams['model']}."
            )
        key += "trained on "
        key += str((hparams["kspace_size"][-3], hparams["kspace_size"][-2]))
        key += " kspace data"

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
            "kspace_size": tuple(hparams["kspace_size"])
        })

    return models


def infer():
    args = Inference.build_args()

    models = load_models(args.lightning_logs_path)
    if 0 <= args.model_option < len(models):
        ckpt_path = models[args.model_option]["model"]
    else:
        print("Choose the reconstructor model to use:")
        for option in models:
            print(option["key"])
        print()
        model_option = int(input("Enter model number: "))
        ckpt_path = models[model_option]["model"]

    data_module = DataModule(
        args.data_path, train_dir=None, val_dir=None, test_dir=None
    )
    infer_model = ReconstructorModule.load_from_checkpoint(ckpt_path)
    if torch.cuda.is_available():
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            max_epochs=1
        )
    else:
        trainer = Trainer(
            enable_checkpointing=False, logger=False, max_epochs=1
        )
    predictions = trainer.predict(
        infer_model, dataloaders=data_module.predict_dataloader()
    )

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(predictions[15]["output"][0], cmap="gray")
    plt.savefig("test.png")
    print(predictions[15]["ssim"])


if __name__ == "__main__":
    infer()