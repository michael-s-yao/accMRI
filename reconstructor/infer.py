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
from typing import List

from pl_modules.data_module import DataModule
from pl_modules.reconstructor_module import ReconstructorModule


def infer():
    args = Inference.build_args()

    data_module = DataModule(
        args.data_path,
        cache_path=args.cache_path,
        train_dir=None,
        val_dir=None,
        test_dir=None,
        center_crop=args.center_crop,
        num_workers=0,
        fixed_acceleration=args.fixed_acceleration,
        tl=args.simulated
    )
    if args.model is None:
        infer_model = ReconstructorModule(
            model="varnet",
            is_multicoil=True,
            use_zero_filled=args.use_zero_filled
        )
    else:
        infer_model = ReconstructorModule.load_from_checkpoint(args.model)

    infer_model.reconstructor = infer_model.reconstructor.eval()
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False,
        logger=False,
        max_epochs=1, progress_bar_refresh_rate=0
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
