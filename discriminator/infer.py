"""
Inference driver program for kspace fidelity estimator.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from args import Inference
from fastmri.models.unet import Unet
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
import torch
from tqdm import tqdm
from typing import Optional, Union

from data.dataset import DiscriminatorDataset
from data.transform import DiscriminatorDataTransform
from pl_modules.discriminator_module import DiscriminatorModule

sys.path.append("..")
sys.path.append("../reconstructor")
import helper.utils.math as M
import helper.utils.transforms as T
from reconstructor.models.loss import structural_similarity


def save_inference(
    savedir: str,
    rimg: Union[torch.Tensor, np.ndarray],
    kspace_shape: Union[torch.Size, tuple],
    sampled_mask: torch.Tensor,
    acquiring_mask_preprocess: torch.Tensor,
    acquiring_mask_postprocess: torch.Tensor,
    fn: str,
    slice_idx: int,
    use_discriminator: bool = True,
    target: Optional[torch.Tensor] = None,
    max_value: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None
) -> None:
    """
    Saves kspace discriminator inference results.
    Input:
        savedir: output directory to save inference results to.
        rimg: real reconstructed image of shape HW.
        acquired_kspace: masked kspace that was already acquired of shape CHW2.
        sampling_kspace_preprocess: masked kspace that is currently being
            acquired of shape CHW2.
        sampling_kspace_postprocess: masked kspace that is currently being
            acquired after discriminator processing of shape CHW2.
        sampled_mask: acquired data mask of shape W.
        acquiring_mask_preprocess: current acquisition mask of shape W.
        acquiring_mask_postprocess: current acquisition mask of shape W after
            discriminator processing.
        fn: filename of raw data.
        slice_idx: slice index of image reconstruction in dataset.
        use_discriminator: specify whether the discriminator neural net was
            used for signal processing.
        target: target reconstruction image of shape HW.
        max_value: specified dynamic range for calculating SSIM.
        rotation: ground truth rotation angle.
    Returns:
        None.
    """
    if not isinstance(rimg, np.ndarray):
        rimg = rimg.detach().cpu().numpy()
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    if not os.path.isdir(os.path.join(savedir, os.path.splitext(fn)[0])):
        os.mkdir(os.path.join(savedir, os.path.splitext(fn)[0]))

    # Save the final image reconstruction.
    plt.gca()
    plt.imshow(rimg, cmap="gray")
    plt.axis("off")
    ssim = structural_similarity(
        torch.Tensor(rimg[np.newaxis, ...]).cpu(),
        torch.unsqueeze(target, dim=0).detach().cpu(),
        max_value
    )
    ssim = 0.0
    plt.title(f"SSIM: {ssim}")
    inference_path = os.path.join(
        savedir, os.path.splitext(fn)[0], f"{slice_idx}"
    )
    if use_discriminator:
        inference_path += "_discriminator.png"
    else:
        inference_path += ".png"
    plt.savefig(inference_path, bbox_inches="tight", dpi=600)

    # Save the target image.
    plt.gca()
    plt.cla()
    plt.imshow(target, cmap="gray")
    plt.axis("off")
    plt.title(f"Target Slice {slice_idx}, Theta = {rotation}")
    inference_path = os.path.join(
        savedir, os.path.splitext(fn)[0], f"{slice_idx}_target.png"
    )
    plt.savefig(inference_path, bbox_inches="tight", dpi=600)

    # Save the sampled kspace mask.
    _, h, _, _ = kspace_shape
    plt.cla()
    plt.gca()
    plt.imshow(
        torch.cat(
            (torch.unsqueeze(sampled_mask, dim=0),) * h, dim=0
        ).detach().cpu().numpy(),
        cmap="gray"
    )
    plt.axis("off")
    plt.title(
        f"{int(torch.sum(sampled_mask).item())} / {sampled_mask.size()[0]}"
    )
    sampled_mask_path = os.path.join(
        savedir, os.path.splitext(fn)[0], f"{slice_idx}_sampled_mask"
    )
    if use_discriminator:
        sampled_mask_path += "_discriminator.png"
    else:
        sampled_mask_path += ".png"
    plt.savefig(sampled_mask_path, bbox_inches="tight", dpi=600)

    # Save the acquiring kspace mask.
    plt.cla()
    plt.gca()
    plt.imshow(
        torch.cat(
            (torch.unsqueeze(acquiring_mask_preprocess, dim=0),) * h, dim=0
        ).detach().cpu().numpy(),
        cmap="spring"
    )
    plt.imshow(
        torch.cat(
            (torch.unsqueeze(acquiring_mask_postprocess, dim=0),) * h, dim=0
        ).detach().cpu().numpy(),
        cmap="gray", alpha=0.2
    )
    plt.axis("off")
    plot_title = f"{int(torch.sum(acquiring_mask_postprocess).item())} / "
    plot_title += f"{int(torch.sum(acquiring_mask_preprocess).item())}, "
    plot_title += f"{np.where(acquiring_mask_postprocess.cpu() > 0.0)}, "
    plot_title += f"{np.where(acquiring_mask_preprocess.cpu() > 0.0)}"
    plt.title(plot_title)
    sampling_mask_path = os.path.join(
        savedir, os.path.splitext(fn)[0], f"{slice_idx}_sampling_mask"
    )
    if use_discriminator:
        sampling_mask_path += "_discriminator.png"
    else:
        sampling_mask_path += ".png"
    plt.savefig(sampling_mask_path, bbox_inches="tight", dpi=600)

    plt.close()


def infer():
    args = Inference.build_args()

    seed = args.seed
    if seed is None or not isinstance(seed, int) or seed < 0:
        seed = int(time.time())
    torch.manual_seed(seed)
    transform = DiscriminatorDataTransform(
        coil_compression=args.coil_compression,
        seed=seed,
        p_transform=0.5,
        rotation=args.rotation,
        dx=[0.0, 0.0],
        dy=[0.0, 0.0],
        inference=True
    )
    dataset = DiscriminatorDataset(
        data_path=args.data_path,
        transform=transform,
        dataset_cache_file=args.cache_path,
        seed=seed,
        split="test",
        is_mlp=True
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    discriminator = None
    if args.model is not None and len(args.model) > 0:
        discriminator = DiscriminatorModule.load_from_checkpoint(
            args.model
        )
        discriminator = discriminator.to(device)
        discriminator.eval()
    reconstructor = None
    if args.reconstructor is not None and len(args.reconstructor) > 0:
        reconstructor = Unet(
            in_chans=1, out_chans=1, chans=256, num_pool_layers=4
        )
        reconstructor.load_state_dict(torch.load(args.reconstructor))
        reconstructor = reconstructor.to(device)
        reconstructor.eval()

    if args.histogram:
        pos_values, neg_values = None, None
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            # Use the compressed kspace dataset to calculate the heatmap.
            if discriminator is not None:
                heatmap = discriminator(
                    torch.unsqueeze(
                        T.apply_mask(item.ref_kspace, item.sampled_mask),
                        dim=0,
                    ).to(device),
                    torch.unsqueeze(
                        T.apply_mask(
                            item.distorted_kspace, item.acquiring_mask
                        ),
                        dim=0,
                    ).to(device),
                    torch.unsqueeze(item.sampled_mask, dim=0).to(device),
                    torch.unsqueeze(item.acquiring_mask, dim=0).to(device)
                )
                heatmap = torch.squeeze(
                    torch.mean(torch.sigmoid(heatmap), dim=-2), dim=0
                )
                if args.histogram:
                    ground_truth = 1.0 - torch.isclose(
                        item.ref_kspace, item.distorted_kspace
                    ).type(torch.float32).to(device)
                    # Sum over the coil and real/imaginary axes.
                    ground_truth = torch.clamp(
                        torch.sum(ground_truth, dim=[1, -1]), 0.0, 1.0
                    )
                    ground_truth = torch.squeeze(ground_truth, dim=0)
                    idxs = torch.nonzero(
                        item.acquiring_mask > 0.0, as_tuple=True
                    )

                    pos_locs = torch.where(ground_truth[idxs[0]] > 0.0)[0]
                    neg_locs = torch.where(ground_truth[idxs[0]] < 1.0)[0]
                    if pos_values is None:
                        pos_values = heatmap[idxs[0]][pos_locs].to(device)
                    else:
                        pos_values = torch.cat(
                            (pos_values, heatmap[idxs[0]][pos_locs])
                        )
                    if neg_values is None:
                        neg_values = heatmap[idxs[0]][neg_locs].to(device)
                    else:
                        neg_values = torch.cat(
                            (neg_values, heatmap[idxs[0]][neg_locs])
                        )
            else:
                heatmap = torch.sigmoid(
                    torch.zeros(item.ref_kspace.size()[-2]).to(device)
                )
            filtered_acq_mask = torch.logical_and(
                item.acquiring_mask.type(torch.bool).to(device),
                heatmap <= max(0.0, min(1.0, args.threshmin))
            )
            # Do the final reconstruction with the kspace prior to coil
            # compression.
            processed_kspace = torch.add(
                T.apply_mask(
                    item.uncompressed_ref_kspace.to(device),
                    item.sampled_mask.to(device)
                ),
                T.apply_mask(
                    item.uncompressed_distorted_kspace.to(device),
                    filtered_acq_mask.to(device)
                )
            )

            if reconstructor is None:
                rimg = M.rss(M.complex_abs(M.ifft2c(processed_kspace)))
                rimg = T.center_crop(rimg, args.center_crop)
            else:
                rimg = M.rss(M.complex_abs(M.ifft2c(processed_kspace)))
                rimg = T.center_crop(
                    torch.unsqueeze(torch.unsqueeze(rimg, dim=0), dim=0),
                    args.center_crop
                )
                mean_rimg, std_rimg = torch.mean(rimg), torch.std(rimg)
                norm_rimg = torch.divide(rimg - mean_rimg, std_rimg)
                # Introduce a batch dimension to arguments as well.
                rimg = reconstructor(norm_rimg)
                mean_rimg = torch.unsqueeze(mean_rimg, dim=-1)
                std_rimg = torch.unsqueeze(std_rimg, dim=-1)
                rimg = (std_rimg * rimg) + mean_rimg

            if args.save_path:
                save_inference(
                    args.save_path,
                    torch.squeeze(rimg),
                    item.uncompressed_ref_kspace.size(),
                    torch.squeeze(item.sampled_mask),
                    torch.squeeze(item.acquiring_mask),
                    torch.squeeze(filtered_acq_mask),
                    item.fn,
                    item.slice_idx,
                    use_discriminator=(discriminator is not None),
                    target=torch.squeeze(
                        T.center_crop(item.target, args.center_crop), dim=0
                    ),
                    max_value=torch.Tensor([item.max_value]),
                    rotation=torch.Tensor([item.theta])
                )
    if args.save_path:
        preds_data = {}
        preds_data["distorted"] = pos_values.detach().cpu().numpy().tolist()
        preds_data["undistorted"] = neg_values.detach().cpu().numpy().tolist()
        with open(os.path.join(args.save_path, "heatmap.pkl"), "w+b") as f:
            pickle.dump(preds_data, f)
        print(
            f"Saved dataset to {os.path.join(args.save_path, 'heatmap.pkl')}"
        )


if __name__ == "__main__":
    infer()
