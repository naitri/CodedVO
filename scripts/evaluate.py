import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_experiment, DatasetName_blender
from data import ImageDepthDataset

# Ensure OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Select device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Argument parser setup
parser = argparse.ArgumentParser(description="Evaluate depth prediction models.")
parser.add_argument("--CONFIG", "-c", type=str, required=True, help="Path to the configuration file.")
parser.add_argument("--DATASET", "-d", type=str, default="datasets", required=False, help="Path to the dataset directory.")
parser.add_argument("--CHECKPOINT", "-s", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--OUTPUT", "-o", type=str, required=True, help="Path to the output directory.")
args = parser.parse_args(sys.argv[1:])

# Load experiment configuration
DATASET_DIR = args.DATASET
EXPERIMENT = get_experiment(args.CONFIG)

# Load datasets
blender_datasets = {name: ImageDepthDataset(os.path.join(DATASET_DIR, name.name), codedDir = "Codedphasecam-27Linear",cache=True, is_blender=True, scale_factor=5000) for name in DatasetName_blender}
test_loaders = {name: DataLoader(dataset, batch_size=1, shuffle=False) for name, dataset in blender_datasets.items()}

# Initialize model
model = EXPERIMENT.model().to(device).eval()
model.load_state_dict(torch.load(args.CHECKPOINT, map_location=device))

L1 = nn.L1Loss()
print(repr(EXPERIMENT))

def to_numpy(img: torch.Tensor):
    return np.clip(img.detach().cpu().numpy(), 0, None)

def sigma_metric(estimated_depth, ground_truth_depth, threshold):
    ratio = torch.max(estimated_depth / ground_truth_depth, ground_truth_depth / estimated_depth)
    return torch.mean((ratio < threshold).float())

def evaluate(dataloader: DataLoader, output_dir):
    model.eval()
    with torch.no_grad():
        metrics = {
            "metric_depth_error": 0,
            "metric_depth_error_under3": 0,
            "abs_rel": 0,
            "sq_rel": 0,
            "rmse": 0,
            "rmse_log": 0,
            "sigma_1_25": 0,
            "sigma_1_25_2": 0,
            "sigma_1_25_3": 0,
            "sample_count": 0,
            "total_inference_time": 0
        }

        for idx, batch in enumerate(dataloader):
            start_time = time.time()
            recon = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
            end_time = time.time()
            batch_inference_time = end_time - start_time
            metrics["total_inference_time"] += batch_inference_time

            metric_gt = EXPERIMENT.depth.output_to_metric(batch["Depth"]).squeeze(1)
            metric_re = EXPERIMENT.depth.output_to_metric(recon).squeeze(1)
            valid_mask = metric_gt > 0
            gt = torch.clamp(metric_gt[valid_mask], 0, 6).to(device)
            pred = torch.clamp(metric_re[valid_mask], 0, 6)

            log_diff = torch.log(pred) - torch.log(gt)
            metrics["rmse_log"] += torch.sqrt(torch.mean(log_diff ** 2))

            mask = gt < 3
            if torch.any(mask).item():
                metrics["metric_depth_error_under3"] += L1(pred[mask], gt[mask]).item() * len(batch)

            metrics["abs_rel"] += torch.mean(torch.abs(pred - gt) / gt).item() * len(batch)
            metrics["sq_rel"] += torch.mean(((pred - gt) ** 2) / gt).item() * len(batch)
            metrics["rmse"] += torch.sqrt(torch.mean(((pred - gt) ** 2))).item() * len(batch)
            metrics["sigma_1_25"] += sigma_metric(pred, gt, 1.25) * len(batch)
            metrics["sigma_1_25_2"] += sigma_metric(pred, gt, 1.25 ** 2) * len(batch)
            metrics["sigma_1_25_3"] += sigma_metric(pred, gt, 1.25 ** 3) * len(batch)
            metrics["metric_depth_error"] += L1(pred, gt).item() * len(batch)
            metrics["sample_count"] += len(batch)

            # Save the predicted depth image
            prediction = (metric_re.squeeze(0).cpu().numpy() * 5000).astype(np.uint16)
            output_path_pred = os.path.join(output_dir, "pred_depth", f"{idx}.png")
            cv2.imwrite(output_path_pred, prediction)

        avg_inference_speed = metrics["total_inference_time"] / len(dataloader)
        avg_fps = 1 / avg_inference_speed

        print(f"Average Inference Speed: {avg_inference_speed:.4f} seconds per batch")
        print(f"Average FPS: {avg_fps:.2f} frames per second")

        avg_metrics = {k: v / metrics["sample_count"] for k, v in metrics.items() if k != "total_inference_time"}

    return avg_metrics

# Ensure output directories exist
output_dir = args.OUTPUT
os.makedirs(os.path.join(output_dir, "pred_depth"), exist_ok=True)

# Evaluate the model on each test dataset
for name, dataloader in test_loaders.items():
    metrics = evaluate(dataloader, output_dir)

    if name in EXPERIMENT.train:
        print(f"[train] {name.name}")
    else:
        print(f"{name.name}")

    print(f"| L1     : {metrics['metric_depth_error']:.3f}")
    print(f"| L1 <3m : {metrics['metric_depth_error_under3']:.3f}")
    print(f"| Mean Absolute Relative Error (abs_rel): {metrics['abs_rel']:.3f}")
    print(f"| Mean Squared Relative Error (sq_rel): {metrics['sq_rel']:.3f}")
    print(f"| Root Mean Squared Error (RMSE): {metrics['rmse']:.3f}")
    print(f"| RMSE Log: {metrics['rmse_log']:.3f}")
    print(f"| Sigma 1.25: {metrics['sigma_1_25']:.3f}")
    print(f"| Sigma 1.25^2: {metrics['sigma_1_25_2']:.3f}")
    print(f"| Sigma 1.25^3: {metrics['sigma_1_25_3']:.3f}")
    print()
