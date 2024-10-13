import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from config import get_experiment, DatasetName, DatasetName_train_icl, DatasetName_train_blender
from data import ImageDepthDataset
from unet import init_weights, count_parameters

# Set the environment variable for OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Select the device to use for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--CONFIG", "-c", type=str, required=True, help="Path to the configuration file.")
parser.add_argument("--SAVE_EVERY", "-e", type=int, default=10, required=False, help="Save checkpoint every N epochs.")
parser.add_argument("--DATASET", "-d", type=str, default="datasets", required=False, help="Path to the dataset directory.")
parser.add_argument("--SAVEDIR", "-s", type=str, default="checkpoints", required=False, help="Directory to save checkpoints.")
args = parser.parse_args(sys.argv[1:])

# Load the experiment configuration
DATASET_DIR = args.DATASET
EXPERIMENT = get_experiment(args.CONFIG)
experiment_name = EXPERIMENT.__class__.__name__.split("/")[-1]
CHECKPOINT_DIR = os.path.join(args.SAVEDIR, experiment_name)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load datasets
##NOTE: Scale factor must be changed depeding on the training and test data. Blender EXR scale factor remains 1.
test_datasets = {name: ImageDepthDataset(DATASET_DIR, name.name, codedDir="Codedphasecam-27Linear", cache=True, scale_factor=1000) for name in DatasetName}
nyu_datasets_train = {name: ImageDepthDataset(DATASET_DIR, name.name, codedDir="Codedphasecam-27Linear", cache=True, scale_factor=1000) for name in DatasetName_train_icl}
blender_datasets_train = {name: ImageDepthDataset(DATASET_DIR, name.name, codedDir="Codedphasecam-27Linear", cache=True, is_blender=True, scale_factor=1) for name in DatasetName_train_blender}

# Combine training datasets
train_datasets = [nyu_datasets_train[name] for name in EXPERIMENT.train if name in nyu_datasets_train] + \
                 [blender_datasets_train[name] for name in EXPERIMENT.train if name in blender_datasets_train]

# Create DataLoader for training
train_dataset = ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=EXPERIMENT.batch_size, shuffle=True, num_workers=4)

# Create DataLoader for testing
test_loaders = {name: DataLoader(dataset, batch_size=EXPERIMENT.batch_size, shuffle=True) for name, dataset in test_datasets.items()}

# Print the loaded testing datasets
for name in test_loaders.keys():
    print(f"Testing dataset: {name.name}")

# Initialize the model and optimizer
model = EXPERIMENT.model().to(device)
init_weights(model)
model_optimizer = torch.optim.Adam(model.parameters(), lr=EXPERIMENT.learning_rate)
L1 = nn.L1Loss()

# Initialize Weights and Biases for logging
wandb.init(project="coded_losses", name=experiment_name, config={**vars(args)})
print(repr(EXPERIMENT))
print(f"Training model with {count_parameters(model)} parameters")

def wandbimg(img: torch.Tensor, vmax=6.5):
    """
    Convert tensor image to WandB image format.

    Args:
        img (torch.Tensor): Input image tensor.
        vmax (float): Maximum value for clipping.

    Returns:
        wandb.Image: Image in WandB format.
    """
    out = np.clip(img.detach().cpu().numpy(), 0, vmax) / vmax * 255
    return wandb.Image(out.astype(np.uint8))

def evaluate(dataloader: DataLoader):
    """
    Evaluate the model on the given dataloader.

    Args:
        dataloader (DataLoader): DataLoader for evaluation.

    Returns:
        tuple: Average L1 error, L1 error for depth < 3m, ground truth depth map, reconstructed depth map.
    """
    model.eval()
    with torch.no_grad():
        metric_depth_error = 0
        metric_depth_error_under3 = 0
        sample_count = 0

        for batch in dataloader:
            recon = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
            metric_gt = batch["Depth"].to(device)
            metric_re = EXPERIMENT.depth.output_to_metric(recon)

            mask = metric_gt < 3
            if torch.any(mask).item():
                metric_depth_error_under3 += L1(metric_re[mask, 0], metric_gt[mask]).item() * len(batch)

            metric_depth_error += L1(metric_re[:, 0], metric_gt).item() * len(batch)
            sample_count += len(batch)

    model.train()
    ground_truth_depth_map = wandbimg(metric_gt[0])
    reconstructed_depth_map = wandbimg(metric_re[0, 0])

    return (
        metric_depth_error / sample_count,
        metric_depth_error_under3 / sample_count,
        ground_truth_depth_map,
        reconstructed_depth_map,
    )

# Training loop
NUM_TEST_SETS = len(DatasetName)
previous_test_error = float('inf')

for epoch in range(EXPERIMENT.epochs):
    start_time = time.monotonic()

    total_error = 0
    for batch in train_loader:
        model_optimizer.zero_grad()
        reconstruction = EXPERIMENT.post_forward(model(batch["Coded"].to(device)))
        metric_gt = batch["Depth"].to(device)
        ground_truth = EXPERIMENT.depth.metric_to_output(metric_gt)
        error = EXPERIMENT.compute_loss(ground_truth, reconstruction, epoch)
        error.backward()
        model_optimizer.step()
        total_error += error.item()

    # Evaluate the model on the test datasets
    test_artifacts = {}
    total_avg_l1 = 0
    total_u3_l1 = 0

    for name, dataloader in test_loaders.items():
        avg_l1, u3_l1, gt_depth_map, re_depth_map = evaluate(dataloader)
        test_artifacts[f"{name.name}: L1"] = avg_l1
        test_artifacts[f"{name.name}: L1 <3m"] = u3_l1

        if epoch % 5 == 0:
            test_artifacts[f"{name.name}: ground truth"] = gt_depth_map
            test_artifacts[f"{name.name}: reconstructed"] = re_depth_map

        if name not in EXPERIMENT.train:
            total_avg_l1 += avg_l1
            total_u3_l1 += u3_l1

    # Log metrics and save the best model
    iterate_values = {
        "train_error": total_error / len(train_loader),
        "test_L1": total_avg_l1 / NUM_TEST_SETS,
        "test_L1_under3": total_u3_l1 / NUM_TEST_SETS,
        **test_artifacts,
    }

    if total_u3_l1 / NUM_TEST_SETS < previous_test_error:
        previous_test_error = total_u3_l1 / NUM_TEST_SETS
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best.pt")

    wandb.log(iterate_values)

    if epoch % args.SAVE_EVERY == 0:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/recon_{epoch}.pt")

    end_time = time.monotonic()
    print(f"Epoch={epoch}: loss={total_error / len(train_loader)} :: {end_time - start_time:.3f}s")

# Save the final model
torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/recon_end.pt")
wandb.finish()
