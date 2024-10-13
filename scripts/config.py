from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Type

import torch
from torch import nn

from unet import U_Net

L1 = nn.L1Loss()
L2 = nn.MSELoss()

# Custom Loss Functions 
def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)

def weighted_depth_loss(input: torch.Tensor, target: torch.Tensor, weight: float):
    diff = torch.abs(input - target) 
    weights = torch.where(target <= 3, torch.exp(weight * (3 - target)), torch.exp(-weight * (target - 3)))
    loss = weights * diff
    return loss.mean()

def weighted_l1_loss(input: torch.Tensor, target: torch.Tensor):
    mask = target <= 3.0
    weights = torch.where(mask, torch.ones_like(target), (3.0 / target))
    diff = L1(input, target)
    loss = weights * diff
    return loss.mean()

def silog_loss(input: torch.Tensor, target: torch.Tensor, variance_focus: float = 0.85):
    # Only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (input > 0)
    d = torch.log(input[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0


# Enum definitions for datasets 
## NOTE: Edit this as per dataset usage. For e.g for testing on ICL-NUIM office trajectory 1 edit DatasetName = Enum("DatasetName", ["office_traj1"]) 
## and change scale_factor accordingly

DepthStyle = Enum("DepthStyle", ["metric", "phi", "normalized"])
DatasetName = Enum("DatasetName", ["nyu_data"])
DatasetName_blender = Enum("DatasetName", ["DiningRoom"])
DatasetName_train_icl = Enum("DatasetName", ["nyu_data"])
DatasetName_train_blender = Enum("DatasetName", ["LivingRoom1"])

# Metric units for convenience
cm = 1e-2
mm = 1e-3

# Abstract base class for depth space representation
class DepthSpace(ABC):
    """Convert metric depth maps to and from model output maps"""

    @abstractmethod
    def output_to_metric(self, out: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def metric_to_output(self, met: torch.Tensor) -> torch.Tensor:
        ...

    def __repr__(self):
        return self.__class__.__name__

# Depth representation classes
class MetricDepth(DepthSpace):
    """Basic metric depth representation"""
    def output_to_metric(self, out: torch.Tensor):
        return out

    def metric_to_output(self, met: torch.Tensor):
        return met

class NormalizedDepth(DepthSpace):
    """Represent depth as 0-1 based on a maximum distance."""
    def __init__(self, max_value: float = 7):
        self.max_value = max_value

    def output_to_metric(self, out: torch.Tensor):
        return out * self.max_value

    def metric_to_output(self, met: torch.Tensor):
        return met / self.max_value

    def __repr__(self):
        return f"{self.__class__.__name__}(max_value={self.max_value})"

class DiopterDepth(DepthSpace):
    """Represent depth as defocus terms away from the focal plane."""
    def __init__(self, f_number: float = 17, focal_length: float = 50 * mm, focus_distance: float = 85 * cm, max_diopter: int = 15):
        self.f_number = f_number
        self.focal_length = focal_length
        self.R = focal_length / (2 * f_number)
        self.focus_distance = focus_distance
        self.k = 2 * torch.pi / 530e-9
        self.max_diopter = max_diopter

    def output_to_metric(self, out: torch.Tensor):
        out2 = out * self.max_diopter * 2 - self.max_diopter
        Wm = out2 / self.k
        depth = 1 / (1 / self.focus_distance + 2 * Wm / self.R**2)
        return depth

    def metric_to_output(self, met: torch.Tensor):
        depth = torch.clamp(met, 1 * mm, None)  # prevent 0 depth
        inv = 1 / depth
        sub = inv - 1 / self.focus_distance
        div = sub / 2
        Wm = div * self.R**2
        Phi2 = Wm * self.k
        return (Phi2 + self.max_diopter) / (2 * self.max_diopter)

    def __repr__(self):
        return f"{self.__class__.__name__}(f_number={self.f_number}, focal_length={self.focal_length*100:.1f}cm, focus_distance={self.focus_distance*100:.1f}cm, max_diopter={self.max_diopter})"

# Experiment class to handle different models and datasets
def get_experiment(class_name: str):
    return [cls for cls in Experiment.__subclasses__() if cls.__name__ == class_name and cls != Experiment][0]()

class Experiment(ABC):
    name: str
    model: Type[nn.Module]
    depth: DepthSpace
    epochs: int
    batch_size: int
    learning_rate: float
    train: List[DatasetName]
    coded: str

    @abstractmethod
    def compute_loss(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0):
        ...

    @abstractmethod
    def post_forward(self, reconstruction: torch.Tensor):
        ...

    def __repr__(self):
        out = f"{self.__class__.__name__}(\n"
        out += f"\tmodel={self.model.__name__}\n"
        out += f"\tdepth={self.depth!r}\n"
        out += f"\tLR={self.learning_rate}\n"
        out += f"\tepochs={self.epochs}\n"
        out += f"\tbatch-size={self.batch_size}\n"
        out += f"\ttrain-set={[item.name for item in self.train]}\n"
        out += f"\tcoded={self.coded!r}\n"
        out += ")"
        return out


class SimpleDiopter(Experiment):
    model = U_Net
    depth = DiopterDepth(max_diopter=13)
    epochs = 200
    batch_size = 8
    learning_rate = 1e-4
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(reconstruction)

    def compute_loss(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0) -> torch.Tensor:
        return L2(reconstruction[:, 0], ground_truth)

class L1LossBlenderNYU(Experiment):
    model = U_Net
    depth = MetricDepth()
    epochs = 80
    batch_size = 3
    learning_rate = 1e-4
    train = [DatasetName_train_icl.nyu_data, DatasetName_train_blender.LivingRoom1]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0):
        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)
        return L1(reconstruction_metric[:, 0], ground_truth_metric)

class MetricWeightedLossBlenderNYU(Experiment):
    model = U_Net
    depth = MetricDepth()
    epochs = 80
    batch_size = 3
    learning_rate = 1e-4
    train = [DatasetName_train_blender.LivingRoom1, DatasetName_train_icl.nyu_data]
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0):
        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)
        return weighted_mse_loss(reconstruction_metric[:, 0], ground_truth_metric, 2 ** (-0.3 * ground_truth_metric))

# Example class using SILog loss
class SILossLiving_Office(Experiment):
    model = U_Net
    depth = MetricDepth()
    epochs = 80
    batch_size = 1
    learning_rate = 1e-4
    coded = "Codedphasecam-27Linear"

    def post_forward(self, reconstruction: torch.Tensor):
        return reconstruction

    def compute_loss(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, idx: int = 0):
        reconstruction_metric = self.depth.output_to_metric(reconstruction)
        ground_truth_metric = self.depth.output_to_metric(ground_truth)
        return silog_loss(reconstruction_metric[:, 0], ground_truth_metric)

