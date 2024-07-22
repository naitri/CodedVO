import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDepthDataset(Dataset):
    def __init__(self, path: str, codedDir: str = "Coded", cache: bool = True, is_blender: bool = False, image_size=(480, 640), scale_factor: float = 5000):
        """
        Initialize the dataset.

        Args:
            path (str): Path to the dataset.
            codedDir (str): Directory containing coded images.
            cache (bool): Whether to cache the dataset in memory.
            is_blender (bool): Whether the dataset is in Blender's EXR format.
            image_size (tuple): Size of the images (height, width).
            scale_factor (float): Scale factor for depth normalization.
        """
        self.path = path
        self.is_blender = is_blender
        self.transform = transforms.Compose([transforms.CenterCrop(image_size)])
        self.data = []
        self.scale_factor = scale_factor

        # Directory path for the coded images
        dir_path = os.path.join(path, codedDir)
        
        # Get list of PNG files in the coded directory
        files = sorted([p for p in os.listdir(dir_path) if p.endswith(".png")])
        
        # Process each file
        for file in files:
            coded_file = os.path.join(path, codedDir, file)
            depth_file = os.path.join(path, "depth", file.replace(".png", ".exr") if is_blender else file)

            # Cache the processed data or store file paths
            if cache:
                self.data.append(self.process(coded_file, depth_file))
            else:
                self.data.append((coded_file, depth_file))

        self.cache = cache
        self.len = len(self.data)

    def process(self, coded_file: str, depth_file: str):
        """
        Process a single pair of coded and depth images.

        Args:
            coded_file (str): Path to the coded image file.
            depth_file (str): Path to the depth image file.

        Returns:
            dict: Processed images in a dictionary.
        """
        # Read the coded image and convert to a tensor
        coded = torch.from_numpy(cv2.imread(coded_file)).moveaxis(-1, 0)
        
        # Read the depth image and convert to a tensor
        if self.is_blender:
            raw_depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            metric_depth = torch.from_numpy(raw_depth[:, :, 0])
        else:
            metric_depth = torch.from_numpy(cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / self.scale_factor)

        return {
            "Coded": self.transform(coded.to(torch.float32)) / 255.0,
            "Depth": self.transform(metric_depth.to(torch.float32)),
        }

    def __len__(self):
        # Return the length of the dataset
        return self.len

    def __getitem__(self, idx):
        # Return the processed data from cache or process on the fly
        if self.cache:
            return self.data[idx]
        else:
            return self.process(*self.data[idx])

    def __repr__(self):
        # Representation of the dataset
        dataset_type = "Blender" if self.is_blender else "ICL"
        return f"{dataset_type}Dataset(path={self.path}, n={self.len}, scale_factor={self.scale_factor})"

# Example usage
'''
if __name__ == "__main__":
    import argparse

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Load and process image-depth dataset.")
    parser.add_argument("--base_dir", type=str, default="./datasets", help="Base directory of the dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the specific dataset.")
    parser.add_argument("--is_blender", action="store_true", help="Specify if using Blender's EXR format.")
    parser.add_argument("--cache", action="store_true", help="Cache the dataset in memory.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(480, 640), help="Size of the images (height, width).")
    parser.add_argument("--scale_factor", type=float, default=5000, help="Scale factor for depth normalization.")
    
    args = parser.parse_args()

    # Create the dataset instance
    dataset = ImageDepthDataset(
        path=os.path.join(args.base_dir, args.dataset_path),
        cache=args.cache,
        is_blender=args.is_blender,
        image_size=tuple(args.image_size),
        scale_factor=args.scale_factor
    )
    print(dataset)
'''