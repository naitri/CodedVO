import os
import cv2
import numpy as np
import torch
import tqdm
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_numpy_image(tensor: torch.Tensor):
    """
    Convert a PyTorch tensor to a NumPy image.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W).
        
    Returns:
        np.ndarray: Output image in NumPy format.
    """
    out = tensor.moveaxis(0, -1)
    return torch.clamp(out, 0, 255).to(torch.uint8).cpu().numpy()

class Camera:
    def __init__(self, name: str, psfs: torch.Tensor, depth_layers: torch.Tensor, use_nonlinear: bool = False):
        """
        Initialize the Camera object.
        
        Args:
            name (str): Name of the camera.
            psfs (torch.Tensor): Point spread functions (PSFs) tensor.
            depth_layers (torch.Tensor): Depth layers tensor.
            use_nonlinear (bool): Whether to use non-linear processing.
        """
        self.name = f"Coded{name}NonLinear" if use_nonlinear else f"Coded{name}Linear"
        n_depths, n_channels, height, width = psfs.shape

        if n_channels != 3 or n_depths != depth_layers.numel() or width != height or width % 2 == 0:
            raise ValueError(f"PSF has wrong shape: {psfs.shape}")

        self.psfs = psfs.to(device)
        self.depth_layers = depth_layers.to(device)
        self.use_nonlinear = use_nonlinear
        self.padding = width // 2

    def capture(self, img: np.ndarray, metric_depth: np.ndarray) -> np.ndarray:
        """
        Capture an image with depth-dependent processing.
        
        Args:
            img (np.ndarray): Input RGB image.
            metric_depth (np.ndarray): Metric depth map.
            
        Returns:
            np.ndarray: Processed image.
        """
        image = torch.from_numpy(img).moveaxis(-1, 0).to(torch.float32).to(device)
        depth = torch.from_numpy(metric_depth).to(device)

        if self.use_nonlinear:
            coded = self.nonlinear(image, depth)
        else:
            coded = self.linear(image, depth)
        return tensor_to_numpy_image(coded)

    def get_depth_layers(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Get depth layers from a depth map.
        
        Args:
            depth_map (torch.Tensor): Input depth map.
            
        Returns:
            torch.Tensor: Quantized depth layers.
        """
        quantized_depth = torch.bucketize(depth_map, self.depth_layers)
        return torch.stack([quantized_depth == j for j in range(len(self.depth_layers))])

    def linear(self, image: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Perform linear depth-dependent convolution.
        
        Args:
            image (torch.Tensor): Input image.
            depth_map (torch.Tensor): Depth map.
            
        Returns:
            torch.Tensor: Convolved image.
        """
        depth_mask = self.get_depth_layers(depth_map)

        return torch.stack(
            [
                torch.sum(
                    torch.nn.functional.conv2d(
                        image[None, channel:channel+1],
                        self.psfs[:, channel:channel+1],
                        stride=1,
                        padding=self.padding,
                    ) * depth_mask, dim=1
                )
                for channel in range(3)
            ], dim=1
        )[0]

    def single_psf_convolution(self, image: torch.Tensor, depth_idx: int, channel_idx: int) -> torch.Tensor:
        """
        Convolve image with a single PSF.
        
        Args:
            image (torch.Tensor): Input image.
            depth_idx (int): Depth index.
            channel_idx (int): Channel index.
            
        Returns:
            torch.Tensor: Convolved image.
        """
        return torch.nn.functional.conv2d(
            image,
            self.psfs[depth_idx:depth_idx+1, channel_idx:channel_idx+1],
            stride=1,
            padding=self.padding,
        )

    def nonlinear(self, img: torch.Tensor, depth_map: torch.Tensor, eps=1e-6) -> torch.Tensor:
        """
        Perform non-linear blurring based on Ikoma et al. 2021 equation 5.
        
        Args:
            img (torch.Tensor): Input image.
            depth_map (torch.Tensor): Depth map.
            eps (float): Small epsilon value to prevent division by zero.
            
        Returns:
            torch.Tensor: Blurred image.
        """
        depth_mask = self.get_depth_layers(depth_map)
        K, _, _ = depth_mask.shape
        depth_mask = depth_mask.to(torch.float)
        depth_mask = torch.flip(depth_mask, dims=(0,))

        out = torch.zeros_like(img)
        img = img.to(torch.float) / 255.0
        depth_sum = torch.cumsum(depth_mask, dim=0)

        for channel in range(3):
            layered = img[channel:channel+1] * depth_mask
            for k in range(K):
                E_k = self.single_psf_convolution(depth_sum[k][None, None], k, channel)
                l_k = self.single_psf_convolution(layered[k][None, None], k, channel) / (E_k + eps)
                for kp in range(k + 1, K):
                    E_kp = self.single_psf_convolution(depth_sum[kp][None, None], kp, channel)
                    a_kp = 1 - self.single_psf_convolution(depth_mask[kp][None, None], kp, channel) / (E_kp + eps)
                    l_k = l_k * a_kp
                out[channel] = out[channel] + l_k

        return torch.clamp(out * 255, 0, 255)

    def process_folder(self, root: str, is_blender: bool = False, scale_factor: float = 5000):
        """
        Process a folder of images and depths.
        
        Args:
            root (str): Root directory.
            is_blender (bool): Whether the depth images are in Blender's EXR format.
            scale_factor (float): Scale factor for depth normalization.
        """
        depth_folder = os.path.join(root, "depth")
        image_folder = os.path.join(root, "rgb")
        output_folder = os.path.join(root, self.name)
        os.makedirs(output_folder, exist_ok=True)
        
        files = os.listdir(image_folder)
        max_depth_value = 0

        for idx, file in tqdm.tqdm(enumerate(files), total=len(files), desc=root):
            
            image_file = os.path.join(image_folder, file)
            depth_file = os.path.join(depth_folder, file).replace(".png", ".exr") if is_blender else os.path.join(depth_folder, file)

            image_bgr = cv2.imread(image_file)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            raw_depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if raw_depth is None:
                print(f"{file} is missing a depth file")
                continue

            metric_depth = raw_depth[:, :, 0] if is_blender else raw_depth / scale_factor

            coded_image_rgb = self.capture(image.astype(np.float32), metric_depth)
            coded_image_bgr = cv2.cvtColor(coded_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder, file), coded_image_bgr)
            max_depth_value = max(max_depth_value, np.max(metric_depth))
        
        print(f"Max Depth Value in the folder: {max_depth_value}")

def main():
    parser = argparse.ArgumentParser(description="Process images with depth-dependent processing.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing the datasets.")
    parser.add_argument("--is_blender", action="store_true", help="Use Blender's EXR depth format.")
    parser.add_argument("--scale_factor", type=float, default=5000, help="Scale factor for depth normalization.")
    args = parser.parse_args()

    # Fixed paths and parameters
    psf_path = os.path.join("..", "phasecam-psfs-27.npy")
    depth_layers = torch.linspace(0.5, 6, 27)

    camera = Camera(
        "phasecam-27",
        torch.from_numpy(np.moveaxis(np.load(psf_path), -1, 1)),
        depth_layers,
        not args.is_blender,
    )

    camera.process_folder(
        args.root,
        is_blender=args.is_blender,
        scale_factor=args.scale_factor,
    )

if __name__ == "__main__":
    main()
