# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import random
import subprocess
import glob
from datetime import datetime
import time
import torch
import logging
import sys
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

from cog import BasePredictor, Input, Path, current_scope

# Import necessary modules from the wan package
import wan
from wan.utils.utils import cache_video
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

# Import our distributed setup module
import distributed_setup

MODEL_CACHE = "./model_cache"
BASE_URL = f"https://weights.replicate.delivery/default/wan2.1/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. "
            f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
    print("[!] Download took:", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
        
        # Ensure model cache directory exists
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Detect number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f"[INFO] Detected {self.num_gpus} available GPU(s)")

        # Download only the 1.3B T2V model
        model_files = ["Wan2.1-T2V-1.3B.tar"]
        
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Define the 1.3B model path
        self.model_path = f"{MODEL_CACHE}/Wan2.1-T2V-1.3B"
        
        # Always using t2v-1.3B task
        self.task = "t2v-1.3B"
        
        # Check if model exists
        if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
            # Try to download the model one more time
            print(f"Model for {self.task} not found or empty. Attempting to download...")
            os.makedirs(self.model_path, exist_ok=True)
            cmd = (
                f"huggingface-cli download Wan-AI/Wan2.1-{self.task} --local-dir {self.model_path}"
            )
            print(f"Running: {cmd}")
            os.system(cmd)

            # Check again after download attempt
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                raise ValueError(
                    f"Model not found at {self.model_path} after download attempt. "
                    f"Please download it manually using: "
                    f"huggingface-cli download Wan-AI/Wan2.1-{self.task} --local-dir {self.model_path}"
                )
        
        # Set up distributed parameters
        self.ulysses_size = 1
        self.ring_size = self.num_gpus
        
        print(f"[INFO] Setting up distributed model across {self.num_gpus} GPUs...")
        
        # Initialize and load the model in distributed mode
        # This launches processes for each GPU that stay alive
        self.processes = distributed_setup.setup_distributed_model(
            model_path=self.model_path,
            task=self.task,
            num_gpus=self.num_gpus
        )
        
        print(f"[INFO] Model successfully loaded across {self.num_gpus} GPUs")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing what you want to generate"
        ),
        aspect_ratio: str = Input(
            description="Video aspect ratio",
            choices=[
                "16:9",
                "9:16",
            ],
            default="16:9",
        ),
        frame_num: int = Input(
            description="Video duration in frames (based on standard 16fps playback)",
            choices=[17, 33, 49, 65, 81],  # Corresponds to ~1-5 seconds
            default=81,
        ),
        resolution: str = Input(
            description="Video resolution",
            choices=["480p"],
            default="480p",
        ),
        sample_steps: int = Input(
            description="Number of sampling steps (higher = better quality but slower)",
            default=30,  # Default to 30 steps
            ge=10,  # Minimum value of 10 steps
            le=50,  # Maximum value of 50 steps
        ),
        sample_guide_scale: float = Input(
            description="Classifier free guidance scale (higher values strengthen prompt adherence)",
            default=6.0,  # Recommended value for 1.3B model
            ge=0.0,  # Minimum value
            le=20.0,  # Maximum value
        ),
        sample_shift: float = Input(
            description="Sampling shift factor for flow matching (recommended range: 8-12)",
            default=8.0,  # Lower bound of recommended range for 1.3B model
            ge=0.0,  # Minimum value
            le=20.0,  # Maximum value
        ),
        seed: int = Input(
            description="Random seed for reproducible results (leave blank for random)",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Map aspect ratio choices to internal resolution format
        aspect_ratio_map = {
            "16:9": "832*480",
            "9:16": "480*832",
        }
        
        # Convert selected aspect ratio to internal resolution format
        internal_resolution = aspect_ratio_map[aspect_ratio]
        
        # Parse resolution into width x height
        width, height = SIZE_CONFIGS[internal_resolution]
        
        # Using only T2V mode with 1.3B model
        print(f"[INFO] Using text-to-video mode with 1.3B model at {aspect_ratio} aspect ratio, {resolution} resolution")
        print(f"[INFO] Generating {frame_num} frames (approximately {frame_num/16:.1f} seconds at 16fps)")
        print(f"[INFO] Using {self.num_gpus} GPUs for distributed inference")
        print(f"[INFO] Distribution strategy: ring_size={self.num_gpus}")

        # Set random seed if not provided
        actual_seed = seed if seed is not None else random.randint(0, 2147483647)
        
        # Get the current date/time for the output filename
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize prompt for filename
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:30]
        # Replace asterisk with 'x' for resolution
        safe_size = internal_resolution.replace('*', 'x')
        # Create output filename
        output_filename = f"{self.task}_{safe_size}_{formatted_prompt}_{formatted_time}.mp4"
        
        print(f"[INFO] Starting video generation directly using distributed model...")
        
        # Generate the video using our already loaded distributed model
        start_time = time.time()
        
        video = distributed_setup.generate_with_model(
            prompt=prompt,
            size=(width, height),
            frame_num=frame_num,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            shift=sample_shift,
            seed=actual_seed
        )
        
        generation_time = time.time() - start_time
        print(f"[INFO] Video generation completed in {generation_time:.2f} seconds")
        
        if video is None:
            raise RuntimeError("Video generation failed, no output produced")
        
        # Save the video to disk with the correct parameters
        output_path = cache_video(
            tensor=video[None],
            save_file=output_filename,
            fps=16,  # WAN_CONFIGS[self.task].sample_fps would be more precise
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        print(f"[INFO] Generated output saved to: {output_path}")
        
        # Make absolutely sure the filename is URL-safe by removing any potentially problematic characters
        safe_basename = ''.join(c for c in output_path if c.isalnum() or c in '._-')
        if safe_basename != output_path:
            safe_output_path = os.path.join(os.path.dirname(output_path), safe_basename)
            print(f"[INFO] Renaming output to ensure URL-safe filename: {safe_output_path}")
            os.rename(output_path, safe_output_path)
            output_path = safe_output_path
        
        # Record metric
        current_scope().record_metric("video_output_count", 1)
        
        # Return the path to the generated video
        return Path(output_path)

