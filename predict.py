# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import random
import subprocess
from pathlib import Path
import glob
from datetime import datetime
import time

from cog import BasePredictor, Input, Path

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
        # Ensure model cache directory exists
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Attempt custom tar download
        model_files = [
            "Wan2.1-T2V-14B.tar",
            "Wan2.1-T2V-1.3B.tar",
            "Wan2.1-I2V-14B-720P.tar",
            "Wan2.1-I2V-14B-480P.tar",
        ]
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Create model cache directory
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Define model paths
        self.model_paths = {
            "t2v-14B": f"{MODEL_CACHE}/Wan2.1-T2V-14B",
            "t2v-1.3B": f"{MODEL_CACHE}/Wan2.1-T2V-1.3B",
            "i2v-14B-720P": f"{MODEL_CACHE}/Wan2.1-I2V-14B-720P",
            "i2v-14B-480P": f"{MODEL_CACHE}/Wan2.1-I2V-14B-480P",
        }

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing what you want to generate"
        ),
        mode: str = Input(
            description="Generation mode: text-to-video, image-to-video, or text-to-image",
            choices=["text-to-video", "image-to-video", "text-to-image"],
            default="text-to-video",
        ),
        image: Path = Input(
            description="Input image for image-to-video generation (only required for image-to-video mode)",
            default=None,
        ),
        model_quality: str = Input(
            description="Model quality (higher quality is slower but better results)",
            choices=["standard", "fast"],
            default="fast",
        ),
        resolution: str = Input(
            description="Output resolution",
            choices=[
                "HD (1280×720)",
                "SD (832×480)",
                "Square (960×960)",
                "Portrait HD (720×1280)",
                "Widescreen (1088×832)",
                "Portrait (832×1088)",
                "Square HD (1024×1024)",
            ],
            default="SD (832×480)",
        ),
        seed: int = Input(
            description="Random seed for reproducible results (leave blank for random)",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # Set default values for removed advanced parameters
        memory_optimization = True
        sample_guide_scale = 6.0
        sample_shift = 8
        use_prompt_extension = False

        # Map user-friendly choices to internal values
        mode_map = {
            "text-to-video": "t2v",
            "image-to-video": "i2v",
            "text-to-image": "t2i",
        }

        resolution_map = {
            "HD (1280×720)": "1280*720",
            "SD (832×480)": "832*480",
            "Square (960×960)": "960*960",
            "Portrait HD (720×1280)": "720*1280",
            "Widescreen (1088×832)": "1088*832",
            "Portrait (832×1088)": "832*1088",
            "Square HD (1024×1024)": "1024*1024",
        }

        model_quality_map = {"standard": "14B", "fast": "1.3B"}

        # Convert user-friendly inputs to internal values
        internal_mode = mode_map[mode]
        internal_resolution = resolution_map[resolution]
        model_size = model_quality_map[model_quality]

        # Validate inputs
        if internal_mode == "i2v" and image is None:
            raise ValueError("An input image is required for image-to-video mode")

        # Set random seed if not provided
        actual_seed = seed if seed is not None else random.randint(0, 2147483647)

        # Determine the task based on mode and model size
        if internal_mode == "t2v":
            task = f"t2v-{model_size}"
            if model_size == "1.3B" and internal_resolution not in ["832*480"]:
                print(
                    "Note: Fast model (1.3B) is optimized for SD resolution. Switching to 832*480."
                )
                internal_resolution = "832*480"
            ckpt_dir = self.model_paths[task]
        elif internal_mode == "i2v":
            if internal_resolution in ["1280*720", "960*960", "720*1280"]:
                task = "i2v-14B-720P"
                ckpt_dir = self.model_paths[task]
            else:
                task = "i2v-14B-480P"
                ckpt_dir = self.model_paths[task]
        elif internal_mode == "t2i":
            task = f"t2i-{model_size}"
            ckpt_dir = self.model_paths[f"t2v-{model_size}"]  # T2I uses T2V model
        else:
            raise ValueError(f"Invalid mode: {internal_mode}")

        # Check if model exists
        if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
            # Try to download the model one more time
            print(f"Model for {task} not found or empty. Attempting to download...")
            os.makedirs(ckpt_dir, exist_ok=True)
            cmd = (
                f"huggingface-cli download Wan-AI/Wan2.1-{task} --local-dir {ckpt_dir}"
            )
            print(f"Running: {cmd}")
            os.system(cmd)

            # Check again after download attempt
            if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
                raise ValueError(
                    f"Model not found at {ckpt_dir} after download attempt. "
                    f"Please download it manually using: "
                    f"huggingface-cli download Wan-AI/Wan2.1-{task} --local-dir {ckpt_dir}"
                )

        # Build command for local generation
        cmd = ["python", "generate.py"]
        cmd.extend(["--task", task])
        cmd.extend(["--size", internal_resolution])
        cmd.extend(["--ckpt_dir", ckpt_dir])
        cmd.extend(["--prompt", prompt])
        cmd.extend(["--base_seed", str(actual_seed)])

        # Add image path for I2V
        if internal_mode == "i2v" and image is not None:
            cmd.extend(["--image", str(image)])

        # Add optimization flags
        if memory_optimization:
            cmd.append("--offload_model")
            cmd.append("True")
            cmd.append("--t5_cpu")

        # Add sample parameters for T2V-1.3B
        if task == "t2v-1.3B":
            cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
            cmd.extend(["--sample_shift", str(sample_shift)])
            # Add steps parameter for faster generation
            cmd.extend(["--sample_steps", "30"])

        # Execute the command
        print(f"Running command: {' '.join(cmd)}")

        # Use subprocess.call to run the command and show output in real-time
        result = subprocess.call(cmd)

        if result != 0:
            raise RuntimeError(f"Generation failed with exit code {result}")

        # Since we're not capturing output, we need to find the output file differently
        # We can use a predictable output pattern based on the task and timestamp

        # Get the current date in the format used by generate.py
        formatted_time = datetime.now().strftime("%Y%m%d")

        # Look for recently created output files
        if internal_mode in ["t2v", "i2v"]:
            pattern = f"{task}*{formatted_time}*.mp4"
            output_files = glob.glob(pattern)
        else:  # t2i
            pattern = f"{task}*{formatted_time}*.png"
            output_files = glob.glob(pattern)

        if not output_files:
            raise RuntimeError(
                f"Could not find generated output file matching pattern: {pattern}"
            )

        # Sort by modification time to get the most recent file
        output_path = sorted(output_files, key=os.path.getmtime)[-1]

        return Path(output_path)
