# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import random
import subprocess
import glob
from datetime import datetime
import time
import torch

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

        # Detect number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f"[INFO] Detected {self.num_gpus} available GPU(s)")

        # List of all model files - will be downloaded on demand based on selected model
        self.model_files = {
            "t2v-1.3B": "Wan2.1-T2V-1.3B.tar",
            "t2v-14B": "Wan2.1-T2V-14B.tar",
            "i2v-14B-720P": "Wan2.1-I2V-14B-720P.tar",
            "i2v-14B-480P": "Wan2.1-I2V-14B-480P.tar",
        }

        # Create model cache directory
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Define model paths for all variants
        self.model_paths = {
            "t2v-1.3B": f"{MODEL_CACHE}/Wan2.1-T2V-1.3B",
            "t2v-14B": f"{MODEL_CACHE}/Wan2.1-T2V-14B",
            "i2v-14B-720P": f"{MODEL_CACHE}/Wan2.1-I2V-14B-720P",
            "i2v-14B-480P": f"{MODEL_CACHE}/Wan2.1-I2V-14B-480P",
        }

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing what you want to generate"
        ),
        image: Path = Input(
            description="Input image for image-to-video generation (optional, if provided will use image-to-video mode)",
            default=None,
        ),
        model_quality: str = Input(
            description="Model quality - larger model has better quality but requires more GPU memory",
            choices=["Standard (1.3B)", "High Quality (14B)"],
            default="Standard (1.3B)",
        ),
        resolution: str = Input(
            description="Output video resolution (some options only available with specific models)",
            choices=[
                # T2V-1.3B resolutions
                "Landscape SD (832×480)",
                "Portrait SD (480×832)",
                "Square SD (624×624)",
                "Wide SD (704×544)",
                "Tall SD (544×704)",
                # T2V/I2V-14B resolutions
                "Landscape HD (1280×720)",
                "Portrait HD (720×1280)",
                "Widescreen HD (1088×832)",
                "Portrait HD (832×1088)",
                "Square HD (960×960)",
                "Square HD+ (1024×1024)",
            ],
            default="Landscape SD (832×480)",
        ),
        sample_steps: int = Input(
            description="Number of sampling steps (higher = better quality but slower)",
            default=50,  # Default for T2V is 50, for I2V is 40
            ge=10,
            le=50,
        ),
        sample_guide_scale: float = Input(
            description="Classifier free guidance scale (higher values strengthen prompt adherence)",
            default=6.0,  # Will be adjusted based on model
            ge=0.0,
            le=20.0,
        ),
        sample_shift: float = Input(
            description="Sampling shift factor for flow matching",
            default=8.0,  # Will be adjusted based on model
            ge=0.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Random seed for reproducible results (leave blank for random)",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # Default optimization settings
        memory_optimization = False

        # Parse model quality selection
        model_size = "14B" if "High Quality" in model_quality else "1.3B"
        
        # Map resolution choices to internal format
        resolution_map = {
            # T2V-1.3B resolutions
            "Landscape SD (832×480)": "832*480",
            "Portrait SD (480×832)": "480*832",
            "Square SD (624×624)": "624*624",
            "Wide SD (704×544)": "704*544",
            "Tall SD (544×704)": "544*704",
            # T2V/I2V-14B resolutions
            "Landscape HD (1280×720)": "1280*720",
            "Portrait HD (720×1280)": "720*1280",
            "Widescreen HD (1088×832)": "1088*832",
            "Portrait HD (832×1088)": "832*1088",
            "Square HD (960×960)": "960*960",
            "Square HD+ (1024×1024)": "1024*1024",
        }
        
        # Convert selected resolution to internal format
        internal_resolution = resolution_map[resolution]
        
        # Check if the resolution is compatible with the selected model size
        hd_resolutions = ["1280*720", "720*1280", "1088*832", "832*1088", "960*960", "1024*1024"]
        
        # If 1.3B model is selected, enforce SD resolution
        if model_size == "1.3B" and internal_resolution in hd_resolutions:
            print(f"[INFO] HD resolution {resolution} is not supported with 1.3B model. Switching to 'Landscape SD (832×480)'")
            internal_resolution = "832*480"
        
        # Determine mode based on presence of image
        if image is not None:
            # If image is provided, use I2V mode - only available with 14B model
            if model_size == "1.3B":
                print("[INFO] Image-to-video is only supported with 14B model. Switching to 14B model.")
                model_size = "14B"
            
            internal_mode = "i2v"
            print(f"[INFO] Using image-to-video mode with {model_size} model at resolution {resolution}")
            
            # Default I2V parameters
            if sample_steps == 50:  # If using default, adjust for I2V
                sample_steps = 40  # I2V works better with 40 steps
            
            # Default I2V sampling parameters
            if sample_guide_scale == 6.0:  # If using default from 1.3B
                sample_guide_scale = 5.0  # Adjust to 14B default
            
            if sample_shift == 8.0:  # If using default from 1.3B
                sample_shift = 5.0  # Adjust to 14B default for I2V
                if internal_resolution in ["832*480", "480*832"]:
                    sample_shift = 3.0  # Special case for SD resolution in I2V
        else:
            # Text-to-video mode
            internal_mode = "t2v"
            print(f"[INFO] Using text-to-video mode with {model_size} model at resolution {resolution}")
            
            # Default T2V parameters
            if model_size == "14B" and sample_guide_scale == 6.0:
                sample_guide_scale = 5.0  # Adjust to 14B default
                
            if model_size == "14B" and sample_shift == 8.0:
                sample_shift = 5.0  # Adjust to 14B default
        
        # Set random seed if not provided
        actual_seed = seed if seed is not None else random.randint(0, 2147483647)
        
        # Set task based on mode, model size, and resolution
        if internal_mode == "t2v":
            task = f"t2v-{model_size}"
        else:  # i2v mode
            # Determine which I2V model to use based on resolution
            if internal_resolution in ["1280*720", "720*1280"]:
                task = f"i2v-{model_size}-720P"
            else:
                task = f"i2v-{model_size}-480P"
        
        # Download model if needed
        model_file = self.model_files[task]
        url = BASE_URL + model_file
        dest_path = os.path.join(MODEL_CACHE, model_file)
        if not os.path.exists(dest_path.replace(".tar", "")):
            download_weights(url, dest_path)
            
        # Get model path
        ckpt_dir = self.model_paths[task]
        
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
        
        # Determine multi-GPU strategy based on available GPUs and model size
        use_distributed = self.num_gpus > 1
        
        if use_distributed:
            print(f"[INFO] Using {self.num_gpus} GPUs for distributed inference")
            
            if model_size == "14B":
                # For 14B model, ulysses attention works better with 4+ GPUs
                if self.num_gpus >= 4:
                    ulysses_size = self.num_gpus
                    ring_size = 1
                else:
                    # With 2-3 GPUs, use a combination
                    ulysses_size = 1
                    ring_size = self.num_gpus
            else:
                # For 1.3B model, ring attention works better
                ulysses_size = 1
                ring_size = self.num_gpus
                
            print(f"[INFO] Distribution strategy: ulysses_size={ulysses_size}, ring_size={ring_size}")
            
            # Start torchrun command for distributed processing
            cmd = ["torchrun", f"--nproc_per_node={self.num_gpus}", "generate.py"]
            cmd.extend(["--task", task])
            cmd.extend(["--size", internal_resolution])
            cmd.extend(["--ckpt_dir", ckpt_dir])
            cmd.extend(["--prompt", prompt])
            cmd.extend(["--base_seed", str(actual_seed)])
            
            # Add image input if in I2V mode
            if internal_mode == "i2v" and image is not None:
                cmd.extend(["--image", str(image)])
            
            # Add FSDP (Fully Sharded Data Parallel) arguments for distributed processing
            cmd.extend(["--dit_fsdp"])
            cmd.extend(["--t5_fsdp"])
            
            # Add distribution strategy settings
            cmd.extend(["--ulysses_size", str(ulysses_size)])
            cmd.extend(["--ring_size", str(ring_size)])
            
            # Add sample parameters
            cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
            cmd.extend(["--sample_shift", str(sample_shift)])
            cmd.extend(["--sample_steps", str(sample_steps)])
        else:
            print("[INFO] Using single GPU for inference")
            
            # Build command for single-GPU generation
            cmd = ["python", "generate.py"]
            cmd.extend(["--task", task])
            cmd.extend(["--size", internal_resolution])
            cmd.extend(["--ckpt_dir", ckpt_dir])
            cmd.extend(["--prompt", prompt])
            cmd.extend(["--base_seed", str(actual_seed)])
            
            # Add image input if in I2V mode
            if internal_mode == "i2v" and image is not None:
                cmd.extend(["--image", str(image)])

            # Add optimization flags for single GPU (more important for 14B model)
            if memory_optimization or model_size == "14B":
                cmd.append("--offload_model")
                cmd.append("True")
                cmd.append("--t5_cpu")

            # Add sample parameters
            cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
            cmd.extend(["--sample_shift", str(sample_shift)])
            cmd.extend(["--sample_steps", str(sample_steps)])

        # Execute the command
        print(f"[INFO] Running command: {' '.join(cmd)}")
        result = subprocess.call(cmd)

        if result != 0:
            raise RuntimeError(f"Generation failed with exit code {result}")

        # Get the current date in the format used by generate.py
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Look for recently created output files with the proper pattern
        pattern = f"{task}*{formatted_time}*.mp4"
        output_files = glob.glob(pattern)

        if not output_files:
            # If no files found with today's date, use a more general pattern as fallback
            fallback_pattern = f"{task}*.mp4"
            output_files = glob.glob(fallback_pattern)
            if not output_files:
                raise RuntimeError(
                    f"Could not find generated output file matching pattern: {pattern} or {fallback_pattern}"
                )
            print(f"[INFO] Using fallback pattern to find output file: {fallback_pattern}")

        # Sort by modification time to get the most recent file
        output_path = sorted(output_files, key=os.path.getmtime)[-1]
        print(f"[INFO] Generated output saved to: {output_path}")

        # Create a simple, URL-safe filename based on the original
        basename = os.path.basename(output_path)
        # Make absolutely sure the filename is URL-safe by removing any potentially problematic characters
        safe_basename = ''.join(c for c in basename if c.isalnum() or c in '._-')

        # If the basename was modified to make it URL-safe, create a new file with the safe name
        if safe_basename != basename:
            safe_output_path = os.path.join(os.path.dirname(output_path), safe_basename)
            print(f"[INFO] Renaming output to ensure URL-safe filename: {safe_output_path}")
            os.rename(output_path, safe_output_path)
            output_path = safe_output_path

        # Simply return the output_path directly since videos are already web-compatible
        return Path(output_path)
