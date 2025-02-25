# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import random
import subprocess
from pathlib import Path

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
    
        # Create model cache directory
        self.model_cache = "./model_cache"
        os.makedirs(self.model_cache, exist_ok=True)
        
        # Define model paths
        self.model_paths = {
            "t2v-14B": f"{self.model_cache}/Wan2.1-T2V-14B",
            "t2v-1.3B": f"{self.model_cache}/Wan2.1-T2V-1.3B",
            "i2v-14B-720P": f"{self.model_cache}/Wan2.1-I2V-14B-720P",
            "i2v-14B-480P": f"{self.model_cache}/Wan2.1-I2V-14B-480P"
        }
        
        # Move existing model directories to model_cache if they exist
        for model_name, model_path in self.model_paths.items():
            old_path = f"./Wan2.1-{model_name}"
            if os.path.exists(old_path) and os.path.isdir(old_path) and not os.path.exists(model_path):
                print(f"Moving existing model {model_name} from {old_path} to {model_path}")
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Use system command for moving as it's more reliable for large directories
                os.system(f"mv {old_path} {model_path}")
                print(f"Successfully moved model {model_name}")
        
        # Check if generate.py exists, if not, download it
        if not os.path.exists("generate.py"):
            print("generate.py not found. Downloading from GitHub...")
            try:
                os.system("curl -O https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py")
                print("Successfully downloaded generate.py")
            except Exception as e:
                print(f"Error downloading generate.py: {str(e)}")
                print("Please download it manually from the Wan2.1 GitHub repository: https://github.com/Wan-Video/Wan2.1")
        
        # Check if models exist, if not, download them
        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path) or not os.listdir(model_path):  # Check if directory is empty
                print(f"Model {model_name} not found at {model_path}. Downloading...")
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(model_path, exist_ok=True)
                    
                    # Download model using huggingface-cli
                    cmd = f"huggingface-cli download Wan-AI/Wan2.1-{model_name} --local-dir {model_path}"
                    print(f"Running: {cmd}")
                    os.system(cmd)
                    print(f"Successfully downloaded model {model_name}")
                except Exception as e:
                    print(f"Error downloading model {model_name}: {str(e)}")
                    print(f"Please download it manually using: huggingface-cli download Wan-AI/Wan2.1-{model_name} --local-dir {model_path}")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing what you want to generate"
        ),
        mode: str = Input(
            description="Generation mode: text-to-video, image-to-video, or text-to-image",
            choices=["text-to-video", "image-to-video", "text-to-image"],
            default="text-to-video"
        ),
        image: Path = Input(
            description="Input image for image-to-video generation (only required for image-to-video mode)",
            default=None
        ),
        model_quality: str = Input(
            description="Model quality (higher quality is slower but better results)",
            choices=["standard", "fast"],
            default="standard"
        ),
        resolution: str = Input(
            description="Output resolution",
            choices=["HD (1280×720)", "SD (832×480)", "Square (960×960)", "Portrait HD (720×1280)", "Widescreen (1088×832)", "Portrait (832×1088)", "Square HD (1024×1024)"],
            default="HD (1280×720)"
        ),
        seed: int = Input(
            description="Random seed for reproducible results (-1 for random)",
            default=-1
        ),
        advanced_settings: bool = Input(
            description="Enable advanced settings (not recommended for most users)",
            default=False
        ),
        use_prompt_extension: bool = Input(
            description="[Advanced] Use prompt extension to enhance generation quality",
            default=False
        ),
        memory_optimization: bool = Input(
            description="[Advanced] Enable memory optimization (only needed if you encounter out-of-memory errors)",
            default=False
        ),
        sample_guide_scale: float = Input(
            description="[Advanced] Guidance scale for sampling (only for fast model)",
            default=6.0
        ),
        sample_shift: int = Input(
            description="[Advanced] Sample shift parameter (only for fast model)",
            default=8
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # Map user-friendly choices to internal values
        mode_map = {
            "text-to-video": "t2v",
            "image-to-video": "i2v",
            "text-to-image": "t2i"
        }
        
        resolution_map = {
            "HD (1280×720)": "1280*720",
            "SD (832×480)": "832*480",
            "Square (960×960)": "960*960",
            "Portrait HD (720×1280)": "720*1280",
            "Widescreen (1088×832)": "1088*832",
            "Portrait (832×1088)": "832*1088",
            "Square HD (1024×1024)": "1024*1024"
        }
        
        model_quality_map = {
            "standard": "14B",
            "fast": "1.3B"
        }
        
        # Convert user-friendly inputs to internal values
        internal_mode = mode_map[mode]
        internal_resolution = resolution_map[resolution]
        model_size = model_quality_map[model_quality]
        
        # Validate inputs
        if internal_mode == "i2v" and image is None:
            raise ValueError("An input image is required for image-to-video mode")
        
        # Set random seed if not provided
        actual_seed = seed if seed >= 0 else random.randint(0, 2147483647)
        
        # Determine the task based on mode and model size
        if internal_mode == "t2v":
            task = f"t2v-{model_size}"
            if model_size == "1.3B" and internal_resolution not in ["832*480"]:
                print("Note: Fast model (1.3B) is optimized for SD resolution. Switching to 832*480.")
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
            cmd = f"huggingface-cli download Wan-AI/Wan2.1-{task} --local-dir {ckpt_dir}"
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
        
        # Add prompt extension if requested
        if advanced_settings and use_prompt_extension:
            cmd.append("--use_prompt_extend")
            cmd.extend(["--prompt_extend_method", "local_qwen"])
            cmd.extend(["--prompt_extend_target_lang", "en"])
        
        # Add optimization flags if requested
        if advanced_settings and memory_optimization:
            cmd.append("--offload_model")
            cmd.append("True")
            cmd.append("--t5_cpu")
        
        # Add sample parameters for T2V-1.3B
        if task == "t2v-1.3B" and advanced_settings:
            cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
            cmd.extend(["--sample_shift", str(sample_shift)])
        
        # Execute the command
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            raise RuntimeError(f"Generation failed with exit code {result.returncode}")
        
        # Parse the output to find the generated video/image path
        output_lines = result.stdout.split('\n')
        output_path = None
        for line in output_lines:
            if "Saved video to" in line:
                output_path = line.split("Saved video to")[-1].strip()
            elif "Saved image to" in line:
                output_path = line.split("Saved image to")[-1].strip()
        
        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("Could not find generated output file")
        
        # Copy the output to a standard location
        if internal_mode in ["t2v", "i2v"]:
            final_path = Path("/tmp/output.mp4")
        else:  # t2i
            final_path = Path("/tmp/output.png")
        
        with open(output_path, "rb") as src, open(final_path, "wb") as dst:
            dst.write(src.read())
        
        return final_path
