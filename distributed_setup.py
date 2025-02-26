import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import time
from functools import partial
import queue
from queue import Empty

# Import necessary modules
import wan
from wan.configs import WAN_CONFIGS
from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)

# Global variables for process communication
_TASK_QUEUES = {}
_RESULT_QUEUES = {}
_STOP_EVENT = None
_READY_EVENTS = []

def _init_process(rank, world_size, model_path, task, ulysses_size, ring_size, 
                  task_queue, result_queue, ready_event, stop_event):
    """
    Initialize a process for distributed model loading and processing
    """
    try:
        # Set up environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.ERROR,
            format=f"[{rank}] [%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
        
        # Setup device
        device = rank
        torch.cuda.set_device(device)
        
        # Initialize the distributed process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        
        # Initialize model parallel environment
        init_distributed_environment(rank=rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=ring_size,
            ulysses_degree=ulysses_size
        )
        
        # Load configuration
        cfg = WAN_CONFIGS[task]
        if rank == 0:
            logging.info(f"Loading model config: {cfg}")
        
        # Create model with distributed parameters
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=device,
            rank=rank,
            t5_fsdp=True,  # Use FSDP for distributed model
            dit_fsdp=True,  # Use FSDP for distributed model
            use_usp=(ulysses_size > 1 or ring_size > 1),
            t5_cpu=False   # Keep T5 on GPU for faster inference
        )
        
        if rank == 0:
            logging.info(f"Model loaded successfully on rank {rank}")
        
        # Signal that this process is ready
        ready_event.set()
        
        # Process tasks until stopped
        while not stop_event.is_set():
            try:
                # Try to get a task with timeout to allow checking stop_event
                try:
                    task = task_queue.get(timeout=1.0)
                    
                    # Unpack task parameters
                    task_id, prompt, size, frame_num, sample_solver, sampling_steps, guide_scale, shift, seed = task
                    
                    # Generate video
                    video = model.generate(
                        input_prompt=prompt,
                        size=size,
                        frame_num=frame_num,
                        shift=shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=seed,
                        offload_model=False  # Keep model in GPU memory
                    )
                    
                    # Only rank 0 returns the actual video, others just participate in generation
                    if rank == 0:
                        result_queue.put((task_id, video))
                    else:
                        # Just acknowledge completion
                        result_queue.put((task_id, None))
                        
                except Empty:
                    # No task available, continue waiting
                    continue
                    
            except Exception as e:
                logging.error(f"Error processing task in process {rank}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                # Put error in result queue
                result_queue.put(("ERROR", str(e)))
                
    except Exception as e:
        logging.error(f"Error in process {rank}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Signal error but still mark as ready to prevent hanging
        ready_event.set()


def setup_distributed_model(model_path, task, num_gpus):
    """
    Sets up a distributed model across all available GPUs
    
    Args:
        model_path: Path to the model weights
        task: Model task (e.g., "t2v-1.3B")
        num_gpus: Number of GPUs to use
        
    Returns:
        Queues and events for communicating with the processes
    """
    global _TASK_QUEUES, _RESULT_QUEUES, _STOP_EVENT, _READY_EVENTS
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    # Create queues for each GPU process
    task_queues = {rank: mp.Queue() for rank in range(num_gpus)}
    result_queues = {rank: mp.Queue() for rank in range(num_gpus)}
    
    # Create events for synchronization
    ready_events = [mp.Event() for _ in range(num_gpus)]
    stop_event = mp.Event()
    
    # Store globally
    _TASK_QUEUES = task_queues
    _RESULT_QUEUES = result_queues
    _STOP_EVENT = stop_event
    _READY_EVENTS = ready_events
    
    # Launch processes for each GPU
    processes = []
    for rank in range(num_gpus):
        # Create and start a process
        p = mp.Process(
            target=_init_process,
            args=(rank, num_gpus, model_path, task, 1, num_gpus, 
                  task_queues[rank], result_queues[rank], ready_events[rank], stop_event)
        )
        p.daemon = True  # Make process a daemon so it exits when the main process exits
        p.start()
        processes.append(p)
    
    # Wait for all processes to be ready
    print(f"Waiting for {num_gpus} GPU processes to initialize...")
    timeout = 300  # 5 minutes timeout
    for i, event in enumerate(ready_events):
        if not event.wait(timeout):
            # Try to clean up
            stop_event.set()
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"Timeout waiting for GPU {i} initialization")
    
    print(f"All {num_gpus} GPU processes initialized successfully")
    return processes


def generate_with_model(prompt, size, frame_num,
                       sample_solver='unipc',
                       sampling_steps=50,
                       guide_scale=5.0,
                       shift=5.0,
                       seed=-1):
    """
    Generate video using the distributed model processes.
    
    Args:
        prompt: Text prompt for generation
        size: Video size (tuple of (width, height))
        frame_num: Number of frames in the video
        sample_solver: Sampling solver to use
        sampling_steps: Number of sampling steps
        guide_scale: Guidance scale
        shift: Sampling shift
        seed: Random seed
        
    Returns:
        Generated video tensor
    """
    global _TASK_QUEUES, _RESULT_QUEUES
    
    if not _TASK_QUEUES or not _RESULT_QUEUES:
        raise RuntimeError("Distributed processes not initialized. Call setup_distributed_model first.")
    
    # Create a unique task ID
    task_id = f"task_{time.time()}"
    
    # Send task to all GPU processes
    for rank, queue in _TASK_QUEUES.items():
        task_data = (task_id, prompt, size, frame_num, sample_solver, 
                    sampling_steps, guide_scale, shift, seed)
        queue.put(task_data)
    
    # Wait for results from all processes
    results = {}
    num_gpus = len(_TASK_QUEUES)
    timeout = 600  # 10 minutes timeout
    
    start_time = time.time()
    while len(results) < num_gpus and (time.time() - start_time) < timeout:
        for rank, queue in _RESULT_QUEUES.items():
            if rank not in results:
                try:
                    result_task_id, result_data = queue.get(block=False)
                    if result_task_id == "ERROR":
                        raise RuntimeError(f"Error in GPU {rank}: {result_data}")
                    if result_task_id == task_id:
                        results[rank] = result_data
                except Empty:
                    # No result yet, continue waiting
                    pass
        time.sleep(0.1)  # Small sleep to avoid CPU spinning
    
    if len(results) < num_gpus:
        raise RuntimeError(f"Timeout waiting for results from all GPUs")
    
    # Return result from rank 0 (which has the actual video)
    return results[0]
