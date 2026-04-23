import os
import sys
import time
import yaml
import json
import random
import torch
import wandb
import logging
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import VRPTrainer as Trainer
from tuner import VRPTuner as Tuner
from utils.utils import copy_all_src


def setup_cuda_optimizations():
    """Configure PyTorch/CUDA optimizations (Flash Attention, cuDNN, TF32)."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"\n{'=' * 60}")
        print(f"CUDA Optimizations Enabled:")
        print(f"  GPU: {gpu_name} (Compute {capability[0]}.{capability[1]})")
        print(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"{'=' * 60}\n")


def init_seeds(seed):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_logger(args):
    """Setup logging to file and console (rank 0 only in DDP)."""
    args.result_dir = os.path.join("result", args.start_time)
    os.makedirs(args.result_dir, exist_ok=True)

    args.log_file = os.path.join(args.result_dir, f"log_{args.rank}.txt")
    logging.basicConfig(
        filename=args.log_file, format="%(asctime)-15s %(message)s", level=logging.INFO
    )
    logger_ = logging.getLogger()

    # Console logging only for rank 0 in DDP
    console = logging.StreamHandler(sys.stdout)
    args.mute = args.ddp and args.rank != 0
    if not args.mute:
        logger_.addHandler(console)

    return logger_


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Task Learning for VRP Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size per GPU"
    )
    parser.add_argument(
        "--n_size", type=int, default=50, help="Problem size (50 or 100)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Data loading workers"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")

    # Checkpointing
    parser.add_argument(
        "--resume", dest="resume", action="store_true", help="Resume from checkpoint"
    )
    parser.add_argument("--epoch", type=int, help="Checkpoint epoch to resume")
    parser.add_argument("--path_id", type=str, help="Checkpoint folder (timestamp)")

    # Mode flags
    parser.add_argument(
        "--test", dest="test", action="store_true", help="Enable testing"
    )
    parser.add_argument(
        "--test_lib", dest="test_lib", action="store_true", help="Test on CVRPLIB"
    )
    parser.add_argument(
        "--test_only", dest="test_only", action="store_true", help="Test only"
    )
    parser.add_argument(
        "--test_search",
        dest="test_search",
        action="store_true",
        help="Apply local search to the best solution found during testing",
    )
    parser.add_argument(
        "--skip", dest="skip", action="store_true", help="Quick test (3 steps)"
    )

    # Fine-tuning
    parser.add_argument(
        "--tune", dest="tune", action="store_true", help="Fine-tune mode"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["mb", "md", "both"],
        help="Variant: mb (mixed backhaul), md (multi-depot), both",
    )

    # Logging
    parser.add_argument("--wandb", type=str, default="", help="W&B project ID")

    return parser.parse_args()


def load_config(args):
    """Load YAML configuration file."""
    with open("config.yaml", "r") as f:
        cfg_yaml = yaml.safe_load(f)

    for key, value in cfg_yaml.items():
        assert not hasattr(args, key), f"Config key already exists: {key}"
        setattr(args, key, value)


def set_test_params(args):
    """Map test indices to actual configurations and generate metric labels."""
    args.env["test_size"] = [
        args.all_test_size[idx] for idx in args.env["test_size_idx"]
    ]
    args.env["test_problem"] = [
        args.all_test_problem[idx] for idx in args.env["test_problem_idx"]
    ]
    args.env["test_distribution"] = [
        args.all_test_distribution[idx] for idx in args.env["test_distribution_idx"]
    ]

    # Generate metric labels: s/{idx}-{size}-metric, p/{idx}-{problem}-metric, d/{idx}-{dist}-metric
    size_labels = [
        f"s/{idx:02d}-{args.all_test_size[idx]}-a8gap%"
        for idx in args.env["test_size_idx"]
    ]
    size_labels += [
        f"s/{idx + len(args.all_test_size):02d}-{args.all_test_size[idx]}-gap%"
        for idx in args.env["test_size_idx"]
    ]

    problem_labels = [
        f"p/{idx:02d}-{args.all_test_problem[idx]}-a8gap%"
        for idx in args.env["test_problem_idx"]
    ]
    problem_labels += [
        f"p/{idx + len(args.all_test_problem):02d}-{args.all_test_problem[idx]}-gap%"
        for idx in args.env["test_problem_idx"]
    ]

    distribution_labels = [
        f"d/{idx:02d}-{args.all_test_distribution[idx]}-a8gap%"
        for idx in args.env["test_distribution_idx"]
    ]
    distribution_labels += [
        f"d/{idx + len(args.all_test_distribution):02d}-{args.all_test_distribution[idx]}-gap%"
        for idx in args.env["test_distribution_idx"]
    ]

    args.test_metric_label = size_labels + problem_labels + distribution_labels


def setup_distributed_training(args):
    """Setup DDP if enabled via torchrun environment variables."""
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.ddp = "LOCAL_RANK" in os.environ.keys()

    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)


def configure_training_settings(args):
    """Configure training settings from arguments."""
    assert args.n_size in [50, 100], f"n_size must be 50 or 100, got {args.n_size}"

    args.env["generator_params"]["num_loc"] = args.n_size
    args.env["test_size_idx"] = [0] if args.n_size == 50 else [1]

    args.model_params["sqrt_embedding_dim"] = args.model_params["embedding_dim"] ** (
        1 / 2
    )
    args.optimizer_params["optimizer"]["lr"] = float(
        args.optimizer_params["optimizer"]["lr"]
    )
    args.optimizer_params["optimizer"]["weight_decay"] = float(
        args.optimizer_params["optimizer"]["weight_decay"]
    )

    args.trainer_params["train_batch_size"] = args.batch_size
    args.trainer_params["model_load"] = {"enable": args.resume}

    if args.resume:
        args.trainer_params["model_load"]["path"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "result", args.path_id
        )
        args.trainer_params["model_load"]["epoch"] = args.epoch

    # 16 VRP variants: C=Closed, O=Open, L=Limit, B=Backhaul, TW=Time Windows
    args.p_set = [
        "C",
        "O",
        "L",
        "B",
        "TW",
        "LB",
        "TWO",
        "LTW",
        "OB",
        "LO",
        "TWB",
        "LOB",
        "LTWB",
        "LTWO",
        "TWOB",
        "LTWOB",
    ]


def setup_logging_and_wandb(args):
    """Setup logging and Weights & Biases."""
    logger = get_logger(args)

    # W&B only on rank 0
    if args.wandb != "" and (not args.ddp or args.rank == 0):
        project_name = (
            f"vrp{args.n_size}-tune-{args.tuner_params.get('variant_present', 'mb')}"
            if args.tune
            else f"vrp{args.n_size}"
        )
        wandb.init(project=project_name, config=args.__dict__, id=args.wandb)

    args.log = logger.info
    return logger


def main(args):
    """Main execution: setup CUDA, initialize, run training or fine-tuning."""
    setup_cuda_optimizations()
    init_seeds(args.seed)

    runner = Tuner(args) if args.tune else Trainer(args)
    args.log(
        f"Starting {'fine-tuning on ' + args.tuner_params['variant_present'] if args.tune else 'training on 16 VRP variants'}"
    )

    # Copy source for reproducibility
    if not args.mute:
        copy_all_src(args.result_dir)
    if args.ddp:
        torch.distributed.barrier()

    args.log("Finished copying source files.")
    runner.run()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    args.start_time = time.strftime("%Y-%m%d-%H%M", time.localtime())

    # Load config
    load_config(args)
    if args.variant is not None:
        args.tuner_params["variant_present"] = args.variant

    # Setup distributed training
    setup_distributed_training(args)

    # Configure settings
    configure_training_settings(args)
    set_test_params(args)

    # Setup logging
    logger = setup_logging_and_wandb(args)

    # Adjust seed for DDP
    if args.ddp:
        args.seed = args.seed + args.rank + (int(time.time()) if args.resume else 0)

    # Log configuration (exclude large lists and non-serializable objects)
    do_not_log = {
        "p_set",
        "task_set",
        "dist_set",
        "n_set",
        "metric_label",
        "test_metric_label",
        "all_test_size",
        "all_test_problem",
        "all_test_distribution",
        "all_tuning_variants",
    }
    log_dict = {
        k: v for k, v in vars(args).items() if k not in do_not_log and not callable(v)
    }
    logger.info(json.dumps(log_dict, indent=4, default=str))

    # Run
    main(args)
