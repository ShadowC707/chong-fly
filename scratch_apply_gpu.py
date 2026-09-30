import os
import re

def patch_file(path, pattern, replacement):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Update optuna_tuner.py
tuner_path = "optimizer/optuna_tuner.py"
patch_file(
    tuner_path,
    r"def run_optuna_study\([\s\S]*?seed: int = 42,[\s\S]*?\) -> Any:",
    r"""def run_optuna_study(
    n_trials: int = 15,
    dataset_path: str = "data/reflex_dataset.pt",
    pretrain: bool = True,
    eval_steps: int = 100,
    seed: int = 42,
    study_name: str = "pareto_search",
    storage: str = "sqlite:///drone_optimization.db",
    device: str = "auto",
) -> Any:"""
)

patch_file(
    tuner_path,
    r"eval_steps=eval_steps,[\s\n]*seed=seed \+ trial.number,[\s\n]*\)",
    r"""eval_steps=eval_steps,
            seed=seed + trial.number,
            device=device,
        )"""
)

patch_file(
    tuner_path,
    r'parser\.add_argument\("--seed", type=int, default=42, help="RNG seed"\)',
    r"""parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")"""
)

patch_file(
    tuner_path,
    r"seed=args\.seed,[\s\n]*\)",
    r"""seed=args.seed,
        device=args.device,
    )"""
)

# 2. Update evaluate.py
eval_path = "optimizer/evaluate.py"
patch_file(
    eval_path,
    r"def objective\([\s\S]*?seed: int = 42,[\s\S]*?\) -> Tuple\[float, float, float\]:",
    r"""def objective(
    trial: Any = None,
    dataset_path: str = "data/reflex_dataset.pt",
    pretrain: bool = True,
    pretrain_epochs: int = 15,
    subset_ratio: float = 0.7,
    eval_steps: int = 100,
    seed: int = 42,
    device: str = "auto",
) -> Tuple[float, float, float]:"""
)

device_logic = r"""
    import torch
    
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: CUDA not available, falling back to CPU.")
            target_device = torch.device("cpu")
            
    # Move policy to device
    try:
        policy = policy.to(target_device)
    except Exception as e:
        print(f"WARNING: Could not move policy to {target_device}: {e}")
        
    if pretrain:
        trial_seed = getattr(trial, "number", seed) if trial is not None else seed
        policy = pretrain_policy(
            policy=policy,
            dataset_path=dataset_path,
            epochs=pretrain_epochs,
            subset_ratio=subset_ratio,
            seed=trial_seed,
            device=target_device,
        )
"""

patch_file(
    eval_path,
    r"    if pretrain:[\s\S]*?seed=trial_seed,[\s\n]*\)",
    device_logic
)

print("Patched!")
