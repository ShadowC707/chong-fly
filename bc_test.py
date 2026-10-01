import sys
import torch
from optimizer.evaluate import create_model
from optimizer.pretrain import pretrain_policy
from simulation.policy import ChongFlyMSPPolicy

policy = create_model({"k_clusters": 32, "pruning_sparsity": 0.5, "solver_type": "CfC"}, allow_fallback=False)
policy = pretrain_policy(policy, epochs=10)
