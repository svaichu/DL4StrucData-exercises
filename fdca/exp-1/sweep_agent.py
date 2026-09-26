"""
wandb sweep agent entrypoint for the JEPA-style Phase-1 exp-1 variant.

wandb's sweep runner calls this script once per trial with sys.argv containing
--param value pairs matching sweep.yaml's `parameters` block (wandb appends
these automatically when it invokes `program:` under `wandb agent`). We reuse
parse_args()/run_training() from train_phase1_jepa.py directly so there is a
single source of truth for the training loop; this file only wires wandb.init()
(which auto-populates cfg overrides from wandb.config when running under
`wandb agent`) to that function.
"""
from __future__ import annotations

import argparse

import wandb

from train_phase1_jepa import parse_args, run_training


def main() -> None:
    # parse_args() supplies defaults for anything the sweep does not override;
    # wandb.init() below reads the actual trial's parameters from wandb.config
    # once invoked via `wandb agent`.
    base_cfg = parse_args([])  # defaults only; sweep controller overrides via wandb.config
    wandb.init(config=vars(base_cfg))
    cfg = argparse.Namespace(**wandb.config.as_dict())
    result = run_training(cfg)
    wandb.log(result)
    wandb.finish()


if __name__ == "__main__":
    main()
