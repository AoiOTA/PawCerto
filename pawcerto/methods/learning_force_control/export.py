"""Consume original raw ac_weights state dictionaries and original two-JIT export.

Only load trusted checkpoint/JIT files. Raw upstream weights contain no config;
callers must supply the resolved configuration belonging to those weights.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import torch
from torch import nn
from .config import build_model


def load_checkpoint(path, *, config):
    model = build_model(config)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def export_policy(model, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, module in [('body_latest.jit', model.actor_body),
                         ('adaptation_module_latest.jit', model.adaptation_module)]:
        torch.jit.script(deepcopy(module).cpu().eval()).save(str(destination / name))
    return destination


class JitPolicy(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        self.body = torch.jit.load(str(directory / 'body_latest.jit'), map_location='cpu').eval()
        self.adaptation = torch.jit.load(str(directory / 'adaptation_module_latest.jit'), map_location='cpu').eval()

    def forward(self, obs_history):
        return self.body(torch.cat((obs_history, self.adaptation(obs_history)), dim=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, config=json.loads(args.config.read_text()))
    export_policy(model, args.output)


if __name__ == '__main__':
    main()
