#!/usr/bin/env python3
"""Build the official URDF's independent MuJoCo physics asset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.mujoco.asset import build_model
if __name__ == '__main__':
    print(build_model())
