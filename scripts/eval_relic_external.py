"""Run the original external ReLIC PLAY task with its explicit pretrained actor."""

import argparse
import json
from pathlib import Path
import sys
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--external-provenance", type=Path, required=True,
                        help="directory containing source.json and external-modifications.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    external = args.external_root.resolve(strict=True)
    policy = args.policy.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    from pawcerto.methods.relic.isaac_eval import (
        UPSTREAM_REVISION, compatible_spot_knee, file_identity, recorder_config,
    )
    manifest = {str(p.relative_to(external)): file_identity(p)["sha256"]
                for p in sorted(external.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts}
    (args.output / "external-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {
        "source_revision": UPSTREAM_REVISION,
        "original_archive_provenance": json.loads((args.external_provenance / "source.json").read_text()),
        "external_modifications": json.loads((args.external_provenance / "external-modifications.json").read_text()),
        "external_root": str(external), "license": file_identity(external / "LICENSE"),
        "model": file_identity(policy), "command": sys.argv,
        "requested_steps": args.steps, "optimizer_updates": 0, "seed": args.seed,
        "source_task": "Isaac-Spot-Interlimb-Play-v0",
        "training_provenance": "pretrained export stage/seed/update count unknown",
        "evidence_boundary": "bounded source PLAY execution on current official Lab/PhysX; not training reproduction or original-runtime equivalence",
        "overrides": ["one environment after PLAY initialization", "explicit device and seed",
                      "public use_newton_actuators=False", "class-local Spot knee constructor compatibility",
                      "public pre-reset recorder", "URDF conversion output directory"],
        "snapshot_timing": "public recorder post-step after reward/termination and before automatic reset",
        "consumer": file_identity(__file__),
        "bridge": file_identity(ROOT / "pawcerto/methods/relic/isaac_eval.py"),
    }
    app = env = None
    records = []
    observations = []
    phase = "initialization"
    attempted_steps = 0
    try:
        import numpy as np
        import torch
        from isaaclab.app import AppLauncher
        torch.set_num_threads(1)
        app = AppLauncher(visualizer="none", device=args.device)
        sys.path.insert(0, str(external / "source/relic"))
        import gymnasium as gym
        import relic.tasks  # registers the original task
        from relic.actuators import SpotKneeActuator
        from relic.tasks.loco_manipulation.config.spot.spot_env_cfg import SpotInterlimbEnvCfg_PLAY
        import isaaclab.actuators.actuator_pd as native_pd
        cfg = SpotInterlimbEnvCfg_PLAY()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.sim.use_newton_actuators = False
        cfg.scene.robot.spawn.usd_dir = str(args.output.resolve() / "converted")
        cfg.scene.robot.actuators["spot_knee"].class_type = compatible_spot_knee(SpotKneeActuator)
        cfg.recorders = recorder_config()
        (args.output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
        metadata["native_pd"] = file_identity(native_pd.__file__)
        lab_root = Path(native_pd.__file__).resolve().parents[4]
        metadata["runtime_source_head"] = subprocess.run(
            ["git", "-C", str(lab_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        metadata["torch_version"] = torch.__version__
        metadata["step_dt"] = cfg.sim.dt * cfg.decimation
        metadata["physics_dt"] = cfg.sim.dt
        env = gym.make("Isaac-Spot-Interlimb-Play-v0", cfg=cfg)
        base = env.unwrapped
        base._external_relic_records = records
        robot = base.scene["robot"]
        action = base.action_manager.get_term("joint_pos")
        metadata["joint_names"] = robot.joint_names
        metadata["body_names"] = robot.body_names
        metadata["action_joint_ids"] = [int(value) for value in action._joint_ids]
        metadata["action_joint_names"] = action._joint_names
        metadata["arm_joint_ids"] = [int(value) for value in action._arm_joint_ids]
        metadata["arm_joint_names"] = action._arm_joint_names
        metadata["reward_terms"] = base.reward_manager.active_terms
        metadata["termination_terms"] = base.termination_manager.active_terms
        metadata["actuator_types"] = {k: f"{type(v).__module__}.{type(v).__qualname__}" for k, v in robot.actuators.items()}
        if len(robot.joint_names) != 19:
            raise ValueError("external source Spot must expose 19 joints")
        model = torch.jit.load(str(policy), map_location=base.device).eval()
        phase = "source_reset"
        obs = env.reset(seed=args.seed)[0]["policy"]
        metadata["ground_contact_force_shape"] = list(base.scene.sensors["robot_to_ground_contact_forces"].data.force_matrix_w.shape)
        metadata["contact_signal_boundary"] = "current official PhysX force_matrix_w warns that it returns normal_force_matrix_w, not total contact force; historical signal equivalence unverified"
        with torch.inference_mode():
            for step in range(args.steps):
                if obs.shape != (1, 84) or obs.dtype != torch.float32 or not torch.isfinite(obs).all():
                    raise ValueError("actual source observation ABI differs from finite float32 (1,84)")
                actions = model(obs)
                if actions.shape != (1, 12) or not torch.isfinite(actions).all():
                    raise ValueError("actual pretrained actor output differs from finite (1,12)")
                observations.append(obs.detach().cpu().numpy().copy())
                phase = "source_env_step"
                attempted_steps = step + 1
                result = env.step(actions)
                phase = "pre_reset_record_validation"
                obs = result[0]["policy"]
                if len(records) != step + 1:
                    raise RuntimeError("pre-reset recorder did not capture exactly one source step")
                if not all(np.isfinite(value).all() for value in records[-1].values()):
                    raise RuntimeError("nonfinite values in original source transition; first invalid row retained")
        phase = "trajectory_serialization"
        arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
        arrays["policy_observation"] = np.stack(observations)
        np.savez_compressed(args.output / "trajectory.npz", **arrays)
        finite = all(np.isfinite(v).all() for v in arrays.values())
        summary = dict(metadata, completed_steps=len(records), finite=bool(finite),
                       termination_count=int(arrays["terminated"].sum()),
                       timeout_count=int(arrays["truncated"].sum()),
                       reward_sum=float(arrays["reward"].sum()),
                       minimum_pre_reset_root_height=float(arrays["root_state_w"][..., 2].min()),
                       physics_run=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, allow_nan=False), flush=True)
        if not finite:
            raise RuntimeError("nonfinite values in original source trajectory")
    except BaseException as exc:
        # Kit may exit inside close(); emit the cause before any serialization or cleanup.
        traceback.print_exc()
        sys.stderr.flush()
        (args.output / "failure.json").write_text(json.dumps(dict(metadata,
            completed_steps=len(records), attempted_steps=attempted_steps, phase=phase,
            error=repr(exc), traceback=traceback.format_exc()),
            indent=2, default=str) + "\n")
        if records:
            import numpy as np
            np.savez_compressed(args.output / "partial-trajectory.npz",
                                policy_observation=np.stack(observations[:len(records)]),
                                **{k: np.stack([r[k] for r in records]) for k in records[0]})
        if len(observations) > len(records):
            import numpy as np
            np.save(args.output / "pending-policy-observation.npy", observations[-1])
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == "__main__":
    main()
