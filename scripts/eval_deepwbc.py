"""Run a specified DeepWBC checkpoint on a fixed task using official Isaac Lab.

--preflight is CPU-only and never imports AppLauncher or constructs a simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    from pawcerto.isaac.deepwbc_runtime import FORCE_MODES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Trusted checkpoint; never auto-selects latest')
    p.add_argument('--config', type=Path, help='Complete config required for upstream checkpoint without config')
    p.add_argument('--task', type=Path, required=True, help='Fixed task JSON; see docs/deepwbc-evaluation.md')
    p.add_argument('--output', type=Path, required=True, help='New evaluation directory, not a training directory')
    p.add_argument('--steps', type=int, required=True, help='Policy steps after source zero-action reset warmup')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--profile', choices=('original-play', 'checkpoint'), default='original-play',
                   help='original-play applies play overrides except terrain geometry; checkpoint preserves thresholds/randomization/schedules')
    p.add_argument('--robot-usd', type=Path)
    p.add_argument('--force-mode', choices=FORCE_MODES)
    p.add_argument('--exported-policy', type=Path, help='Optional exported policy.pt to compare on identical real observations')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--preflight', action='store_true', help='CPU config/checkpoint/task/assets checks only; writes no physics evidence')
    return p


def prepare(args):
    from pawcerto.methods.deepwbc.export import load_checkpoint
    from pawcerto.methods.deepwbc.evaluation import fixed_task_config
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if args.output.exists():
        raise FileExistsError(f'Refusing existing output directory: {args.output}')
    supplied = json.loads(args.config.read_text()) if args.config else None
    loaded = load_checkpoint(args.checkpoint, config=supplied, device='cpu')
    task = json.loads(args.task.read_text())
    config = fixed_task_config(loaded.config, task, num_envs=args.num_envs, seed=args.seed, profile=args.profile)
    saved = config.get('pawcerto_runtime', {})
    usd = args.robot_usd or saved.get('robot_usd')
    if usd is None or not Path(usd).is_file():
        raise ValueError('An existing converted original Go1/WidowX --robot-usd is required')
    usd = Path(usd).resolve()
    # Check the files the real runtime reads, without importing its simulator.
    import math
    import xml.etree.ElementTree as ET
    from pawcerto.isaac.deepwbc_runtime import SIM_DOF_NAMES, link_paths, FORCE_MODES, FORCE_SIGNAL_EVIDENCE
    physical_usd = Path(usd.read_text().strip()).resolve() if usd.suffix == '.txt' else usd
    if not physical_usd.is_file():
        raise FileNotFoundError(f'USD pointer target does not exist: {physical_usd}')
    prepared = next((p for p in (physical_usd.parent, physical_usd.parent.parent)
                     if (p/'source-manifest.json').is_file()), None)
    if prepared is None:
        raise FileNotFoundError('Converted USD requires adjacent source-manifest.json')
    manifest = json.loads((prepared/'source-manifest.json').read_text())
    tree = ET.parse(prepared/'merged.urdf')
    paths = link_paths(tree)
    for link in tree.findall('link'):
        if link.find('collision') is not None and link.get('name') not in paths:
            raise ValueError('Merged URDF collision link is disconnected')
    for name in SIM_DOF_NAMES:
        for key in ('effort', 'velocity'):
            value = manifest['limits'][name][key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid runtime {key} limit for {name}')
    # Asset identity follows the physical USD, independent of pointer spelling.
    usd = physical_usd
    digest = hashlib.sha256(usd.read_bytes()).hexdigest()
    if saved.get('robot_usd_sha256') and digest != saved['robot_usd_sha256']:
        raise ValueError('Robot USD SHA256 differs from checkpoint')
    force_mode = args.force_mode or saved.get('force_mode')
    if force_mode not in FORCE_MODES:
        raise ValueError('Specify an explicit --force-mode for checkpoints without a saved signal')
    config['pawcerto_runtime'] = dict(saved, robot_usd=str(usd), robot_usd_sha256=digest, force_mode=force_mode,
        force_signal_parity=FORCE_SIGNAL_EVIDENCE[force_mode],
        resolved_usd=str(physical_usd.resolve()), resolved_usd_sha256=hashlib.sha256(physical_usd.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((prepared/'source-manifest.json').read_bytes()).hexdigest(),
        merged_urdf_sha256=hashlib.sha256((prepared/'merged.urdf').read_bytes()).hexdigest())
    exported = None
    if args.exported_policy:
        import torch
        exported = torch.jit.load(str(args.exported_policy), map_location='cpu').eval()
        with torch.inference_mode():
            obs = torch.zeros(2, config['env']['num_observations'])
            reference = loaded.model.act_inference(obs, hist_encoding=True)
            torch.testing.assert_close(exported(obs), reference, atol=1e-6, rtol=1e-5)
    return loaded, config, task, exported


def main(argv=None):
    args = parser().parse_args(argv)
    loaded, config, task, exported = prepare(args)
    metadata = dict(checkpoint=str(loaded.checkpoint_path), checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_iteration=loaded.iteration, fixed_task=task, config=config, profile=args.profile,
        requested_steps=args.steps, evidence_source='cpu_preflight' if args.preflight else 'isaac_lab_runtime',
        history_encoding=True, warmup='reset performs one unrecorded zero-action policy step',
        terrain='checkpoint geometry retained; original play.py 600x600 resize not applied',
        force_signal_parity=config['pawcerto_runtime']['force_signal_parity'],
        sphere_helper_parity='reconstructed helper; unpublished upstream extension unavailable',
        exported_policy=str(args.exported_policy.resolve()) if args.exported_policy else None,
        exported_policy_sha256=hashlib.sha256(args.exported_policy.read_bytes()).hexdigest() if args.exported_policy else None)
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'preflight.json').write_text(json.dumps(dict(metadata,
            physics_run=False, evaluation_success=False), indent=2, allow_nan=False)+'\n')
        print('CPU preflight passed; no simulator launched and no physical evaluation performed.')
        return
    from pawcerto.methods.deepwbc.evaluation import TrajectoryWriter, evaluate_fixed_policy
    writer = TrajectoryWriter(args.output, metadata)
    launcher = None
    entered_evaluation = False
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(visualizer='none' if args.headless else 'kit', device=args.device)
        import random
        import numpy as np
        import torch
        from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac
        from pawcerto.methods.deepwbc.terrain import build_terrain, make_terrain_cfg
        from pawcerto.methods.deepwbc.training.isaac_env import DeepWBCIsaacTrainingEnv
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        terrain = build_terrain(seed=args.seed, cfg=config['terrain'])
        terrain_cfg, paths = make_terrain_cfg(terrain)
        runtime = Go1WidowXIsaac(num_envs=args.num_envs, device=args.device,
            usd_path=config['pawcerto_runtime']['robot_usd'], config=config,
            force_signal=config['pawcerto_runtime']['force_mode'], terrain_cfg=terrain_cfg,
            terrain_contact_paths=paths, env_origins=torch.zeros(args.num_envs, 3, device=args.device))
        env = DeepWBCIsaacTrainingEnv(runtime, config)
        model = loaded.model.to(args.device).eval()
        if exported is not None:
            exported = exported.to(args.device)
        entered_evaluation = True
        summary = evaluate_fixed_policy(env, lambda obs: model.act_inference(obs, hist_encoding=True),
            steps=args.steps, writer=writer, exported_policy=exported)
        print(json.dumps(summary, allow_nan=False))
    except BaseException as exc:
        if not entered_evaluation:
            writer.close(status='failed', error=f'{type(exc).__name__}: {exc}')
        # Kit shutdown can exit directly: preserve the traceback before close.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
