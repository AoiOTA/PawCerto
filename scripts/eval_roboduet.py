"""Run fixed RoboDuet five-module policies through official Isaac Lab physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Full checkpoint; export official_play before physical consumption')
    source.add_argument('--export', type=Path, help='Existing five-module official_play directory')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/roboduet_evaluation.json')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--usd', type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; preserve earlier evidence and choose a new directory')
    launcher = AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.methods.roboduet.export import ExportedRoboDuet, export_checkpoint
        from pawcerto.methods.roboduet.evaluation import nominal_config, validate_protocol, evaluate_seed
        from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac, DEFAULT_USD
        from pawcerto.artifacts import file_identity
        torch.set_num_threads(1)
        args.output.mkdir(parents=True, exist_ok=False)
        protocol_bytes = args.protocol.read_bytes()
        protocol = json.loads(protocol_bytes)
        (args.output/'protocol.json').write_bytes(protocol_bytes)
        directory = args.export
        if args.checkpoint:
            directory = args.output/'policy'
            names = json.loads((ROOT/'configs/roboduet_go1_arx5_joint_names.json').read_text())
            export_checkpoint(args.checkpoint, directory, names, mode='official_play')
        policy = ExportedRoboDuet(directory)
        if policy.mode != protocol['mode'] or policy.mode != 'official_play':
            raise ValueError('Main physical protocol requires official_play')
        validate_protocol(protocol, policy.config)
        config = nominal_config(policy.config, len(protocol['cases']))
        (args.output/'evaluation_config.json').write_text(json.dumps(config,indent=2)+'\n')
        kwargs = dict(num_envs=len(protocol['cases']), device=args.device)
        saved_asset = policy.config.get('pawcerto_asset', {})
        usd = Path(args.usd or saved_asset.get('usd_path') or DEFAULT_USD)
        if usd.suffix == '.txt':
            usd = Path(usd.read_text().strip())
        asset_identity = file_identity(usd)
        if not args.usd and saved_asset.get('usd_sha256') not in (None, asset_identity['sha256']):
            raise ValueError('Saved RoboDuet USD content changed')
        kwargs['usd_path'] = usd
        runtime = Go1Arx5Isaac(**kwargs)
        if list(runtime.joint_names) != policy.joint_names:
            raise ValueError('Export joint names differ from actual physical runtime order')
        results = []
        for seed in protocol['seeds']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = RoboDuetIsaacTrainingEnv(runtime, config)
            recorder = evaluate_seed(env, policy, protocol)
            metadata = dict(seed=seed, export=policy.info, protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                            body_names=runtime.body_names, joint_names=runtime.joint_names,
                            physical_asset=asset_identity,
                            state_quaternion_order='xyzw', physics_dt=runtime.dt, control_dt=env.dt,
                            evidence='official Isaac Lab physical rollout; terminal samples captured before autoreset',
                            head_contact_scope='control summaries use final substep; substeps.pt retains every 5ms contact sample',
                            substep_effort_scope='sent_effort_command is the exact combined20 passed to runtime set_joint_effort_target; excludes implicit spring/damper and is not measured solver joint force',
                            saturation_scope='leg commanded effort and policy action clipping at last substep; arm mixed controller entries are position targets')
            result = recorder.save(args.output/f'seed_{seed}', metadata)
            results.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
