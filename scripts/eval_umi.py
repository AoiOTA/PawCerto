"""Evaluate a full UMI checkpoint with saved inputs plus author eval semantics."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='Full model_N.pt or official directory containing model.pt and config.json')
    parser.add_argument('--usd-path',type=Path,help='Explicit robot USD or USD-path text file; otherwise use the recorded asset, then the workspace default')
    parser.add_argument('--urdf-path',type=Path,help='Matching merged URDF; defaults to the recorded path, then the original robot URDF')
    parser.add_argument('--trajectory',type=Path,default=ROOT/'reference/data/tossing.pkl')
    parser.add_argument('--split-manifest',type=Path)
    parser.add_argument('--partition',choices=('validation','test'),default='validation')
    parser.add_argument('--joint-names',type=Path,help='Optional exact order; defaults to saved robot binding')
    parser.add_argument('--seed',type=int,default=2026,help='Preselected evaluation seed is 2026')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--force-signal',choices=('reconstructed-solver','normal-contact'),
                        default='reconstructed-solver',help='EMD input for this evaluation; does not relabel checkpoint training')
    parser.add_argument('--ground-contact-diagnostics',action=argparse.BooleanOptionalAction,default=True,
                        help='Report separate ground-filtered normal foot forces; never changes reward or current port normal-contact metrics')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.partition == 'test' and not args.split_manifest:
        parser.error('--partition test requires --split-manifest')
    checkpoint_path=args.checkpoint/'model.pt' if args.checkpoint.is_dir() else args.checkpoint
    config_path=checkpoint_path.parent/'config.json'
    if not checkpoint_path.is_file() or not config_path.is_file():
        parser.error('A full checkpoint and its adjacent config.json are required; actor.ts alone omits the stochastic trial policy std.')
    launcher=AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.isaac.runtime import Go2Arx5Isaac,DEFAULT_USD,DEFAULT_URDF
        from pawcerto.methods.umi_on_legs.training import build_algorithm,load_config
        from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv
        from pawcerto.methods.umi_on_legs.training.evaluation import author_eval_config,evaluate_completed_episodes
        from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        source_config=load_config(config_path)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        from pawcerto.methods.umi_on_legs.actuation import require_same_actuation
        require_same_actuation(source_config,checkpoint.get('config',{}))
        config=author_eval_config(source_config,args.trajectory.resolve(),args.device,args.seed)
        from pawcerto.methods.umi_on_legs.data_split import configure_selection,evaluation_identity,checkpoint_training_selection
        training_selection=checkpoint_training_selection(checkpoint)
        selection=configure_selection(config,args.trajectory,args.split_manifest,args.partition)
        identity=evaluation_identity(training_selection,selection)
        checkpoint_runtime=checkpoint.get('config',{}).get('pawcerto_runtime')
        config['pawcerto_runtime']=runtime_contract(
            args.force_signal, config.get('joint_velocity_limit_override_rad_s'), config.get('actuation_mode','external-pd'))
        config['evaluation_weights']=dict(checkpoint=str(checkpoint_path.resolve()),
                                         training_pawcerto_runtime=checkpoint_runtime)
        from pawcerto.methods.umi_on_legs.robot_binding import robot_binding,joint_order,require_same_robot
        require_same_robot(config,checkpoint.get('config',{}))
        binding=robot_binding(config)
        checkpoint_asset=checkpoint.get('config',{}).get('pawcerto_asset')
        usd_path=Path(args.usd_path or (checkpoint_asset or {}).get('usd_path') or binding['usd_path'])
        if usd_path.suffix == '.txt':
            usd_path=Path(usd_path.read_text().strip())
        usd_path=usd_path.resolve()
        from pawcerto.artifacts import file_identity
        asset_identity=file_identity(usd_path)
        urdf_path=Path(args.urdf_path or (checkpoint_asset or {}).get('urdf_path') or binding['urdf_path'])
        urdf_identity=file_identity(urdf_path)
        config['pawcerto_asset']=dict(usd_path=asset_identity['path'],usd_sha256=asset_identity['sha256'],
                                    urdf_path=urdf_identity['path'],urdf_sha256=urdf_identity['sha256'])
        config['env']['headless']=not launcher.has_window
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'resolved_config.json').write_text(json.dumps(config,indent=2))
        runtime=Go2Arx5Isaac(config,joint_order(config,args.joint_names),250,args.device,usd_path=usd_path,urdf_path=urdf_path,training=True,
                           ground_contact_diagnostics=args.ground_contact_diagnostics,
                           force_signal=args.force_signal)
        env=UmiIsaacTrainingEnv(runtime,config,args.trajectory,args.seed)
        model=build_algorithm(config,args.device).actor_critic
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        # Author evaluate.py loads weights, not saved task EMA/curriculum state.
        start=time.monotonic()
        report,episodes=evaluate_completed_episodes(env,model,config,
            progress=lambda steps,done:print(json.dumps(dict(policy_steps=steps,completed_episodes=done)),flush=True))
        task=config['env']['tasks']['reaching']
        history_frames=int(np.rint(task['pose_latency']/task['sequence_sampler']['dt']))+1
        lag_substeps=max(0,history_frames-2)
        report.update(checkpoint=str(checkpoint_path.resolve()),seed=args.seed,
                      engine='Isaac Lab PhysX',
                      asset_identity=config['pawcerto_asset'],
                      checkpoint_training_asset=checkpoint.get('config',{}).get('pawcerto_asset'),
                      protocol='Checkpoint configuration plus author eval semantics',
                      trajectory=str(args.trajectory.resolve()),
                      trajectory_pool=(selection['partition'] if selection else 'Full input pool; no holdout selection'),
                      data_identity=identity,
                      trajectory_coverage='Author random sampling with replacement; not exhaustive ID coverage',
                      resolved_config=str((args.output/'resolved_config.json').resolve()),
                      dense_rewards=True,policy_deterministic=True,
                      stochastic_pre_eval_trial=True,physical_randomization=True,observation_noise=True,
                      push_robots=False,transport_robots=False,
                      ground_contact_diagnostics=args.ground_contact_diagnostics,
                      emd_force_signal=args.force_signal,
                      runtime_metadata=config['pawcerto_runtime'],
                      checkpoint_training_runtime=checkpoint_runtime,
                      support_metric_note='supported_feet counts current port foot normal-contact world Fz >1N from ContactSensor.net_forces_w, which can include self-contact normal forces. ground_supported_feet counts the separate ground-filtered normal world Fz >1N from force_matrix_w. Neither channel is the original Gym forceSensor signal. Dense diagnostic sampling is every physics step.',
                      configured_pose_latency_s=task['pose_latency'],
                      effective_history_lag_after_warmup_s=lag_substeps*runtime.dt,
                      wall_seconds_including_trial=time.monotonic()-start,
                      reward_power_note='Electrical power follows the author estimator sum(abs(torque)*torque_constant*voltage).',
                      source_boundary=('Saved actor observation scales/latency are retained; this is not the current Hydra starter command or exact paper equivalence. '
                          + ('EvenMassDistribution consumes reconstructed current PhysX solver-only foot world Fz on the supported branch; original Gym GPU pointwise identity was not tested.'
                             if args.force_signal=='reconstructed-solver' else
                             'EvenMassDistribution explicitly consumes the previous normal-contact foot Fz proxy.')
                          + ' The checkpoint training source is reported separately; missing legacy metadata is not inferred.'))
        metrics=report['metrics']
        report['author_named_metrics'] = None if metrics is None or env.native_servo else {
            'eval/task/reaching/pos_err/mean':metrics['position_error/mean'],
            'eval/task/reaching/orn_err/mean':metrics['orientation_error/mean'],
            'eval/time_outs/sum':metrics['time_outs/sum'],
            'eval/constraint/energy/sum_electrical_power/mean':metrics['electrical_power/mean'],
            'eval/constraint/energy/sum_joint_energy/mean':metrics['mechanical_power/mean']}
        if env.native_servo:
            report['reward_power_note'] = ('Energy reward retains the author formula and sampling using '
                'servo_effort_prestate_estimate; mechanical/electrical estimates are not measured drive power or energy.')
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        (args.output/'completed_episodes.json').write_text(json.dumps(episodes,indent=2,allow_nan=False))
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':
    main()
