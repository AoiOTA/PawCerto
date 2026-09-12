"""Device-owner-only physical test of two shared collider material channels."""
import argparse
import json
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args)
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab_physx.physics import PhysxCfg
    from pawcerto.isaac.deepwbc_material_probe import build_fixture
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device,
        physics=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=False)))
    metadata = build_fixture(sim.stage, args.output)
    # Reuse the official inverted-group mechanism used by InteractiveScene.
    # Cases 0 and 3 start at identical poses with different expected friction;
    # both must follow their own contact pair without cross-environment forces.
    from isaaclab import cloner
    cloner.filter_collisions(
        sim.stage,
        physicsscene_path=sim.cfg.physics_prim_path,
        collision_root_path='/World/CollisionGroups',
        prim_paths=[r['path'] for r in metadata['cases']], global_paths=[])
    from pawcerto.isaac.deepwbc_material_channels import author_collision_groups
    metadata['production_groups'] = author_collision_groups(sim.stage,metadata['channel_memberships'],
        metadata['box_roots'],metadata['terrain_paths'],old_collision_root='/World/CollisionGroups')
    metadata['isolation'] = 'Production single-membership contact groups replace old Lab groups; cases 0 and 3 exactly overlap'
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    sim.stage.GetRootLayer().Export(str(args.output/'fixture.usda'))
    bodies = [RigidObject(RigidObjectCfg(prim_path=r['path']+'/Mover', spawn=None)) for r in metadata['cases']]
    sim.reset()
    for _ in range(metadata['settle_steps']):
        sim.step(render=False)
        for b in bodies: b.update(.005)
    for b in bodies:
        velocity = torch.zeros((1,6), device=args.device); velocity[:,0] = 2.
        b.write_root_velocity_to_sim(velocity)
    rows = []
    for step in range(metadata['slide_steps']):
        sim.step(render=False)
        sample = []
        for b in bodies:
            b.update(.005)
            sample.append(b.data.root_state_w.torch[0].detach().cpu().numpy().copy())
        rows.append(sample)
    data = np.array(rows)
    if not np.isfinite(data).all(): raise ValueError('Nonfinite probe state')
    np.savez(args.output/'samples.npz', root_state=data)
    results=[]
    for i, case in enumerate(metadata['cases']):
        vx=data[:,i,7]; dv=-np.diff(vx)/.005
        mask=(vx[:-1]>.2)&(np.arange(len(dv))>3)
        mu=float(dv[mask].mean()/9.81) if mask.any() else None
        result=dict(pair=case['pair'], raw_friction=case['raw_friction'], expected_mu=case['expected_mu'],
            measured_mu=mu, samples=int(mask.sum()), final_z=float(data[-1,i,2]),
            max_abs_vz=float(abs(data[:,i,9]).max()), max_angular_speed=float(np.linalg.norm(data[:,i,10:13],axis=-1).max()))
        result['passed'] = result['final_z'] < -1. if case['pair']=='isolated' else (
            mu is not None and abs(mu-case['expected_mu'])<.035 and abs(result['final_z']-.05)<.01)
        results.append(result)
    passed=all(r['passed'] for r in results)
    (args.output/'analysis.json').write_text(json.dumps(dict(passed=passed, results=results,
        evidence='Physical motion of independent shared-geometry cubes; not articulation or training equivalence'),indent=2)+'\n')
    if not passed: raise AssertionError('Material/filter motion discriminator failed; see analysis.json')
except BaseException:
    traceback.print_exc()
    raise
finally:
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
