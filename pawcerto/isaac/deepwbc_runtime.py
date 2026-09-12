"""Go1/WidowX preparation and official Isaac Lab physics adapter.

Importing this module never imports or starts Isaac Sim. Native reaction and
reconstructed sensor wrenches retain explicit, approximate signal identities.
"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import torch

from pawcerto.methods.deepwbc.config import default_config, UPSTREAM_SHA
from pawcerto.methods.deepwbc.controller import DeepWBCController, POLICY_DOF_NAMES

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URDF = ROOT / 'third_party/deepwbc-reference/legged_gym/resources/robots/widowGo1/urdf/widowGo1.urdf'
DEFAULT_USD = ROOT / 'reference/isaac/go1_widowx/usd_path.txt'
SIM_DOF_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                      for joint in ('hip', 'thigh', 'calf')) + POLICY_DOF_NAMES[12:]
FEET = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
EE_BODY = 'wx250s_ee_gripper_link'
FORCE_MODES = ('native_joint_wrench', 'reconstructed_sensor_wrench')
FORCE_SIGNAL_EVIDENCE = {
    'native_joint_wrench': 'native incoming reaction; differs from original Gym sensor',
    'reconstructed_sensor_wrench': 'approximate PRE solver residual plus gravity; three-scene contact bits matched, Fz-squared mean absolute error about 10-13% of Gym mean in contact',
}


def source_tree(urdf=SOURCE_URDF):
    """Resolve meshes and make link identifiers legal USD names; no physics edit."""
    urdf = Path(urdf).resolve()
    tree = ET.parse(urdf)
    names = {link.get('name'): link.get('name').replace('/', '_') for link in tree.findall('link')}
    if len(set(names.values())) != len(names):
        raise ValueError('Link name normalization collides')
    for link in tree.findall('link'):
        link.set('name', names[link.get('name')])
    for joint in tree.findall('joint'):
        for tag in ('parent', 'child'):
            ref = joint.find(tag)
            ref.set('link', names[ref.get('link')])
    for mesh in tree.findall('.//mesh'):
        path = (urdf.parent / mesh.get('filename')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    return tree


def prepare_urdf(output, urdf=SOURCE_URDF):
    """CPU preparation only; retains mimic, inertia, collisions and protected links."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tree = source_tree(urdf)
    tree.write(output / 'source.urdf', encoding='utf-8', xml_declaration=True)
    joints = {j.get('name'): j for j in tree.findall('joint') if j.get('type') != 'fixed'}
    if set(joints) != set(SIM_DOF_NAMES):
        raise ValueError('Source articulation must retain original 18+2 joints')
    manifest = {
        'revision': UPSTREAM_SHA, 'source_urdf': str(Path(urdf).resolve()),
        'joint_names': list(SIM_DOF_NAMES),
        'limits': {n: {k: (float(joints[n].find('limit').get(k)) if joints[n].find('limit').get(k) is not None else None) for k in ('lower','upper','effort','velocity')} for n in SIM_DOF_NAMES},
        'protected_fixed_joints': [j.get('name') for j in tree.findall('joint') if j.get('dont_collapse') == 'true'],
        'source_mass': sum(float(x.get('value')) for x in tree.findall('link/inertial/mass')),
        'source_collisions': len(tree.findall('link/collision')),
        'body_name_map': {l.get('name'): l.get('name').replace('/', '_') for l in ET.parse(urdf).findall('link')},
        'mimic': {'joint': 'widow_right_finger', 'reference': 'widow_left_finger', 'multiplier': -1., 'offset': 0.},
    }
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return tree, manifest


def lab_initial_joint_positions(config, limits):
    """Legal construction pose only; original task/controller reset stays zero.

    Lab validates default pose before the task can apply source reset writes.
    Original passive finger defaults are outside their own URDF limits.
    """
    positions = dict(config['init_state']['default_joint_angles'])
    for name in ('widow_left_finger', 'widow_right_finger'):
        positions[name] = min(max(positions[name],limits[name]['lower']),limits[name]['upper'])
    return positions


def link_paths(tree, root='{ENV_REGEX_NS}/Robot/Geometry/base'):
    paths = {'base': root}
    pending = list(tree.findall('joint'))
    while pending:
        advanced = False
        for joint in pending[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent in paths:
                paths[child] = paths[parent] + '/' + child
                pending.remove(joint)
                advanced = True
        if not advanced:
            raise ValueError('Disconnected or cyclic source link hierarchy')
    return paths


def material_properties_for_friction(materials, friction):
    """Build the documented [N,shapes,static/dynamic/restitution] payload.

    Reject observed unsafe inputs before the native setter can segfault. The
    original negative-friction recipe is not silently changed into another one.
    """
    if materials.ndim != 3 or materials.shape[-1] != 3 or friction.shape != (materials.shape[0],1):
        raise ValueError('Expected material[N,shapes,3] and friction[N,1]')
    friction = friction.to(device=materials.device,dtype=torch.float32)
    if not torch.isfinite(friction).all() or torch.any(friction < 0):
        raise ValueError(f'PhysX requires finite nonnegative friction; source samples have minimum {friction.min().item()}')
    result = materials.to(dtype=torch.float32).clone().contiguous()
    result[:,:,:2] = friction[:,None]
    if not torch.isfinite(result).all():
        raise ValueError('PhysX material payload contains nonfinite coefficients')
    return result


def transform_wrench_to_link(force, torque, joint_pos_b, joint_quat_b):
    """Rotate child-anchor wrench to link axes and move moment to link origin.

    Quaternions xyzw; r is link-origin to joint-anchor in link coordinates.
    This changes reference frame only, not the native reaction-force semantics.
    """
    def rotate(value):
        xyz, w = joint_quat_b[..., :3], joint_quat_b[..., 3:]
        cross = torch.cross(xyz.expand_as(value),value,dim=-1)
        return value + 2 * (w * cross + torch.cross(xyz.expand_as(value),cross,dim=-1))
    f, t = rotate(force), rotate(torque)
    return torch.cat((f,t+torch.cross(joint_pos_b.expand_as(f),f,dim=-1)),dim=-1)


class Go1WidowXIsaac:
    """One official PhysX step per step_control; task owner provides terrain.

    Must be constructed after AppLauncher. The terrain config and collision
    filter paths are explicit so this adapter cannot silently replace the task's
    original terrain. The reconstructed signal remains an explicit approximation.
    """
    def __init__(self, *, terrain_cfg, terrain_contact_paths, num_envs=1,
                 device='cuda:0', usd_path=DEFAULT_USD, config=None, force_signal='native_joint_wrench', env_origins=None):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg
        from isaaclab.utils import configclass
        from isaaclab_physx.physics import PhysxCfg
        self.config = default_config() if config is None else config
        if force_signal not in FORCE_MODES:
            raise ValueError(f'Unknown DeepWBC force signal: {force_signal}')
        self.force_signal = force_signal
        self.device, self.num_envs = device, num_envs
        self.dt = self.config['sim']['dt']
        self.joint_names = list(SIM_DOF_NAMES)
        path = Path(usd_path)
        if path.suffix == '.txt':
            path = Path(path.read_text().strip())
        prepared = next((p for p in (path.parent, path.parent.parent) if (p / 'source-manifest.json').exists()), None)
        if prepared is None:
            raise FileNotFoundError('Converted USD requires adjacent source-manifest.json')
        manifest = json.loads((prepared / 'source-manifest.json').read_text())
        tree = ET.parse(prepared / 'merged.urdf')
        self.torque_limits = torch.tensor([manifest['limits'][n]['effort'] for n in self.joint_names],device=device)
        velocities = {n: manifest['limits'][n]['velocity'] for n in self.joint_names}
        self.controller = DeepWBCController(num_envs, self.joint_names, self.torque_limits, self.config, device)
        asset = self.config['asset']
        physx = self.config['sim']['physx']
        # Collision offsets are authored in shared geometry at conversion time.
        # Never uninstance geometry to author clone-local collision opinions.
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            terrain = terrain_cfg
            robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
                spawn=sim_utils.UsdFileCfg(usd_path=str(path), activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                        linear_damping=asset['linear_damping'],angular_damping=asset['angular_damping'],
                        max_linear_velocity=asset['max_linear_velocity'],max_angular_velocity=math.degrees(asset['max_angular_velocity']),
                        max_depenetration_velocity=physx['max_depenetration_velocity']),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=physx['num_position_iterations'],
                        solver_velocity_iteration_count=physx['num_velocity_iterations'])),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(self.config['init_state']['pos']),
                    joint_pos=lab_initial_joint_positions(self.config,manifest['limits'])),
                actuators={'effort': ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,
                    effort_limit_sim={n:manifest['limits'][n]['effort'] for n in self.joint_names},
                    velocity_limit_sim=velocities,armature=asset['armature'],friction=0.,dynamic_friction=0.,viscous_friction=0.)})
            joint_wrench = JointWrenchSensorCfg(prim_path='{ENV_REGEX_NS}/Robot',update_period=0.)
            box = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Box',spawn=sim_utils.CuboidCfg(
                size=(self.config['box']['box_size'],)*3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(density=1000.),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=physx['contact_offset'],rest_offset=physx['rest_offset'])),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,.2,self.config['box']['box_env_origins_z'])))
        scene_cfg = SceneCfg(num_envs=num_envs,env_spacing=4.)
        paths = link_paths(tree)
        self.contact_names = [l.get('name') for l in tree.findall('link') if l.find('collision') is not None]
        for name in self.contact_names:
            filters = list(terrain_contact_paths) + ['{ENV_REGEX_NS}/Box'] + [paths[n] for n in self.contact_names if n != name]
            setattr(scene_cfg,'contact_'+name,ContactSensorCfg(prim_path=paths[name],update_period=0.,
                filter_prim_paths_expr=filters,track_friction_forces=True,max_contact_data_count_per_prim=32))
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt,device=device,
            gravity=tuple(self.config['sim']['gravity']),physics=PhysxCfg(solver_type=physx['solver_type'],bounce_threshold_velocity=physx['bounce_threshold_velocity'],enable_external_forces_every_iteration=False)))
        self.scene = InteractiveScene(scene_cfg)
        if any(self.config['terrain'][k] != 1. for k in ('static_friction','dynamic_friction')):
            raise ValueError('Signed-friction pair representation requires original terrain friction 1')
        from pawcerto.isaac.deepwbc_material_channels import author_material_channels
        self.material_channel_evidence = author_material_channels(self.sim.stage,
            [p+'/Robot' for p in self.scene.env_prim_paths],
            [p+'/Box' for p in self.scene.env_prim_paths], terrain_contact_paths)
        self.sim.reset()
        self.env_origins = torch.zeros(num_envs,3,device=device) if env_origins is None else torch.as_tensor(env_origins,device=device).clone()
        if self.env_origins.shape != (num_envs,3):
            raise ValueError('Expected one world terrain origin per environment')
        self.scene.update(self.dt)
        self.robot = self.scene['robot']
        from pawcerto.isaac.deepwbc_material_channels import discover_channel_masks
        import warp as wp
        initial_materials = wp.to_torch(self.robot.root_view.get_material_properties())
        self._material_channel_masks = discover_channel_masks(initial_materials,
            self.material_channel_evidence['collision_count'])
        # f=0 labels are read before any physics step, then restore source f=1.
        self._write_material_channels(torch.ones(num_envs,1))
        if set(self.robot.joint_names) != set(self.joint_names):
            raise ValueError('Imported USD does not preserve original 20 named DOFs')
        self.joint_ids = [self.robot.joint_names.index(n) for n in self.joint_names]
        self.dof_pos_limits = self.robot.data.joint_pos_limits.torch[0,self.joint_ids].clone()
        self.body_names = self.robot.body_names
        self.ee_id = self.body_names.index(EE_BODY)
        self.base_id = self.body_names.index('base')
        self.feet_indices = [self.body_names.index(n) for n in FEET]
        self._wrench_feet_ids = [self.scene['joint_wrench'].body_names.index(n) for n in FEET]
        self._original_masses = self.robot.data.body_mass.torch.clone()
        self._original_inertias = self.robot.data.body_inertia.torch.clone()
        self._original_coms = self.robot.data.body_com_pose_b.torch.clone()
        self._original_box_masses = self.scene['box'].data.body_mass.torch.clone()
        self._original_box_inertias = self.scene['box'].data.body_inertia.torch.clone()
        self.material_channel_evidence['native_total_mass'] = self._original_masses.sum(1).cpu().tolist()
        self.material_channel_evidence['native_shape_count'] = initial_materials.shape[1]
        print('DeepWBC material channels:', self.material_channel_evidence, flush=True)
        self._joint_frame_pos, self._joint_frame_quat = self._foot_joint_frames()
        self.motor_strength = torch.ones(num_envs,18,device=device)
        self.time = torch.zeros(num_envs,device=device)
        self.last_torques = torch.zeros(num_envs,20,device=device)
        self._solver_force = None
        self._sensor_wrenches = torch.zeros(num_envs,4,6,device=device)

    def _native_tensor(self, method):
        import warp as wp
        value = getattr(self.robot.root_view, method)()
        return (value if isinstance(value,torch.Tensor) else wp.to_torch(value)).to(self.device).clone()

    def _initialize_sensor_reconstruction(self):
        """Read actual properties after creation randomization, before first PD step."""
        from .solver_force import StockSolverForce
        view = self.robot.root_view
        if view.shared_metatype.fixed_base:
            raise ValueError('Sensor reconstruction requires the original floating base')
        if (list(view.shared_metatype.link_names) != self.body_names
                or list(view.shared_metatype.dof_names) != list(self.robot.joint_names)):
            raise ValueError('Sensor PRE kinematics must use the native body and DOF order')
        parents = [-1] + [view.shared_metatype.link_parent_indices[n] for n in self.body_names[1:]]
        def field(name, radians=False):
            values = [[self.sim.stage.GetPrimAtPath(path).GetAttribute('physxRigidBody:'+name).Get()
                       for path in paths] for paths in view.link_paths]
            if any(value is None for row in values for value in row):
                raise ValueError('Missing composed body property '+name)
            result = torch.tensor(values,dtype=torch.float32,device=self.device)
            return torch.deg2rad(result) if radians else result
        self._solver_force = StockSolverForce(parents,self._native_tensor('get_masses'),dt=self.dt,
            body_inertias=self._native_tensor('get_inertias').reshape(self.num_envs,len(parents),3,3),
            dof_max_velocity=self._native_tensor('get_dof_max_velocities'),
            body_max_linear_velocity=field('maxLinearVelocity'),body_max_angular_velocity=field('maxAngularVelocity',True),
            body_linear_damping=field('linearDamping'),body_angular_damping=field('angularDamping'),
            armature=self._native_tensor('get_dof_armatures'),disable_gravity=self._native_tensor('get_disable_gravities'),
            retain_accelerations=field('retainAccelerations'),gyroscopic_forces=field('enableGyroscopicForces'),
            external_forces_every_iteration=self.sim.cfg.physics.enable_external_forces_every_iteration)
        self._sensor_com_b = self.robot.data.body_com_pose_b.torch[...,:3].clone()
        self._sensor_gravity = self._solver_force.masses[...,None]*torch.tensor(self.config['sim']['gravity'],device=self.device)

    def _sensor_pre_step(self):
        from isaaclab.utils.math import matrix_from_quat
        if self._solver_force is None:
            self._initialize_sensor_reconstruction()
        # Refresh COM kinematics before state-dependent engine buffers, exactly
        # as in the real probe. Public C must precede COM velocity readback.
        com_position = self.robot.data.body_com_pose_w.torch[...,:3].clone()
        methods = dict(mass_matrix='get_generalized_mass_matrices',gravity='get_gravity_compensation_forces',
            coriolis='get_coriolis_and_centrifugal_compensation_forces',jacobian='get_jacobians',
            com_velocity='get_link_velocities',dof_velocity='get_dof_velocities',actuation='get_dof_actuation_forces')
        pre = {name:self._native_tensor(method) for name,method in methods.items()}
        rotation = matrix_from_quat(self.robot.data.body_link_quat_w.torch).clone()
        pre.update(com_position=com_position,link_rotation=rotation,
                   external_wrench_terms=torch.zeros(self.num_envs,device=self.device))
        predicted,_,_ = self._solver_force.predict(**pre)
        return predicted,rotation

    def joints(self):
        return (self.robot.data.joint_pos.torch[:,self.joint_ids],
                self.robot.data.joint_vel.torch[:,self.joint_ids])

    def push_actions(self, policy_actions):
        return self.controller.push(policy_actions)

    def step_control(self):
        """Explicit PD once per physics step; push_actions once per policy step."""
        q,qd = self.joints()
        self.last_torques = self.controller.torques(q,qd,self.motor_strength)
        self.robot.set_joint_effort_target(self.last_torques,joint_ids=self.joint_ids)
        if self.force_signal == 'reconstructed_sensor_wrench':
            for name in ('instantaneous_wrench_composer','permanent_wrench_composer'):
                if getattr(self.robot,name).active:
                    raise ValueError('Sensor reconstruction has unmodeled external wrench: '+name)
        self.scene.write_data_to_sim()
        if self.force_signal == 'reconstructed_sensor_wrench':
            predicted,rotation = self._sensor_pre_step()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        if self.force_signal == 'reconstructed_sensor_wrench':
            self._sensor_wrenches[:] = self._solver_force.approximate_sensor_wrench(
                self._native_tensor('get_link_accelerations'),predicted,link_rotation=rotation,
                com_offset_b=self._sensor_com_b,applied_force_w=self._sensor_gravity)[:,self.feet_indices]
        self.time += self.dt
        return self.state()

    def reset(self, ids, root_states, joint_pos, joint_vel):
        """Task-generated local-world pose (xyzw), link velocity, and joint state."""
        from isaaclab.utils.math import quat_apply
        pose = root_states[:,:7].clone()
        pose[:,:3] += self.env_origins[ids]
        velocity = root_states[:,7:13].clone()
        com = self.robot.data.body_com_pose_b.torch[ids,self.base_id,:3]
        velocity[:,:3] += torch.cross(velocity[:,3:],quat_apply(pose[:,3:],com),dim=-1)
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.reset(ids)
        self.controller.reset(ids)
        self.time[ids] = 0
        self.last_torques[ids] = 0
        self._sensor_wrenches[ids] = 0
        self.sim.forward()
        self.scene.update(self.dt)

    def state(self):
        d = self.robot.data
        q,qd = self.joints()
        contacts = torch.zeros(self.num_envs,len(self.body_names),3,device=self.device)
        for name in self.contact_names:
            data = self.scene['contact_'+name].data
            contacts[:,self.body_names.index(name)] = data.net_normal_forces_w.torch[:,0] + data.friction_force_matrix_w.torch[:,0].sum(1)
        if self.force_signal == 'reconstructed_sensor_wrench':
            foot_joint_wrenches = self._sensor_wrenches.clone()
        else:
            wrench = self.scene['joint_wrench'].data
            foot_joint_wrenches = transform_wrench_to_link(
                wrench.force.torch[:,self._wrench_feet_ids],wrench.torque.torch[:,self._wrench_feet_ids],
                self._joint_frame_pos,self._joint_frame_quat)
        root = torch.cat((d.root_link_pos_w.torch-self.env_origins,d.root_link_quat_w.torch,
                          d.root_link_lin_vel_w.torch,d.root_link_ang_vel_w.torch),dim=-1)
        return dict(root_states=root,dof_pos=q,dof_vel=qd,torques=self.last_torques,
            actions=self.controller.actions,latest_sim_actions=self.controller.history[:,-1],
            ee_pos=d.body_link_pos_w.torch[:,self.ee_id]-self.env_origins,
            ee_orn=d.body_link_quat_w.torch[:,self.ee_id],
            ee_lin_vel=d.body_link_lin_vel_w.torch[:,self.ee_id],
            ee_ang_vel=d.body_link_ang_vel_w.torch[:,self.ee_id],
            contact_forces=contacts,foot_wrenches=foot_joint_wrenches,
            force_signal=self.force_signal,time=self.time.clone())

    def _foot_joint_frames(self):
        """Read actual imported child anchors; no assumed link/COM identity."""
        from pxr import Usd, UsdPhysics
        stage = self.sim.stage
        frames = {}
        root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = joint.GetBody1Rel().GetTargets()
                if targets and targets[0].name in FEET:
                    pos = joint.GetLocalPos1Attr().Get()
                    quat = joint.GetLocalRot1Attr().Get()
                    frames[targets[0].name] = (list(pos), [*quat.GetImaginary(),quat.GetReal()])
        if set(frames) != set(FEET):
            raise ValueError('USD must preserve four incoming foot joint frames')
        return (torch.tensor([frames[n][0] for n in FEET],device=self.device),
                torch.tensor([frames[n][1] for n in FEET],device=self.device))

    def _write_material_channels(self, friction, materials=None):
        import warp as wp
        from pawcerto.isaac.deepwbc_material_channels import channel_material_properties
        if materials is None:
            materials = channel_material_properties(
                wp.to_torch(self.robot.root_view.get_material_properties()),friction,self._material_channel_masks)
        self.robot.root_view.set_material_properties(wp.from_torch(materials),
            wp.from_torch(torch.arange(self.num_envs,dtype=torch.int32)))
        actual = wp.to_torch(self.robot.root_view.get_material_properties())
        if not torch.equal(actual, materials):
            raise ValueError('Native material setter did not retain signed-friction pair coefficients')
        self.material_channel_evidence['last_source_friction'] = friction.detach().cpu().reshape(-1).tolist()

    def apply_creation_randomization(self, mass_params, friction, motor_strength, box_added_mass):
        """Apply caller-sampled original deltas once, using supported public setters.

        Base means original root body0 (the mount body after fixed merging), not
        an undocumented change to the torso. Inertias scale with mass at fixed geometry, corresponding to the original
        recomputeInertia request; exact Gym recomputation remains unverified.
        """
        import warp as wp
        from pawcerto.isaac.deepwbc_material_channels import channel_material_properties
        materials = channel_material_properties(
            wp.to_torch(self.robot.root_view.get_material_properties()),friction,self._material_channel_masks)
        masses = self._original_masses.clone()
        masses[:,self.base_id] += mass_params[:,0]
        masses[:,self.ee_id] += mass_params[:,4]
        if torch.any(masses <= 0):
            raise ValueError('Creation randomization produced nonpositive source body mass')
        self.robot.set_masses_index(masses=masses)
        self.robot.set_inertias_index(inertias=self._original_inertias*(masses/self._original_masses)[...,None])
        coms = self._original_coms.clone()
        coms[:,self.base_id,:3] += mass_params[:,1:4]
        self.robot.set_coms_index(coms=coms)
        self._write_material_channels(friction, materials)
        self.motor_strength[:] = motor_strength
        box_mass = self._original_box_masses + box_added_mass.reshape(self.num_envs,1)
        self.scene['box'].set_masses_index(masses=box_mass)
        self.scene['box'].set_inertias_index(inertias=self._original_box_inertias*(box_mass/self._original_box_masses)[...,None])
        self._solver_force = None

    def reset_box(self, ids, positions):
        box = self.scene['box']
        pose = torch.cat((box.data.root_link_pos_w.torch[ids],box.data.root_link_quat_w.torch[ids]),dim=-1).clone()
        pose[:,:2] = positions[:,:2] + self.env_origins[ids,:2]
        box.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        # Original reset writes box position only; keep its orientation/twist.

    def push_base_velocity_xy(self, velocity):
        from isaaclab.utils.math import quat_apply
        d = self.robot.data
        full = torch.cat((d.root_link_lin_vel_w.torch,d.root_link_ang_vel_w.torch),dim=-1).clone()
        full[:,:2] = velocity
        com = d.body_com_pose_b.torch[:,self.base_id,:3]
        full[:,:3] += torch.cross(full[:,3:],quat_apply(d.root_link_quat_w.torch,com),dim=-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=full)
