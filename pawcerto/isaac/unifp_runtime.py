"""Official Isaac Lab B2/Z1 runtime; original joint ordering and xyzw quaternions."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_physx.physics import PhysxCfg
from pawcerto.robots.urdf import read_joint_limits
from pawcerto.methods.unifp.terrain import make_terrain_cfg
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = ROOT / 'reference/isaac/b2_z1/usd_path.txt'
JOINT_NAMES = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')] + ['z1_waist','z1_shoulder','z1_elbow','z1_wrist_angle','z1_forearm_roll','z1_wrist_rotate','z1_jointGripper']

@sim_utils.clone
def spawn_unifp(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Apply source physics overrides once before Lab clones the articulation."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    instances = set()
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.HasAPI(UsdPhysics.CollisionAPI) and child.IsInstanceProxy():
            # A prepared shared geometry layer can already carry these values.
            # Keep its instances shared; only legacy assets need local overrides.
            if (child.GetAttribute('physxCollision:contactOffset').Get() is not None
                and math.isclose(child.GetAttribute('physxCollision:contactOffset').Get(), cfg.collision_props.contact_offset, rel_tol=1e-6)
                and child.GetAttribute('physxCollision:restOffset').Get() == cfg.collision_props.rest_offset):
                continue
            parent = child.GetParent()
            while not parent.IsInstance():
                parent = parent.GetParent()
            instances.add(str(parent.GetPath()))
    for instance in instances:
        sim_utils.make_uninstanceable(instance, stage=stage)
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            sim_utils.modify_collision_properties(str(child.GetPath()), cfg.collision_props, stage=stage)
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            if child.IsInstance():
                # Lab's nested modifier skips instance roots, even when the
                # root itself is a rigid body (imported gripperMover). Author
                # public USD physics attributes on that root without expanding
                # or copying its referenced geometry.
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                for attribute, value in (
                    ('DisableGravity', cfg.rigid_props.disable_gravity),
                    ('LinearDamping', cfg.rigid_props.linear_damping),
                    ('AngularDamping', cfg.rigid_props.angular_damping),
                    ('MaxLinearVelocity', cfg.rigid_props.max_linear_velocity),
                    ('MaxAngularVelocity', cfg.rigid_props.max_angular_velocity),
                    ('MaxDepenetrationVelocity', cfg.rigid_props.max_depenetration_velocity)):
                    getattr(api, 'Create'+attribute+'Attr')(value)
            else:
                sim_utils.modify_rigid_body_properties(str(child.GetPath()), cfg.rigid_props, stage=stage)
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
    print('[UniFP] Source articulation physics overrides complete', flush=True)
    return prim


class B2Z1Isaac:
    def __init__(self, num_envs, device, usd_path, config, terrain):
        self.num_envs, self.device, self.config, self.terrain = num_envs, device, config, terrain
        self.dt = config.sim.dt
        path = Path(usd_path)
        if path.suffix == '.txt':
            path = Path(path.read_text().strip())
        tree = ET.parse(path.parent.parent / 'merged.urdf')
        self.joint_names = list(JOINT_NAMES)
        limits = read_joint_limits(path.parent.parent / 'merged.urdf', JOINT_NAMES)
        self.dof_pos_limits = torch.tensor([[limits[n][k] for k in ('lower','upper')] for n in JOINT_NAMES],device=device)
        self.dof_vel_limits = torch.tensor([limits[n]['velocity'] for n in JOINT_NAMES],device=device)
        self.torque_limits = torch.tensor([limits[n]['effort'] for n in JOINT_NAMES],device=device)
        ground, ground_paths = make_terrain_cfg(terrain)
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=config.sim.physx.contact_offset,rest_offset=config.sim.physx.rest_offset)
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            terrain_asset = ground
            robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
                spawn=sim_utils.UsdFileCfg(func=spawn_unifp,usd_path=str(path),activate_contact_sensors=True,
                    collision_props=collision,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=config.asset.disable_gravity,
                        linear_damping=config.asset.linear_damping,angular_damping=config.asset.angular_damping,
                        max_linear_velocity=config.asset.max_linear_velocity,max_angular_velocity=math.degrees(config.asset.max_angular_velocity),
                        max_depenetration_velocity=config.sim.physx.max_depenetration_velocity),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=config.asset.self_collisions==0,
                        solver_position_iteration_count=config.sim.physx.num_position_iterations,
                        solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations)),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(config.init_state.pos),joint_pos=config.init_state.default_joint_angles),
                actuators={'effort':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,
                    effort_limit_sim=1e9,velocity_limit_sim={n:limits[n]['velocity'] for n in JOINT_NAMES},
                    armature=config.asset.armature,friction=0.,dynamic_friction=0.,viscous_friction=0.)})
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt,device=device,
            physics=PhysxCfg(enable_external_forces_every_iteration=False)))
        cfg = SceneCfg(num_envs=num_envs,env_spacing=0.)
        paths = {'base_link':'{ENV_REGEX_NS}/Robot/Geometry/base_link'}
        pending = list(tree.findall('joint'))
        while pending:
            progressed = False
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent in paths:
                    child = joint.find('child').get('link')
                    paths[child] = paths[parent]+'/'+child
                    pending.remove(joint); progressed = True
            if not progressed:
                raise ValueError('Disconnected imported URDF tree')
        self.contact_names = [l.get('name') for l in tree.findall('link') if l.find('collision') is not None]
        for name in self.contact_names:
            filters = ground_paths+[paths[other] for other in self.contact_names if other!=name]
            setattr(cfg,'contact_'+name,ContactSensorCfg(prim_path=paths[name],update_period=0.,
                filter_prim_paths_expr=filters,track_friction_forces=True,max_contact_data_count_per_prim=64))
        self.scene = InteractiveScene(cfg)
        self.sim.reset(); self.scene.update(self.dt)
        self.robot = self.scene['robot']
        self.body_names = self.robot.body_names
        self.joint_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        self.base_id = self.body_names.index('base_link')
        self.ee_id = self.body_names.index(config.asset.gripper_name)
        self.feet_indices = [self.body_names.index(leg+'_foot') for leg in ('FL','FR','RL','RR')]
        self.time = torch.zeros(num_envs,device=device)
        self.mass_params_tensor = torch.zeros(num_envs,22,device=device)
        self._randomize_physics()
        self.refresh()

    def _randomize_physics(self):
        import warp as wp
        cfg = self.config.domain_rand
        masses = self.robot.data.body_mass.torch.clone()
        old_mass = masses.clone()
        inertias = self.robot.data.body_inertia.torch.clone()
        coms = self.robot.data.body_com_pose_b.torch.clone()
        # Original Gym asset/property probe confirms props[0]=base_link and
        # props[1]=FL_hip. Preserve this surprising actual producer.
        com_id = self.body_names.index('FL_hip')
        for i in range(self.num_envs):
            mass = np.random.uniform(*cfg.added_mass_range) if cfg.randomize_base_mass else 0.
            grip = np.random.uniform(*cfg.gripper_added_mass_range) if cfg.randomize_gripper_mass else 0.
            com = np.random.uniform([cfg.added_com_range_x[0],cfg.added_com_range_y[0],cfg.added_com_range_z[0]],
                [cfg.added_com_range_x[1],cfg.added_com_range_y[1],cfg.added_com_range_z[1]]) if cfg.randomize_base_com else np.zeros(3)
            if cfg.randomize_leg_mass:
                raise ValueError('Original UniFP disables leg-mass randomization; changed body-index recipe is not implemented')
            masses[i,self.base_id] += mass
            masses[i,self.ee_id] += grip
            coms[i,com_id,:3] += torch.tensor(com,device=self.device,dtype=coms.dtype)
            self.mass_params_tensor[i,:5] = torch.tensor([mass,*com,grip],device=self.device)
        self.robot.set_masses_index(masses=masses)
        self.robot.set_inertias_index(inertias=inertias*(masses/old_mass)[...,None])
        self.robot.set_coms_index(coms=coms)
        buckets = torch.randint(0,256,(self.num_envs,1))
        values = torch.rand(256,1)*(cfg.friction_range[1]-cfg.friction_range[0])+cfg.friction_range[0]
        self.friction_coeffs_tensor = (values[buckets].squeeze(-1) if cfg.randomize_friction else torch.ones(self.num_envs,1)).to(self.device)
        materials = wp.to_torch(self.robot.root_view.get_material_properties()).clone()
        materials[:,:,0:2] = self.friction_coeffs_tensor.to(materials.device)[:,:,None]
        self.robot.root_view.set_material_properties(wp.from_torch(materials),wp.from_torch(torch.arange(self.num_envs,dtype=torch.int32)))

    def joints(self):
        return self.robot.data.joint_pos.torch[:,self.joint_ids],self.robot.data.joint_vel.torch[:,self.joint_ids]

    def refresh(self):
        d = self.robot.data
        self.root_states = torch.cat((d.root_link_pos_w.torch,d.root_link_quat_w.torch,
            d.root_link_lin_vel_w.torch,d.root_link_ang_vel_w.torch),dim=-1)
        self.rigid_state = torch.cat((d.body_link_pos_w.torch,d.body_link_quat_w.torch,
            d.body_link_lin_vel_w.torch,d.body_link_ang_vel_w.torch),dim=-1)
        self.contact_forces = torch.zeros(self.num_envs,len(self.body_names),3,device=self.device)
        for name in self.contact_names:
            d = self.scene['contact_'+name].data
            self.contact_forces[:,self.body_names.index(name)] = d.net_normal_forces_w.torch[:,0]+d.friction_force_matrix_w.torch[:,0].sum(dim=1)

    def write_reset(self, ids, roots, q, qd):
        self.robot.write_root_pose_to_sim_index(root_pose=roots[:,:7],env_ids=ids)
        self.write_root_velocity(ids,roots[:,7:13],roots[:,3:7])
        self.robot.write_joint_position_to_sim_index(position=q,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.reset(ids)
        self.time[ids] = 0.

    def write_root_velocity(self, ids, velocity, quaternion):
        com = self.robot.data.body_com_pose_b.torch[ids,self.base_id,:3]
        value = velocity.clone()
        value[:,:3] += torch.cross(value[:,3:],quat_apply(quaternion,com),dim=-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=value,env_ids=ids)

    def step_control(self, torques, forces):
        self.robot.set_joint_effort_target(torques,joint_ids=self.joint_ids)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(forces=forces,torques=torch.zeros_like(forces),is_global=True)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.time += self.dt
