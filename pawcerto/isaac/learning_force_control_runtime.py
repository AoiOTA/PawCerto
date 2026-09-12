"""Official Isaac Lab B1+Z1 physics path, one control substep per call."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_physx.physics import PhysxCfg
from pawcerto.robots.b1_z1 import JOINT_NAMES, DEFAULT_POS, P_GAINS, D_GAINS
from pawcerto.isaac.learning_force_control_terrain import ForceControlTerrain
from pawcerto.methods.learning_force_control.training.config import default_config, resolve_config
from pawcerto.robots.urdf import read_joint_limits
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_USD = ROOT / 'reference/isaac/b1_z1/usd_path.txt'


@sim_utils.clone
def spawn_learning_force_control(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Apply source physics overrides once before Lab clones the articulation."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            sim_utils.modify_collision_properties(str(child.GetPath()), cfg.collision_props, stage=stage)
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            # Instance roots are editable; the generic recursive modifier skips
            # them (observed for gripperMover). Author body properties directly
            # without deinstancing shared collision geometry.
            rigid=PhysxSchema.PhysxRigidBodyAPI.Apply(child)
            rigid.CreateDisableGravityAttr(cfg.rigid_props.disable_gravity)
            rigid.CreateLinearDampingAttr(cfg.rigid_props.linear_damping)
            rigid.CreateAngularDampingAttr(cfg.rigid_props.angular_damping)
            rigid.CreateMaxLinearVelocityAttr(cfg.rigid_props.max_linear_velocity)
            rigid.CreateMaxAngularVelocityAttr(cfg.rigid_props.max_angular_velocity)
            rigid.CreateMaxDepenetrationVelocityAttr(cfg.rigid_props.max_depenetration_velocity)
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
    print('[Learning Force Control] Source articulation physics overrides complete', flush=True)
    return prim


class B1Z1Isaac:
    def __init__(self, num_envs=1, device='cuda:0', usd_path=DEFAULT_USD, config=None):
        self.dt = .005
        self.device, self.num_envs = device, num_envs
        self.cfg = resolve_config(default_config()) if config is None else config
        self.terrain = ForceControlTerrain(self.cfg, device)
        self.joint_names = list(JOINT_NAMES)
        self.default_pos = torch.tensor(DEFAULT_POS, device=device)
        path = Path(usd_path)
        if path.suffix == '.txt':
            path = Path(path.read_text().strip())
        tree = ET.parse(path.parent.parent / 'merged.urdf')
        limits = read_joint_limits(path.parent.parent / 'merged.urdf', JOINT_NAMES)
        velocity = {n: limits[n]['velocity'] for n in JOINT_NAMES}
        self.dof_pos_limits = torch.tensor([[limits[n][k] for k in ('lower','upper')] for n in JOINT_NAMES],device=device)
        self.torque_limits = torch.tensor([limits[n]['effort'] for n in JOINT_NAMES], device=device)
        kp = dict(zip(JOINT_NAMES, self.cfg.commands.p_gains_legs + self.cfg.commands.p_gains_arm))
        kd = dict(zip(JOINT_NAMES, self.cfg.commands.d_gains_legs + self.cfg.commands.d_gains_arm))
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=.01, rest_offset=0.)
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
                spawn=sim_utils.UsdFileCfg(func=spawn_learning_force_control, usd_path=str(path), activate_contact_sensors=True,
                    collision_props=collision,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,
                        max_linear_velocity=1000.,max_angular_velocity=math.degrees(1000.),max_depenetration_velocity=1.),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.65),joint_pos=dict(zip(JOINT_NAMES,DEFAULT_POS))),
                actuators={'mixed': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=kp,damping=kd,
                    effort_limit_sim={n:limits[n]["effort"] for n in JOINT_NAMES},velocity_limit_sim=velocity,armature=0.,friction=0.,dynamic_friction=0.,viscous_friction=0.)})
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device=device,
            physics=PhysxCfg(enable_external_forces_every_iteration=False)))
        self.terrain.spawn()
        cfg = SceneCfg(num_envs=num_envs,env_spacing=5.)
        paths = {'base': '{ENV_REGEX_NS}/Robot/Geometry/base'}
        pending = list(tree.findall('joint'))
        while pending:
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent in paths:
                    child = joint.find('child').get('link')
                    paths[child] = paths[parent] + '/' + child
                    pending.remove(joint)
        self.contact_names = [link.get('name') for link in tree.findall('link') if link.find('collision') is not None]
        for name in self.contact_names:
            # PhysX exposes tangential force only for configured sensor/filter pairs.
            # Each body expression pairs corresponding environments; the static
            # ground is shared. Include self-collision partners as well as ground.
            filters = self.terrain.collision_paths + [paths[other] for other in self.contact_names if other != name]
            setattr(cfg,'contact_' + name,ContactSensorCfg(prim_path=paths[name],update_period=0.,
                filter_prim_paths_expr=filters,track_friction_forces=True,max_contact_data_count_per_prim=32))
        print(f'[Learning Force Control] Constructing {num_envs} environments', flush=True)
        self.scene = InteractiveScene(cfg)
        print('[Learning Force Control] Scene constructed; initializing physics', flush=True)
        self.sim.reset()
        self.scene.update(self.dt)
        self.robot = self.scene['robot']
        self.body_names = self.robot.body_names
        self.joint_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        self.ee_id = self.body_names.index('link06')
        # Original Gym find('gripperStator')=-1 indexes its last body.
        # Bind that proven behavior by name, independent of Lab body ordering.
        self.ee_measurement_id = self.body_names.index('gripperMover')
        self.feet_indices = [self.body_names.index(leg+'_foot') for leg in ('FL','FR','RL','RR')]
        self.base_id = self.body_names.index('base')
        self._original_masses = self.robot.data.body_mass.torch.clone()
        self._original_inertias = self.robot.data.body_inertia.torch.clone()
        self.set_base_com(torch.zeros((num_envs,3),device=device))
        self._gravity = torch.tensor([0.,0.,-9.81],device=device)
        self.time = torch.zeros(num_envs,device=device)
        self.dof_vel_limits = torch.tensor([limits[n]['velocity'] for n in JOINT_NAMES],device=device)
        self.env_origins = self.terrain.env_origins[8,8].expand(num_envs,-1).clone()
        root = torch.zeros(num_envs,13,device=device);root[:,:3]=self.env_origins;root[:,2]+=.65;root[:,6]=1.
        self.write_reset(torch.arange(num_envs,device=device), self.default_pos.expand(num_envs,-1), root)

    def write_reset(self, ids, dof_pos, root_state, dof_vel=None):
        pose=root_state[:,:7].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        velocity=root_state[:,7:13].clone()
        com=self.robot.data.body_com_pose_b.torch[ids,self.base_id,:3]
        velocity[:,:3]+=torch.cross(velocity[:,3:],quat_apply(pose[:,3:],com),dim=-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=dof_pos,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(dof_pos) if dof_vel is None else dof_vel,joint_ids=self.joint_ids,env_ids=ids)
        self.robot.reset(ids);self.time[ids]=0
        self.sim.forward();self.scene.update(self.dt)
        return self.read_state()

    def write_root_state(self, ids, root_state):
        pose=root_state[:,:7].clone()
        velocity=root_state[:,7:13].clone()
        com=self.robot.data.body_com_pose_b.torch[ids,self.base_id,:3]
        velocity[:,:3]+=torch.cross(velocity[:,3:],quat_apply(pose[:,3:],com),dim=-1)
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.sim.forward();self.scene.update(self.dt)

    def joints(self):
        return self.robot.data.joint_pos.torch[:,self.joint_ids], self.robot.data.joint_vel.torch[:,self.joint_ids]

    def step(self, position_target, torques=None, world_forces=None):
        if torques is not None:
            raise ValueError("Released source mode 1 submits only position targets")
        self.robot.set_joint_position_target(position_target,joint_ids=self.joint_ids)
        if world_forces is not None:
            self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(forces=world_forces,is_global=True)
        self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.dt)
        self.time+=self.dt
        return self.read_state()

    def read_state(self):
        import warp as wp
        d=self.robot.data
        q,qd=self.joints()
        contact_forces=torch.zeros((self.num_envs,len(self.body_names),3),device=self.device)
        for name in self.contact_names:
            contact=self.scene['contact_'+name].data
            contact_forces[:,self.body_names.index(name)]=(contact.net_normal_forces_w.torch[:,0]+contact.friction_force_matrix_w.torch[:,0].sum(1))
        root=torch.cat((d.root_link_pos_w.torch,d.root_link_quat_w.torch,d.root_link_lin_vel_w.torch,d.root_link_ang_vel_w.torch),-1)
        bodies=torch.cat((d.body_link_pos_w.torch,d.body_link_quat_w.torch,d.body_link_lin_vel_w.torch,d.body_link_ang_vel_w.torch),-1)
        measured=wp.to_torch(self.robot.root_view.get_dof_projected_joint_forces())[:,self.joint_ids]
        return dict(root_states=root,root_state=root,dof_pos=q,dof_vel=qd,rigid_body_state=bodies,contact_forces=contact_forces,dof_force=measured,time=self.time.clone())

    def set_joint_gains(self, stiffness, damping):
        self.robot.write_joint_stiffness_to_sim_index(stiffness=stiffness,joint_ids=self.joint_ids)
        self.robot.write_joint_damping_to_sim_index(damping=damping,joint_ids=self.joint_ids)

    def set_material_properties(self, friction, restitution):
        import warp as wp
        materials = wp.to_torch(self.robot.root_view.get_material_properties()).clone()
        materials[:,:,0] = friction.to(materials.device)[:,None]
        materials[:,:,1] = friction.to(materials.device)[:,None]
        materials[:,:,2] = restitution.to(materials.device)[:,None]
        self.robot.root_view.set_material_properties(wp.from_torch(materials),
            wp.from_torch(torch.arange(self.num_envs,dtype=torch.int32)))

    def set_base_com(self, displacement):
        com = self.robot.data.body_com_pose_b.torch[:,[self.base_id]].clone()
        com[:,:, :3] = displacement[:,None]
        self.robot.set_coms_index(coms=com,body_ids=[self.base_id])

    def set_base_mass_delta(self, delta, body_name='base'):
        # Source rigid-body property randomization addresses base (Gym body 0).
        body_id = self.body_names.index(body_name)
        original_mass = self._original_masses[:,[body_id]]
        mass = original_mass + delta[:,None]
        self.robot.set_masses_index(masses=mass,body_ids=[body_id])
        self.robot.set_inertias_index(inertias=self._original_inertias[:,[body_id]]*(mass/original_mass)[...,None],body_ids=[body_id])

    def set_gravity(self, gravity):
        from isaaclab_physx.physics.physx_manager import PhysxManager
        PhysxManager.get_physics_sim_view().set_gravity(tuple(float(v) for v in gravity))
        self._gravity = torch.as_tensor(gravity,device=self.device).clone()

    def training_state(self):
        import warp as wp
        state=self.read_state()
        return {'root_state':state['root_state'].clone(),'dof_pos':state['dof_pos'].clone(),
            'dof_vel':state['dof_vel'].clone(),'time':self.time.clone(),'gravity':self._gravity.clone(),
            'masses':self.robot.data.body_mass.torch.clone(),'inertias':self.robot.data.body_inertia.torch.clone(),
            'coms':self.robot.data.body_com_pose_b.torch.clone(),
            'materials':wp.to_torch(self.robot.root_view.get_material_properties()).clone(),
            'terrain_height_samples':self.terrain.height_samples.clone(),
            'terrain_frictions':self.terrain.terrain_cell_frictions.clone(),
            'terrain_restitutions':self.terrain.terrain_cell_restitutions.clone()}

    def load_training_state(self, state):
        import warp as wp
        # Terrain is authored static geometry. The runner must recreate it with
        # the original construction seed before restoring the dynamic state.
        for key,actual in [('terrain_height_samples',self.terrain.height_samples),
                           ('terrain_frictions',self.terrain.terrain_cell_frictions),
                           ('terrain_restitutions',self.terrain.terrain_cell_restitutions)]:
            if not torch.equal(state[key].to(actual.device),actual):
                raise ValueError('Resume terrain differs: recreate with original construction RNG: '+key)
        self.robot.set_masses_index(masses=state['masses'].to(self.device))
        self.robot.set_coms_index(coms=state['coms'].to(self.device))
        # PhysX diagonalizes this tensor and updates the principal-axis COM
        # rotation. Restore inertia last; writing the old basis afterwards
        # rotates the inertia incorrectly (observed 0.037 kg m² error).
        self.robot.set_inertias_index(inertias=state['inertias'].to(self.device))
        self.robot.root_view.set_material_properties(wp.from_torch(state['materials'].cpu().contiguous()),
            wp.from_torch(torch.arange(self.num_envs,dtype=torch.int32)))
        self.set_gravity(state['gravity'])
        self.write_reset(torch.arange(self.num_envs,device=self.device),state['dof_pos'].to(self.device),
            state['root_state'].to(self.device),state['dof_vel'].to(self.device))
        self.time[:]=state['time'].to(self.device)
