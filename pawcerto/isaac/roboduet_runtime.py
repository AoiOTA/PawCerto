"""Official Isaac Lab Go1+ARX5 physics path, one control substep per call."""
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
from pawcerto.robots.go1_arx5 import JOINT_NAMES, DEFAULT_POS
from pawcerto.robots.urdf import read_joint_limits
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_USD = ROOT / 'reference/isaac/go1_arx5/usd_path.txt'


@sim_utils.clone
def spawn_roboduet(prim_path, cfg, translation=None, orientation=None, **kwargs):
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
            sim_utils.modify_rigid_body_properties(str(child.GetPath()), cfg.rigid_props, stage=stage)
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
    print('[RoboDuet] Source articulation physics overrides complete', flush=True)
    return prim


class Go1Arx5Isaac:
    def __init__(self, num_envs=1, device='cuda:0', usd_path=DEFAULT_USD):
        self.dt = .005
        self.device, self.num_envs = device, num_envs
        self.stage = 1
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
        kp = dict(zip(JOINT_NAMES, [0.] * 12 + [40.,70.,70.,25.,25.,25.,50.,50.]))
        kd = dict(zip(JOINT_NAMES, [0.] * 12 + [3.,15.,15.,2.,2.,2.,20.,20.]))
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=.01, rest_offset=0.)
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.CuboidCfg(
                size=(4 * math.ceil(math.sqrt(num_envs)) + 20,) * 2 + (.1,), collision_props=collision,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.05)))
            robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
                spawn=sim_utils.UsdFileCfg(func=spawn_roboduet, usd_path=str(path), activate_contact_sensors=True,
                    collision_props=collision,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,
                        max_linear_velocity=1000.,max_angular_velocity=math.degrees(1000.),max_depenetration_velocity=1.),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=4,solver_velocity_iteration_count=1)),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.34),joint_pos=dict(zip(JOINT_NAMES,DEFAULT_POS))),
                actuators={'mixed': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=kp,damping=kd,
                    effort_limit_sim=1e9,velocity_limit_sim=velocity,armature=0.,friction=0.,dynamic_friction=0.,viscous_friction=0.)})
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device=device,
            physics=PhysxCfg(enable_external_forces_every_iteration=False)))
        cfg = SceneCfg(num_envs=num_envs,env_spacing=4.)
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
            filters = ['/World/Ground/geometry/mesh'] + [paths[other] for other in self.contact_names if other != name]
            setattr(cfg,'contact_' + name,ContactSensorCfg(prim_path=paths[name],update_period=0.,
                filter_prim_paths_expr=filters,track_friction_forces=True,max_contact_data_count_per_prim=32))
        print(f'[RoboDuet] Constructing {num_envs} environments', flush=True)
        self.scene = InteractiveScene(cfg)
        print('[RoboDuet] Scene constructed; initializing physics', flush=True)
        self.sim.reset()
        self.scene.update(self.dt)
        self.robot = self.scene['robot']
        self.body_names = self.robot.body_names
        self.joint_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        self.ee_id = self.body_names.index('zarx_body6')
        self.feet_indices = [self.body_names.index(leg+'_foot') for leg in ('FL','FR','RL','RR')]
        mass = self.robot.data.body_mass.torch[:,[self.ee_id]].clone()
        inertia = self.robot.data.body_inertia.torch[:,[self.ee_id]].clone()
        self.robot.set_masses_index(masses=mass+.1,body_ids=[self.ee_id])
        self.robot.set_inertias_index(inertias=inertia*((mass+.1)/mass)[...,None],body_ids=[self.ee_id])
        self.base_id = self.body_names.index('base')
        self._original_masses = self.robot.data.body_mass.torch.clone()
        self._original_inertias = self.robot.data.body_inertia.torch.clone()
        self.set_base_com(torch.zeros((num_envs,3),device=device))
        self._gravity = torch.tensor([0.,0.,-9.81],device=device)
        self.time = torch.zeros(num_envs,device=device)
        self.reset()

    def set_stage(self, stage):
        if stage not in (1,2):
            raise ValueError('RoboDuet stage must be 1 or 2')
        self.stage = stage

    def reset(self, ids=None, joint_pos=None, root_pose=None, root_velocity=None):
        ids = torch.arange(self.num_envs,device=self.device) if ids is None else ids
        pose = self.robot.data.default_root_pose.torch[ids].clone() if root_pose is None else root_pose.clone()
        if root_pose is not None:
            # Public runtime quaternions are wxyz; Lab 3 warp transforms are xyzw.
            pose[:,3:] = root_pose[:,[4,5,6,3]]
        pose[:,:3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        velocity = torch.zeros((len(ids),6),device=self.device) if root_velocity is None else root_velocity.clone()
        # The current official link-velocity setter writes native COM velocity.
        # Use the public COM setter with the explicit rigid-body conversion.
        com = self.robot.data.body_com_pose_b.torch[ids,self.base_id,:3]
        velocity[:,:3] += torch.cross(velocity[:,3:],quat_apply(pose[:,3:],com),dim=-1)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        q = self.robot.data.default_joint_pos.torch[ids].clone()
        if joint_pos is not None:
            q[:,self.joint_ids] = joint_pos
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids)
        self.robot.reset(ids)
        self.time[ids] = 0.
        self.sim.forward()
        self.scene.update(self.dt)
        return self.state()

    def joints(self):
        return self.robot.data.joint_pos.torch[:,self.joint_ids], self.robot.data.joint_vel.torch[:,self.joint_ids]

    def step_control(self, leg_torque, arm_target, *, capture_state=True):
        combined = torch.cat((leg_torque,arm_target),dim=-1)
        self.robot.set_joint_effort_target(combined,joint_ids=self.joint_ids)
        # Gym EFFORT drive ignores its target write but retains a zero-rest
        # spring/damper. The nominal arm target is applied as extra effort.
        self.robot.set_joint_position_target(torch.zeros_like(combined),joint_ids=self.joint_ids)
        locked = self.joint_ids[12 if self.stage == 1 else 18:]
        default = self.default_pos[12 if self.stage == 1 else 18:].expand(self.num_envs,-1)
        self.robot.write_joint_position_to_sim_index(position=default,joint_ids=locked)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default),joint_ids=locked)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.time += self.dt
        # Training only consumes the policy-boundary state. Keep sensor reads
        # optional without changing physics, scene timestamps, or joint reads.
        return self.state() if capture_state else None

    def state(self):
        d = self.robot.data
        q,qd = self.joints()
        forces = torch.zeros((self.num_envs,len(self.body_names),3),device=self.device)
        for name in self.contact_names:
            contact = self.scene['contact_'+name].data
            # Both fields are world-frame N for the latest .005 s physics step.
            forces[:,self.body_names.index(name)] = (contact.net_normal_forces_w.torch[:,0]
                + contact.friction_force_matrix_w.torch[:,0].sum(dim=1))
        result = {'joint_pos':q,'joint_vel':qd,'root_pos':d.root_link_pos_w.torch-self.scene.env_origins,
            'root_quat':d.root_link_quat_w.torch[:,[3,0,1,2]],'root_lin_vel':d.root_link_lin_vel_w.torch,
            'root_ang_vel':d.root_link_ang_vel_w.torch,'body_pos':d.body_link_pos_w.torch-self.scene.env_origins[:,None],
            'body_quat':d.body_link_quat_w.torch[:,:,[3,0,1,2]],'body_lin_vel':d.body_link_lin_vel_w.torch,
            'body_ang_vel':d.body_link_ang_vel_w.torch,'contact_forces':forces,'time':self.time.clone()}
        for field in ('pos','quat','lin_vel','ang_vel'):
            result['ee_'+field] = result['body_'+field][:,self.ee_id]
        return result

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

    def set_base_mass_delta(self, delta, body_name='trunk'):
        # Named consumer-selected torso repairs source body0 mount randomization.
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
        state = self.state()
        return {**{k:state[k].clone() for k in ('root_pos','root_quat','root_lin_vel','root_ang_vel','joint_pos','joint_vel','time')},
            'stage':self.stage,'gravity':self._gravity.clone(),
            'masses':self.robot.data.body_mass.torch.clone(),
            'inertias':self.robot.data.body_inertia.torch.clone(),
            'coms':self.robot.data.body_com_pose_b.torch.clone(),
            'materials':wp.to_torch(self.robot.root_view.get_material_properties()).clone()}

    def load_training_state(self, state):
        import warp as wp
        self.robot.set_masses_index(masses=state['masses'].to(self.device))
        self.robot.set_inertias_index(inertias=state['inertias'].to(self.device))
        self.robot.set_coms_index(coms=state['coms'].to(self.device))
        self.robot.root_view.set_material_properties(wp.from_torch(state['materials'].cpu().contiguous()),
            wp.from_torch(torch.arange(self.num_envs,dtype=torch.int32)))
        self.set_gravity(state['gravity'])
        self.set_stage(state['stage'])
        self.reset(joint_pos=state['joint_pos'].to(self.device),
            root_pose=torch.cat((state['root_pos'],state['root_quat']),dim=-1).to(self.device),
            root_velocity=torch.cat((state['root_lin_vel'],state['root_ang_vel']),dim=-1).to(self.device))
        self.robot.write_joint_velocity_to_sim_index(velocity=state['joint_vel'].to(self.device),joint_ids=self.joint_ids)
        self.time[:] = state['time'].to(self.device)
        self.sim.forward()
        self.scene.update(self.dt)
