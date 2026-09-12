"""Isaac Lab 3 native PhysX state and explicit original UMI PD efforts."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul
from isaaclab_physx.physics import PhysxCfg
from pawcerto.methods.umi_on_legs import RobotState
from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract
from pawcerto.robots.urdf import read_joint_limits
from pawcerto.methods.umi_on_legs.robot_binding import robot_binding, joint_order

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = ROOT / 'reference/isaac/go2_arx5/usd_path.txt'
DEFAULT_URDF = ROOT / 'reference/isaac/go2_arx5/go2_arx5_merged.urdf'


def euler_xyz_quat(angles):
    """PyTorch3D XYZ uses Rx @ Ry @ Rz, unlike Lab's roll-pitch-yaw helper."""
    zero = torch.zeros_like(angles[:, 0])
    qx = quat_from_euler_xyz(angles[:, 0], zero, zero)
    qy = quat_from_euler_xyz(zero, angles[:, 1], zero)
    qz = quat_from_euler_xyz(zero, zero, angles[:, 2])
    return quat_mul(quat_mul(qx, qy), qz)


class Go2Arx5Isaac:
    def __init__(self, config, joint_names, num_envs=1, device='cuda:0', usd_path=None, training=False,
                 ground_contact_diagnostics=False, force_signal="normal-contact", urdf_path=None):
        if force_signal not in ("normal-contact", "reconstructed-solver"):
            raise ValueError(f"Unknown force signal: {force_signal}")
        self.force_signal = force_signal
        self.joint_velocity_limit_override_rad_s = config.get('joint_velocity_limit_override_rad_s')
        runtime_contract(force_signal, self.joint_velocity_limit_override_rad_s)
        self.binding = robot_binding(config)
        if list(joint_names) != joint_order(config):
            raise ValueError('Runtime joint order differs from UMI robot binding')
        self.config = config
        self.dt = config['env']['cfg']['sim']['dt']
        self.device = device
        self.joint_names = list(joint_names)
        self.training = training
        self.ground_contact_diagnostics = ground_contact_diagnostics
        usd_path = Path(usd_path or config.get('pawcerto_asset', {}).get('usd_path') or self.binding['usd_path'])
        if usd_path.suffix == '.txt':
            usd_path = Path(usd_path.read_text().strip())
        initial = config['env']['cfg']['init_state']
        # This supported Isaac Lab 3 runtime, like the UMI config, uses xyzw.
        initial_rotation = tuple(initial['rot'])
        offset = config['env']['controller']['offset']['data']
        ground_width = 4 * math.ceil(math.sqrt(num_envs)) + 20
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=config['env']['cfg']['sim']['physx']['contact_offset'],
            rest_offset=config['env']['cfg']['sim']['physx']['rest_offset'])
        if config['env']['cfg']['domain_rand'].get('randomize_dof_velocity', False):
            raise ValueError('DOF velocity randomization is not implemented; preserve the original URDF limits')
        self.urdf_path = Path(urdf_path or config.get('pawcerto_asset', {}).get('urdf_path') or self.binding['urdf_path']).resolve()
        asset = ET.parse(self.urdf_path)
        joints = asset.findall('joint')
        source_limits = read_joint_limits(self.urdf_path, joint_names)
        velocity_limits = {}
        for name in joint_names:
            velocity = source_limits[name]['velocity']
            if not math.isfinite(velocity) or velocity <= 0:
                raise ValueError(f'Invalid original URDF velocity limit for {name}: {velocity}')
            velocity_limits[name] = velocity

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.CuboidCfg(
                size=(ground_width, ground_width, .1), collision_props=collision_props,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)))
            robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), activate_contact_sensors=True,
                    collision_props=collision_props,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                        linear_damping=0., angular_damping=0., max_linear_velocity=1000.,
                        # Original Gym stores rad/s; Lab's USD property uses deg/s.
                        max_angular_velocity=math.degrees(config['env']['cfg']['asset']['max_angular_velocity']),
                        max_depenetration_velocity=1.),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0)),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(initial['pos']),
                    rot=initial_rotation, joint_pos=dict(zip(joint_names, offset))),
                actuators={'effort': IdealPDActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=0.,
                    effort_limit=1e9, effort_limit_sim=1e9, velocity_limit_sim=velocity_limits, armature=0.,
                    friction=0., dynamic_friction=0., viscous_friction=0.)})

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device,
            # Keep the validated force-application branch across Lab defaults.
            physics=PhysxCfg(enable_external_forces_every_iteration=False)))
        scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=4.)
        from isaaclab.sensors import ContactSensorCfg
        root_body = self.binding['root_body']
        paths = {root_body: '{ENV_REGEX_NS}/Robot/Geometry/' + root_body}
        pending = list(joints)
        while pending:
            remaining = len(pending)
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent in paths:
                    child = joint.find('child').get('link')
                    paths[child] = paths[parent] + '/' + child
                    pending.remove(joint)
            if len(pending) == remaining:
                raise ValueError(f'URDF joints are not reachable from UMI base link: {[joint.get("name") for joint in pending]}')
        self.contact_names = [link.get('name') for link in asset.findall('link') if link.find('collision') is not None]
        for name in self.contact_names:
            filters = ['/World/Ground/geometry/mesh'] if ground_contact_diagnostics and name.endswith('_foot') else []
            setattr(scene_cfg, 'contact_' + name, ContactSensorCfg(prim_path=paths[name], update_period=0.,
                    filter_prim_paths_expr=filters))
        self.scene = InteractiveScene(scene_cfg)
        from pxr import Usd, UsdPhysics, PhysxSchema
        # Three imported mesh colliders are instance proxies; author their offsets
        # on this scene's instances without modifying the referenced asset.
        collider_instances = set()
        for prim in self.sim.stage.Traverse(Usd.TraverseInstanceProxies()):
            if (str(prim.GetPath()).startswith('/World/envs/')
                    and prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsInstanceProxy()):
                ancestor = prim.GetParent()
                while not ancestor.IsInstance():
                    ancestor = ancestor.GetParent()
                collider_instances.add(str(ancestor.GetPath()))
        for path in collider_instances:
            sim_utils.make_uninstanceable(path, stage=self.sim.stage)
        for prim in self.sim.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/envs/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.modify_collision_properties(str(prim.GetPath()), collision_props, stage=self.sim.stage)
            if str(prim.GetPath()).startswith('/World/envs/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Lab v3.0.0-beta2.patch1 stops below the first rigid body.
                # Keep these idempotent per-link writes for that release fallback.
                sim_utils.modify_rigid_body_properties(
                    str(prim.GetPath()), scene_cfg.robot.spawn.rigid_props, stage=self.sim.stage)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self.sim.reset()
        self.scene.update(self.dt)
        self.robot = self.scene['robot']
        self.body_names = self.robot.body_names
        self.joint_ids = [self.robot.joint_names.index(name) for name in joint_names]
        self.tcp_body_id = self.robot.body_names.index(self.binding['tcp_body'])
        self.num_envs = num_envs
        self.time = torch.zeros(num_envs, device=device)
        if training:
            self._randomize_properties(asset)
        if self.joint_velocity_limit_override_rad_s is not None:
            import warp as wp
            self.robot.write_joint_velocity_limit_to_sim_index(
                limits=torch.full((num_envs, len(joint_names)), 1000., device=device),
                joint_ids=self.joint_ids)
            native_limits = wp.to_torch(self.robot.root_view.get_dof_max_velocities())
            if not bool(torch.all(native_limits[:, self.joint_ids] == 1000.)):
                raise RuntimeError('Native joint velocity override readback differs from 1000 rad/s')
        self._solver_force_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._feet_solver_force_z = None
        if force_signal == 'reconstructed-solver':
            self._initialize_solver_force()
            if (self.joint_velocity_limit_override_rad_s is not None
                    and not bool(torch.all(self._solver_force.dof_max_velocity == 1000.))):
                raise RuntimeError('Reconstructed force joint velocity cache differs from native override')
        self.reset()

    def _initialize_solver_force(self):
        import warp as wp
        from .solver_force import StockSolverForce
        view = self.robot.root_view
        if view.shared_metatype.fixed_base:
            raise ValueError('Solver force requires a floating base')
        parents = [-1] + [view.shared_metatype.link_parent_indices[name] for name in self.body_names[1:]]
        def tensor(method):
            return wp.to_torch(getattr(view, method)()).to(self.device)
        def field(name, radians=False):
            values = []
            for paths in view.link_paths:
                row = []
                for path in paths:
                    prim = self.sim.stage.GetPrimAtPath(path)
                    attr = prim.GetAttribute('physxRigidBody:' + name)
                    value = attr.Get() if attr else None
                    if value is None:
                        raise ValueError(f'Missing composed body property {path}: {name}')
                    row.append(value)
                values.append(row)
            result = torch.tensor(values, dtype=torch.float32, device=self.device)
            return torch.deg2rad(result) if radians else result
        self._solver_force = StockSolverForce(parents, tensor('get_masses'), dt=self.dt,
            body_inertias=tensor('get_inertias').reshape(self.num_envs, len(self.body_names), 3, 3),
            dof_max_velocity=tensor('get_dof_max_velocities'),
            body_max_linear_velocity=field('maxLinearVelocity'),
            body_max_angular_velocity=field('maxAngularVelocity', radians=True),
            body_linear_damping=field('linearDamping'), body_angular_damping=field('angularDamping'),
            armature=tensor('get_dof_armatures'), disable_gravity=tensor('get_disable_gravities'),
            retain_accelerations=field('retainAccelerations'), gyroscopic_forces=field('enableGyroscopicForces'),
            external_forces_every_iteration=self.sim.cfg.physics.enable_external_forces_every_iteration)
        self._solver_foot_ids = [self.body_names.index(leg + '_foot') for leg in ('FR', 'FL', 'RR', 'RL')]

    def _randomize_properties(self, asset):
        import warp as wp
        rand = self.config['env']['cfg']['domain_rand']
        def uniform(shape, limits):
            low, high = torch.as_tensor(limits, device=self.device)
            return torch.rand(shape, device=self.device) * (high-low) + low
        body_ids = [self.body_names.index(name) for name in rand['randomize_rigid_body_masses']]
        original_mass = self.robot.data.body_mass.torch[:, body_ids].clone()
        mass = (original_mass + uniform(original_mass.shape, rand['added_mass_range'])).clamp_min(.01)
        inertia = self.robot.data.body_inertia.torch[:, body_ids].clone()
        self.robot.set_masses_index(masses=mass, body_ids=body_ids)
        self.robot.set_inertias_index(inertias=inertia * (mass/original_mass)[..., None], body_ids=body_ids)
        com_ids = [self.body_names.index(name) for name in rand['randomize_rigid_body_com']]
        com = self.robot.data.body_com_pose_b.torch[:, com_ids].clone()
        com[..., :3] += uniform((self.num_envs, len(com_ids), 3), rand['rigid_body_com_range']['data'])
        self.robot.set_coms_index(coms=com, body_ids=com_ids)
        friction = uniform((self.num_envs, 18), rand['dof_friction_range'])
        damping = uniform((self.num_envs, 18), rand['dof_damping_range'])
        # Gym's dimensionless coefficient uses PhysX's legacy joint-friction
        # model. Lab's similarly named helper writes the new static effort
        # (Nm) instead. The public coefficient API is deprecated but preserves
        # the required model in the currently supported PhysX tensor runtime.
        view = self.robot.root_view
        indices = wp.array(list(range(self.num_envs)), dtype=wp.uint32, device='cpu')
        friction_properties = wp.to_torch(view.get_dof_friction_properties()).clone()
        if torch.any(friction_properties[:, self.joint_ids] != 0):
            friction_properties[:, self.joint_ids] = 0
            view.set_dof_friction_properties(wp.from_torch(friction_properties), indices)
        coefficients = wp.to_torch(view.get_dof_friction_coefficients()).clone()
        coefficients[:, self.joint_ids] = friction.cpu()
        view.set_dof_friction_coefficients(wp.from_torch(coefficients), indices)
        self.robot.write_joint_damping_to_sim_index(damping=damping, joint_ids=self.joint_ids)
        # PhysX enumerates shapes within its breadth-first body order; the critic
        # retains the original Gym depth-first body order (and per-body order).
        links = {link.get('name'): link for link in asset.findall('link')}
        children = {}
        for joint in asset.findall('joint'):
            children.setdefault(joint.find('parent').get('link'), []).append(joint.find('child').get('link'))
        gym_bodies = []
        def visit(name):
            gym_bodies.append(name)
            for child in sorted(children.get(name, [])):
                visit(child)
        visit(self.binding['root_body'])
        lab_shapes = [(name, i) for name in self.body_names for i in range(len(links[name].findall('collision')))]
        gym_shapes = [(name, i) for name in gym_bodies for i in range(len(links[name].findall('collision')))]
        if len(lab_shapes) != self.robot.root_view.max_shapes:
            raise ValueError(f'URDF/PhysX shape count differs: {len(lab_shapes)} vs {self.robot.root_view.max_shapes}')
        canonical_to_lab = [lab_shapes.index(shape) for shape in gym_shapes]
        buckets = uniform((rand['num_friction_buckets'],), rand['friction_range'])
        material_friction = buckets[torch.randint(len(buckets), (self.num_envs, len(gym_shapes)), device=self.device)]
        materials = wp.to_torch(self.robot.root_view.get_material_properties()).clone()
        materials[:, canonical_to_lab, 0] = material_friction.cpu()
        materials[:, canonical_to_lab, 1] = material_friction.cpu()
        materials[:, :, 2] = 0
        self.robot.root_view.set_material_properties(wp.from_torch(materials),
            wp.from_torch(torch.arange(self.num_envs, dtype=torch.int32)))
        self._setup = {'rigidbody_mass': mass, 'rigidbody_com_offset': com[..., :3].reshape(self.num_envs, -1),
                       'dof_friction': friction, 'dof_damping': damping, 'rigid_shape_friction': material_friction}

    def training_setup(self):
        return self._setup

    def apply_domain_randomization(self, global_step):
        rand = self.config['env']['cfg']['domain_rand']
        policy_dt = self.dt * self.config['env']['controller']['decimation_count']
        if rand['push_robots'] and global_step % int(rand['push_interval_s']/policy_dt) == 0:
            velocity = (torch.rand((self.num_envs, 6), device=self.device) * 2 - 1) * torch.tensor(rand['max_push_vel']['data'], device=self.device)
            self._solver_force_valid[:] = False
            self.robot.write_root_velocity_to_sim_index(root_velocity=velocity)
        if rand['transport_robots'] and global_step % int(rand['transport_interval_s']/policy_dt) == 0:
            pose = self.robot.data.root_link_pose_w.torch.clone()
            pose[:, :3] += torch.randn((self.num_envs, 3), device=self.device) * torch.tensor(rand['transport_pos_noise_std']['data'], device=self.device)
            angles = torch.randn((self.num_envs, 3), device=self.device) * torch.tensor(rand['transport_euler_noise_std']['data'], device=self.device)
            pose[:, 3:] = quat_mul(pose[:, 3:], euler_xyz_quat(angles))
            self._solver_force_valid[:] = False
            self.robot.write_root_pose_to_sim_index(root_pose=pose)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._solver_force_valid[ids] = False
        pose = self.robot.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += self.scene.env_origins[ids]
        q = self.robot.data.default_joint_pos.torch[ids].clone()
        if self.training:
            initial = self.config['env']['cfg']['init_state']
            pose[:, :3] += (torch.rand((len(ids), 3), device=self.device)*2-1) * torch.tensor(initial['pos_noise'], device=self.device)
            angles = (torch.rand((len(ids), 3), device=self.device)*2-1) * torch.tensor(initial['euler_noise'], device=self.device)
            pose[:, 3:] = quat_mul(pose[:, 3:], euler_xyz_quat(angles))
            scale = self.config['env']['dof_pos_reset_range_scale']
            limits = self.robot.data.joint_pos_limits.torch[ids]
            q += torch.randn_like(q) * scale * (limits[..., 1] - limits[..., 0])
            q = q.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
        self.robot.reset(ids)
        self.time[ids] = 0
        self.sim.forward()
        self.scene.update(self.dt)
        return self.state()

    def joints(self):
        return (self.robot.data.joint_pos.torch[:, self.joint_ids],
                self.robot.data.joint_vel.torch[:, self.joint_ids])

    def state(self):
        data = self.robot.data
        root_rotation = matrix_from_quat(data.root_link_quat_w.torch)
        rotation = matrix_from_quat(data.body_link_quat_w.torch[:, self.tcp_body_id])
        position = (data.body_link_pos_w.torch[:, self.tcp_body_id] - self.scene.env_origins)
        # The binding carries the robot-specific fixed transform to the policy TCP.
        ee_rotation = torch.tensor(self.binding['tcp_rotation'], device=self.device)
        ee_offset = torch.tensor(self.binding['tcp_xyz'], device=self.device)
        ee_pose = torch.eye(4, device=self.device).repeat(len(position), 1, 1)
        ee_pose[:, :3, :3] = rotation @ ee_rotation
        ee_pose[:, :3, 3] = position + (rotation @ ee_offset)
        angular = data.root_link_ang_vel_w.torch
        q, qd = self.joints()
        return RobotState((root_rotation.transpose(-1, -2) @ angular[..., None]).squeeze(-1),
                          -root_rotation[:, 2, :], q, qd, ee_pose, self.time.clone())

    def training_state(self):
        data = self.robot.data
        rotation = matrix_from_quat(data.root_link_quat_w.torch)
        forces = torch.zeros((self.num_envs, len(self.body_names), 3), device=self.device)
        for name in self.contact_names:
            forces[:, self.body_names.index(name)] = self.scene['contact_' + name].data.net_forces_w.torch[:, 0]
        result = {'root_pos': data.root_link_pos_w.torch - self.scene.env_origins,
                'local_root_lin_vel': (rotation.transpose(-1, -2) @ data.root_link_lin_vel_w.torch[..., None]).squeeze(-1),
                'contact_forces': forces,
                'body_pos': data.body_link_pos_w.torch - self.scene.env_origins[:, None],
                'feet_force_z': forces[:, [self.body_names.index(leg + '_foot') for leg in ('FR', 'FL', 'RR', 'RL')], 2]}

        if self.force_signal == 'reconstructed-solver':
            result['feet_solver_force_valid'] = self._solver_force_valid.clone()
            if self._feet_solver_force_z is not None:
                result['feet_solver_force_z'] = self._feet_solver_force_z
        return result

    def feet_ground_force_z(self):
        """Ground-filtered normal Fz, separate from the current port's normal-contact reward input."""
        if not self.ground_contact_diagnostics:
            raise RuntimeError('Ground contact diagnostics were not enabled for this scene')
        forces = []
        for leg in ('FR', 'FL', 'RR', 'RL'):
            matrix = self.scene['contact_' + leg + '_foot'].data.force_matrix_w
            if matrix is None or matrix.torch.shape[2] != 1:
                raise RuntimeError(f'{leg} foot ground filter did not resolve exactly one ground collision shape')
            forces.append(matrix.torch[:, 0, 0, 2])
        return torch.stack(forces, dim=-1)

    @torch.no_grad()
    def step_torque(self, torque):
        self._solver_force_valid[:] = False
        reconstructed = self.force_signal == 'reconstructed-solver'
        if reconstructed:
            import warp as wp
            def tensor(value):
                return wp.to_torch(value)
            wrench_terms = []
            for name in ('instantaneous_wrench_composer', 'permanent_wrench_composer'):
                composer = getattr(self.robot, name)
                if composer.active:
                    raise ValueError(f'Unsupported active {name} before write_data_to_sim')
                for field in ('global_force_w', 'global_torque_w', 'global_force_at_com_w', 'local_force_b', 'local_torque_b'):
                    wrench_terms.append(tensor(getattr(composer, field)).clone())
            wrench = torch.stack(wrench_terms)
        self.robot.set_joint_effort_target_index(target=torque.to(self.device).contiguous(), joint_ids=self.joint_ids)
        self.scene.write_data_to_sim()
        if reconstructed:
            view = self.robot.root_view
            com_position = self.robot.data.body_com_pose_w.torch[..., :3]
            prediction, _, _ = self._solver_force.predict(
                mass_matrix=tensor(view.get_generalized_mass_matrices()),
                gravity=tensor(view.get_gravity_compensation_forces()),
                coriolis=tensor(view.get_coriolis_and_centrifugal_compensation_forces()),
                jacobian=tensor(view.get_jacobians()),
                com_position=com_position, com_velocity=tensor(view.get_link_velocities()),
                link_rotation=matrix_from_quat(self.robot.data.body_link_quat_w.torch),
                dof_velocity=tensor(view.get_dof_velocities()), actuation=tensor(view.get_dof_actuation_forces()),
                external_wrench_terms=wrench)
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.time += self.dt
        if reconstructed:
            force = self._solver_force.residual_force(tensor(view.get_link_accelerations()), prediction)
            # Free dynamics consumed PRE velocities; POST speed is not its clamp condition.
            post_velocity = tensor(view.get_link_velocities())
            post_dof_velocity = tensor(view.get_dof_velocities())
            if not bool(torch.isfinite(post_velocity).all() and torch.isfinite(post_dof_velocity).all()):
                raise FloatingPointError('Nonfinite stock dynamics post-step velocity')
            self._feet_solver_force_z = force[:, self._solver_foot_ids, 2]
            self._solver_force_valid[:] = True
