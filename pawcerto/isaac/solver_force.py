"""Conditional stock-PhysX free dynamics and solver-force reconstruction.

This tensor path supplies reconstructed EMD input when explicitly selected by
the runtime; it is not an equivalence claim for the original Gym forceSensor. Inputs must describe the same pre-physics
state: floating-base world linear/angular coordinates, then internal DOF order;
Jacobian linear rows and body positions/velocities refer to world COMs.
"""
import math

import torch


def _finite(**values):
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite stock dynamics input/output: {name}")


class StockSolverForce:
    """Evaluate zero configured body damping with stock body-speed damping.

    Joint drives and joint friction belong to the solver; they are deliberately
    not subtracted from the supplied actuation force. Nonzero armature, retained
    external accelerations and external-forces-every-TGS-iteration are not
    supported here. Callers must capture wrench terms before write_data_to_sim
    consumes instantaneous commands, and preserve failures instead of replacing
    them with zero forces.
    """

    def __init__(self, parents, masses, *, body_inertias, dt, dof_max_velocity,
                 body_max_linear_velocity, body_max_angular_velocity,
                 body_linear_damping, body_angular_damping, armature,
                 disable_gravity, retain_accelerations, gyroscopic_forces,
                 external_forces_every_iteration):
        self.parents = tuple(parents)
        if masses.ndim != 2 or len(parents) != masses.shape[1] or parents[0] != -1:
            raise ValueError("Expected one floating-root, parent-before-child physical body tree")
        if any(parent < 0 or parent >= child for child, parent in enumerate(parents[1:], 1)):
            raise ValueError("Physical body parents must precede their children")
        if body_inertias.shape != (*masses.shape, 3, 3):
            raise ValueError("Expected full COM inertia matrices expressed in rigid-body-prim frames")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be finite and positive")
        if external_forces_every_iteration:
            raise ValueError("External-forces-every-TGS-iteration is unsupported")
        static = dict(masses=masses, body_inertias=body_inertias, dof_max_velocity=dof_max_velocity,
                      body_max_linear_velocity=body_max_linear_velocity,
                      body_max_angular_velocity=body_max_angular_velocity,
                      body_linear_damping=body_linear_damping,
                      body_angular_damping=body_angular_damping, armature=armature,
                      disable_gravity=disable_gravity, retain_accelerations=retain_accelerations,
                      gyroscopic_forces=gyroscopic_forces)
        _finite(**static)
        for name in ("body_linear_damping", "body_angular_damping", "armature",
                     "disable_gravity", "retain_accelerations"):
            if bool(torch.any(static[name] != 0)):
                raise ValueError(f"Unsupported nonzero {name}")
        if not bool(torch.all(gyroscopic_forces != 0)):
            raise ValueError("This verified branch requires gyroscopic forces enabled")
        for name in ("masses", "dof_max_velocity", "body_max_linear_velocity", "body_max_angular_velocity"):
            if bool(torch.any(static[name] <= 0)):
                raise ValueError(f"Expected positive {name}")
        self.masses = masses.clone()
        # Native get_inertias() is about the COM but expressed in the body prim
        # frame. It is a full matrix, not principal moments in the COM frame.
        self.body_inertias = body_inertias.clone()
        self.dt = dt
        self.dof_max_velocity = dof_max_velocity.clone()
        self.body_max_linear_velocity = body_max_linear_velocity.clone()
        self.body_max_angular_velocity = body_max_angular_velocity.clone()
        # Siblings share a depth and can be propagated in one tensor operation.
        depths = [0]
        for parent in parents[1:]:
            depths.append(depths[parent] + 1)
        self.levels = []
        for depth in range(1, max(depths) + 1):
            children = [i for i, value in enumerate(depths) if value == depth]
            self.levels.append((torch.tensor(children, device=masses.device),
                                torch.tensor([parents[i] for i in children], device=masses.device)))

    def check_velocity_branch(self, com_velocity, dof_velocity):
        """Validate and measure full body speeds after FD DOF preprocessing.

        The caller supplies the reconstructed forward-dynamics COM velocity.
        Body speed limits add damping bias; they do not clip this velocity.
        DOF global scaling and separate gyro clipping are covered by predict.
        """
        _finite(com_velocity=com_velocity, dof_velocity=dof_velocity)
        linear = torch.linalg.vector_norm(com_velocity[..., :3], dim=-1)
        angular = torch.linalg.vector_norm(com_velocity[..., 3:], dim=-1)
        return linear, angular

    def _com_bias(self, com_position, com_velocity):
        bias = torch.zeros_like(com_velocity)
        cross = torch.linalg.cross
        for children, parents in self.levels:
            r = com_position[:, children] - com_position[:, parents]
            wp = com_velocity[:, parents, 3:]
            dw = com_velocity[:, children, 3:] - wp
            dv = com_velocity[:, children, :3] - com_velocity[:, parents, :3] - cross(wp, r)
            parent_bias = bias[:, parents]
            bias[:, children, 3:] = parent_bias[..., 3:] + cross(wp, dw)
            bias[:, children, :3] = (parent_bias[..., :3] + cross(parent_bias[..., 3:], r)
                                    + cross(wp, cross(wp, r)) + 2 * cross(wp, dv) + cross(dw, dv))
        return bias

    @torch.no_grad()
    def predict(self, *, mass_matrix, coriolis, gravity, jacobian, com_position,
                com_velocity, link_rotation, dof_velocity, actuation, external_wrench_terms):
        """Return world COM free acceleration, generalized qdd and COM bias.

        link_rotation is the same PRE body-prim-to-world rotation, not the COM
        principal-axis rotation. com_velocity must be read AFTER public C:
        that getter propagates min(raw_qd, +limit), including its asymmetric
        negative side. dof_velocity must retain the raw DOF values. FD instead
        scales all DOFs of each articulation by one ratio, leaving root twist
        unchanged, and clips only its gyroscopic angular velocity to 1/dt.
        Body speed limits add inertial damping bias using the full FD velocity;
        they do not change the recursive COM bias or the gyro-only clipping.
        """
        n, bodies = self.masses.shape
        dofs = actuation.shape[-1]
        if (mass_matrix.shape != (n, dofs + 6, dofs + 6)
                or jacobian.shape != (n, bodies, 6, dofs + 6)
                or com_position.shape != (n, bodies, 3)
                or com_velocity.shape != (n, bodies, 6)
                or link_rotation.shape != (n, bodies, 3, 3)
                or coriolis.shape != (n, dofs + 6) or gravity.shape != coriolis.shape
                or dof_velocity.shape != (n, dofs) or self.dof_max_velocity.shape != (n, dofs)):
            raise ValueError("Incomplete floating-base M/C/G/J or mismatched physical-body/DOF order")
        _finite(mass_matrix=mass_matrix, coriolis=coriolis, gravity=gravity,
                jacobian=jacobian, com_position=com_position, link_rotation=link_rotation, actuation=actuation,
                com_velocity=com_velocity, dof_velocity=dof_velocity, external_wrench_terms=external_wrench_terms)
        if bool(torch.any(external_wrench_terms != 0)):
            raise ValueError("Nonzero external wrench is unsupported in this branch")
        if dofs:
            # A zero DOF speed gives an infinite allowed ratio; min with 1
            # correctly leaves that articulation unscaled if all speeds are 0.
            ratio = (self.dof_max_velocity / dof_velocity.abs()).amin(dim=-1, keepdim=True).clamp(max=1.)
            fd_dof_velocity = ratio * dof_velocity
            public_c_dof_velocity = torch.minimum(dof_velocity, self.dof_max_velocity)
            delta_velocity = torch.cat((torch.zeros((n, 6), dtype=dof_velocity.dtype, device=dof_velocity.device),
                                        fd_dof_velocity - public_c_dof_velocity), dim=-1)
            # Native com_velocity already represents J*nu_C. Applying only the
            # increment preserves the original no-DOF-clamp numerical path.
            fd_velocity = com_velocity + (jacobian @ delta_velocity[:, None, :, None]).squeeze(-1)
        else:
            fd_velocity = com_velocity
        linear_speed, angular_speed = self.check_velocity_branch(fd_velocity, dof_velocity)
        force = torch.cat((torch.zeros((n, 6), dtype=actuation.dtype, device=actuation.device), actuation), -1)
        inertia_world = link_rotation @ self.body_inertias @ link_rotation.transpose(-1, -2)
        bias_c = self._com_bias(com_position, com_velocity)
        bias = self._com_bias(com_position, fd_velocity)
        omega_c, omega_f = com_velocity[..., 3:], fd_velocity[..., 3:]
        gyro_c = torch.linalg.cross(omega_c, (inertia_world @ omega_c.unsqueeze(-1)).squeeze(-1))
        gyro_f = torch.linalg.cross(omega_f, (inertia_world @ omega_f.unsqueeze(-1)).squeeze(-1))
        scale = (self.dt * angular_speed).clamp_min(1.).reciprocal()
        # Algebraically gyro_f*s^2 - gyro_c. This form also retains the prior
        # gyro-only expression exactly when the two velocities are identical.
        gyro_delta = (gyro_f - gyro_c) + (scale.square() - 1.)[..., None] * gyro_f
        bias_delta = bias - bias_c
        delta_wrench = torch.cat((self.masses[..., None] * bias_delta[..., :3],
                                  (inertia_world @ bias_delta[..., 3:, None]).squeeze(-1) + gyro_delta), dim=-1)
        # PhysX computeSpatialInertiaW (GPU forwardDynamic2.cu) adds
        # speed-limit damping to FD bias for every body,
        # including the floating root. Use the full FD COM velocity here:
        # the 1/dt angular clip above belongs only to the gyroscopic term.
        # Positive limits make the denominator safe at zero speed; below and
        # at each limit the added scale is exactly zero.
        linear_scale = 1. - self.body_max_linear_velocity / torch.maximum(
            linear_speed, self.body_max_linear_velocity)
        angular_scale = 1. - self.body_max_angular_velocity / torch.maximum(
            angular_speed, self.body_max_angular_velocity)
        speed_damping = torch.cat((
            fd_velocity[..., :3] * self.masses[..., None] * linear_scale[..., None] * (1. / self.dt),
            (inertia_world @ omega_f.unsqueeze(-1)).squeeze(-1) * angular_scale[..., None] * (1. / self.dt)), dim=-1)
        delta_wrench = delta_wrench + speed_damping
        delta_coriolis = (jacobian.transpose(-1, -2) @ delta_wrench.unsqueeze(-1)).sum(dim=1).squeeze(-1)
        qdd = torch.linalg.solve(mass_matrix, (force - coriolis - delta_coriolis - gravity).unsqueeze(-1)).squeeze(-1)
        prediction = torch.matmul(jacobian, qdd[:, None, :, None]).squeeze(-1) + bias
        _finite(free_qdd=qdd, com_bias=bias, free_acceleration=prediction)
        return prediction, qdd, bias

    @torch.no_grad()
    def residual_force(self, reported_acceleration, free_acceleration):
        """Mass-weighted world linear residual; includes all solver contributions."""
        if reported_acceleration.shape != free_acceleration.shape:
            raise ValueError("Reported/free acceleration shapes differ")
        _finite(reported_acceleration=reported_acceleration, free_acceleration=free_acceleration)
        result = self.masses[..., None] * (reported_acceleration - free_acceleration)[..., :3]
        _finite(solver_force_residual=result)
        return result

    @torch.no_grad()
    def approximate_sensor_wrench(self, reported_acceleration, free_acceleration, *,
                        link_rotation, com_offset_b, applied_force_w):
        """Experimental solver residual plus applied force at COM, PRE body axes.

        Reported alpha contains free alpha plus solver delta-w/dt. Subtraction
        removes FD gyro already: no second omega cross I omega is added. Use
        PRE COM inertia and shift the COM moment to the body origin.
        This is a sensor-equivalence hypothesis, not a production force mode.
        """
        shape = (*self.masses.shape, 3)
        if (reported_acceleration.shape != (*self.masses.shape, 6)
                or link_rotation.shape != (*self.masses.shape, 3, 3)
                or com_offset_b.shape != shape or applied_force_w.shape != shape):
            raise ValueError("Incomplete six-axis PRE wrench inputs")
        _finite(link_rotation=link_rotation, com_offset_b=com_offset_b, applied_force_w=applied_force_w)
        force_w = self.residual_force(reported_acceleration, free_acceleration) + applied_force_w
        inertia_w = link_rotation @ self.body_inertias @ link_rotation.transpose(-1, -2)
        moment_w = (inertia_w @ (reported_acceleration-free_acceleration)[..., 3:, None]).squeeze(-1)
        lever_w = (link_rotation @ com_offset_b[..., None]).squeeze(-1)
        moment_w = moment_w + torch.linalg.cross(lever_w, force_w)
        world = torch.stack((force_w, moment_w), dim=-1)
        body = link_rotation.transpose(-1, -2) @ world
        result = torch.cat((body[..., 0], body[..., 1]), dim=-1)
        _finite(residual_wrench=result)
        return result
