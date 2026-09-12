"""Read the current Go2/ARX5 contact solution without advancing simulation."""
import mujoco
import numpy as np

FEET = ('FR_foot', 'FL_foot', 'RR_foot', 'RL_foot')
FORCE_THRESHOLD_N = 1.0


def geom_role(model, geom_id):
    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name
    if model.body_rootid[body_id] != model.body('base').id:
        return 'external'
    if body_name in FEET:
        return 'foot'
    mesh_id = int(model.geom_dataid[geom_id])
    # Both finray fingers use the official collider mesh. link6 also contains
    # wrist/structural geoms; do not exempt that whole body as a gripper.
    if (body_name == 'link6' and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and model.mesh(mesh_id).name == 'collider'):
        return 'gripper_finger'
    return 'nonfoot_nongripper'


def contact_snapshot(sim):
    """Return foot net Fz, foot-ground Fz and all current contact records.

    This reads the existing solver state only. The caller must label its solve
    phase: endpoint mj_forward or mj_step before the next kinematic refresh.
    """
    model, data = sim.model, sim.data
    foot_forces_z = np.zeros(4)
    foot_ground_forces_z = np.zeros(4)
    ground_geom_id = model.geom('ground').id
    foot_ids = {model.body(name).id: i for i, name in enumerate(FEET)}
    records = []
    for index, contact in enumerate(data.contact):
        local_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local_force)
        world_force = contact.frame.reshape(3, 3).T @ local_force[:3]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        body_ids = [int(model.geom_bodyid[g]) for g in geom_ids]
        for side, (body_id, sign) in enumerate(zip(body_ids, (-1, 1))):
            if body_id in foot_ids:
                foot_forces_z[foot_ids[body_id]] += sign * world_force[2]
                if geom_ids[1-side] == ground_geom_id:
                    foot_ground_forces_z[foot_ids[body_id]] += sign * world_force[2]
        roles = [geom_role(model, g) for g in geom_ids]
        norm = float(np.linalg.norm(world_force))
        records.append({'time_s': float(data.time), 'body_names': [model.body(b).name for b in body_ids],
                        'geom_ids': geom_ids, 'geom_roles': roles,
                        'nonfoot_nongripper': 'nonfoot_nongripper' in roles,
                        'foot_ground': ground_geom_id in geom_ids and 'foot' in roles,
                        'kind': 'external' if 'external' in roles else 'self',
                        'position_world_m': contact.pos.tolist(), 'separation_m': float(contact.dist),
                        'force_world_on_geom2_N': world_force.tolist(), 'force_norm_N': norm,
                        'normal_force_N': float(local_force[0]),
                        'force_norm_gt_1N': norm > FORCE_THRESHOLD_N})
    return foot_forces_z, foot_ground_forces_z, records
