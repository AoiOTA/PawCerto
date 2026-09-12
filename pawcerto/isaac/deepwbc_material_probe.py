"""Independent USD fixture for the signed-friction pair hypothesis.

This module starts no application and does not change the training runtime.
"""
from pathlib import Path


def build_fixture(stage, output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    output = Path(output)
    template = Usd.Stage.CreateNew(str(output / 'shared-collision.usda'))
    root = UsdGeom.Xform.Define(template, '/Geometry')
    template.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(template, '/Geometry/Shape')
    cube.CreateSizeAttr(1.)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    cube.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
    cube.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    template.GetRootLayer().Save()
    rows = []
    for f in (-.5, -.25, .25, 1.):
        for pair in ('robot_ground', 'robot_box', 'robot_robot'):
            rows.append(dict(pair=pair, raw_friction=f, expected_mu=max(0., f) if pair == 'robot_robot' else (f+1)/2))
    rows.extend([dict(pair='box_ground', raw_friction=1., expected_mu=1.),
                 dict(pair='isolated', raw_friction=1., expected_mu=None)])

    def material(path, value, mode):
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(value); api.CreateDynamicFrictionAttr(value); api.CreateRestitutionAttr(0.)
        mat.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
        mat.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set(mode)
        return mat

    def body(path, pos, dynamic, kinematic=False):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if dynamic:
            api = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            api.CreateKinematicEnabledAttr(kinematic)
            mass = UsdPhysics.MassAPI.Apply(x.GetPrim())
            mass.CreateMassAttr(1.); mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.1*.1/6))
            x.GetPrim().AddAppliedSchema('PhysxRigidBodyAPI')
            for name in ('linearDamping', 'angularDamping'):
                x.GetPrim().CreateAttribute('physxRigidBody:'+name, Sdf.ValueTypeNames.Float).Set(0.)
        return x.GetPrim()

    def shape(path, size, mat, instance=True):
        if not instance:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.)
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, materialPurpose='physics')
            return path
        x = UsdGeom.Xform.Define(stage, path)
        x.GetPrim().GetReferences().AddReference(str(output / 'shared-collision.usda'), '/Geometry')
        x.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(x.GetPrim()).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
        x.GetPrim().SetInstanceable(True)
        return path + '/Shape'

    def exclude(author, targets):
        rel = UsdPhysics.FilteredPairsAPI.Apply(author).CreateFilteredPairsRel()
        for target in targets: rel.AddTarget(target)

    proxy_paths = []
    channel_memberships, box_roots, terrain_paths = [], [], []
    for i, row in enumerate(rows):
        path = '/World/Cases/case_%02d' % i
        lane = 0 if i == 3 else i
        row.update(path=path, position=[0., lane*3., .055], overlaps_case=0 if i == 3 else None)
        f = row['raw_friction']
        ext = material(path+'/ExternalMaterial', (f+1)/2, 'multiply')
        self_mat = material(path+'/SelfMaterial', max(0.,f), 'average')
        ordinary = material(path+'/OrdinaryMaterial', 1., 'average')
        mover = body(path+'/Mover', row['position'], True)
        robot_mover = row['pair'] not in ('box_ground',)
        mover_ext = shape(path+'/Mover/External', (.1,)*3, ext if robot_mover else ordinary)
        proxy_paths.append(mover_ext)
        mover_self = shape(path+'/Mover/Self', (.1,)*3, self_mat) if robot_mover else None
        if mover_self: proxy_paths.append(mover_self)
        ext_paths = [mover_ext] if robot_mover else []
        self_paths = [mover_self] if robot_mover else []
        robot_support = row['pair'] == 'robot_robot'
        support = body(path+'/Support', (0., lane*3., -.05), robot_support or row['pair']=='robot_box', True)
        support_ext = shape(path+'/Support/External', (10.,1.,.1), ext if robot_support else ordinary,
                            instance=robot_support or row['pair']=='robot_box')
        if stage.GetPrimAtPath(support_ext).IsInstanceProxy(): proxy_paths.append(support_ext)
        if robot_support:
            support_self = shape(path+'/Support/Self', (10.,1.,.1), self_mat)
            proxy_paths.append(support_self)
            # Production group membership leaves only self/self contact.
            ext_paths.append(support_ext); self_paths.append(support_self)
        if not robot_support and row['pair'] != 'robot_box': terrain_paths.append(support_ext)
        if row['pair'] == 'robot_box': box_roots.append(str(support.GetPath()))
        elif not robot_mover: box_roots.append(str(mover.GetPath()))
        else:
            empty_box = path+'/UnusedBoxGroup'
            UsdGeom.Xform.Define(stage,empty_box)
            box_roots.append(empty_box)
        channel_memberships.append((ext_paths,self_paths))
        if row['pair'] == 'isolated': exclude(mover, [support_ext])
    proxy_info = []
    for path in proxy_paths:
        prim = stage.GetPrimAtPath(path)
        instance = prim.GetParent()
        if not prim.IsInstanceProxy() or not instance.IsInstance():
            raise ValueError('Shared instance lost: '+path)
        proxy_info.append(dict(path=path, prototype=str(instance.GetPrototype().GetPath())))
    return dict(cases=rows, collision_instances=proxy_info, channel_memberships=channel_memberships,
                box_roots=box_roots,terrain_paths=terrain_paths,
                boundary='USD authoring only until device-owned motion verifies proxy-target filtering; no default runtime changes',
                dt=.005, mass=1., initial_vx=2., settle_steps=40, slide_steps=160)
