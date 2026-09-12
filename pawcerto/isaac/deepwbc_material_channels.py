"""Supported-API contact-pair representation of DeepWBC's signed friction.

Geometry stays instanced; the policy's original sampled friction is unchanged.
Only the fixed original opposing terrain/box coefficient 1 is represented here.
"""
import torch


def discover_channel_masks(materials, collision_count):
    """Read role masks from the actual f=0 coefficients, never USD shape order."""
    if materials.ndim != 3 or materials.shape[1:] != (2*collision_count, 3):
        raise ValueError(f'Expected {2*collision_count} actual collision shapes, got {tuple(materials.shape)}')
    if not torch.isfinite(materials).all() or not torch.equal(materials[...,0], materials[...,1]):
        raise ValueError('Unexpected initial friction payload')
    external = materials[...,0] == .5
    internal = materials[...,0] == 0.
    if not torch.all(external.sum(1)==collision_count) or not torch.all(internal.sum(1)==collision_count):
        raise ValueError('Native material readback does not preserve both f=0 channel bindings')
    if not torch.all(external | internal): raise ValueError('Unclassified native collision material')
    return external, internal


def channel_material_properties(materials, friction, masks):
    if friction.shape != (materials.shape[0],1): raise ValueError('Expected one friction sample per environment')
    friction = friction.to(device=materials.device, dtype=torch.float32)
    if not torch.isfinite(friction).all() or torch.any(friction < -1):
        raise ValueError('External channel requires finite original friction >= -1')
    external, internal = (m.to(materials.device) for m in masks)
    if external.shape != materials.shape[:2] or internal.shape != external.shape:
        raise ValueError('Material channel masks do not match native shape count')
    result = materials.to(dtype=torch.float32).clone().contiguous()
    value = torch.where(external, (friction+1)/2, friction.clamp_min(0))
    result[...,:2] = value[...,None]
    return result


def author_material_channels(stage, robot_roots, box_roots, terrain_colliders):
    """Before physics parsing, duplicate collision references and filter pairs.

    Author relationships on real rigid bodies or terrain colliders; instance
    proxies occur only as relationship targets. Existing exclusions are retained.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    def make_material(path, coefficient, mode):
        material = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(coefficient); api.CreateDynamicFrictionAttr(coefficient)
        api.CreateRestitutionAttr(0.)
        material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
        material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set(mode)
        return material

    def bind(prim, material):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')

    pending_filters = {}

    def exclude(prim, targets):
        if prim.IsInstanceProxy(): raise ValueError('Cannot author pair filters on a proxy')
        key = str(prim.GetPath())
        if key not in pending_filters:
            old = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            pending_filters[key] = (prim, list(old))
        pending_filters[key][1].extend(targets)

    terrain = [stage.GetPrimAtPath(p) for p in terrain_colliders]
    if any(not p or p.IsInstanceProxy() or not p.HasAPI(UsdPhysics.CollisionAPI) for p in terrain):
        raise ValueError('Expected real terrain collider prims')
    ordinary = make_material('/World/DeepWBCOrdinaryMaterial', 1., 'average')
    for prim in terrain: bind(prim, ordinary)
    # Collect before mutation: cloned environments may inherit env_0 additions.
    source_sets = []
    for root in robot_roots:
        robot = stage.GetPrimAtPath(root)
        colliders = [p for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        bodies = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        source_sets.append((colliders, bodies))
    counts, prototypes, channels = [], set(), []
    for env_i, ((colliders, bodies), box_path) in enumerate(zip(source_sets, box_roots, strict=True)):
        material_root = str(stage.GetPrimAtPath(robot_roots[env_i]).GetParent().GetPath())+'/DeepWBCMaterials'
        ext_mat = make_material(material_root+'/External', .5, 'multiply')
        self_mat = make_material(material_root+'/Self', 0., 'average')
        external_paths, self_paths = [], []
        seen = set()
        for collider in colliders:
            instance = collider
            while instance.IsInstanceProxy(): instance = instance.GetParent()
            if instance.GetPath() in seen:
                raise ValueError('Expected one collider per source geometry root: '+str(collider.GetPath()))
            seen.add(instance.GetPath())
            duplicate_path = str(instance.GetPath())+'_DeepWBCSelf'
            duplicate = stage.GetPrimAtPath(duplicate_path)
            if not duplicate:
                duplicate = stage.DefinePrim(duplicate_path, instance.GetTypeName())
                references = None
                for spec in instance.GetPrimStack():
                    refs = spec.referenceList.GetAppliedItems()
                    if refs:
                        references = [Sdf.Reference(Sdf.ComputeAssetPathRelativeToLayer(spec.layer, r.assetPath), r.primPath, r.layerOffset) for r in refs]
                        break
                if instance.IsInstance():
                    if not references: raise ValueError('No original shared collision reference')
                    duplicate.GetReferences().SetReferences(references)
                else:
                    duplicate.SetMetadata('apiSchemas', instance.GetMetadata('apiSchemas'))
                for attr in instance.GetAttributes():
                    if attr.HasAuthoredValueOpinion():
                        duplicate.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
                duplicate.SetInstanceable(instance.IsInstance())
            bind(instance, ext_mat); bind(duplicate, self_mat)
            self_path = str(collider.GetPath()).replace(str(instance.GetPath()), duplicate_path, 1)
            self_collider = stage.GetPrimAtPath(self_path)
            if instance.IsInstance() and (not self_collider.IsInstanceProxy() or duplicate.GetPrototype() != instance.GetPrototype()):
                raise ValueError(f'Collision duplication changed shared prototype: {instance.GetPath()} {instance.GetPrototype().GetPath()} vs {duplicate.GetPrototype().GetPath()}')
            if not self_collider.HasAPI(UsdPhysics.CollisionAPI): raise ValueError('Duplicate lost CollisionAPI')
            if UsdGeom.Xformable(duplicate).GetLocalTransformation() != UsdGeom.Xformable(instance).GetLocalTransformation():
                raise ValueError('Collision duplication changed local transform')
            external_paths.append(collider.GetPath()); self_paths.append(self_collider.GetPath())
            if instance.IsInstance(): prototypes.add(str(instance.GetPrototype().GetPath()))
        for body in bodies: exclude(body, external_paths)
        box = stage.GetPrimAtPath(box_path)
        if not box.HasAPI(UsdPhysics.RigidBodyAPI): raise ValueError('Expected actual box rigid body')
        bind(box, ordinary); exclude(box, self_paths)
        channels.append((external_paths,self_paths))
        counts.append(len(colliders))
    # Batch final relationship opinions. Per-target notices can repeatedly
    # register the entire large terrain mesh before its first GPU upload.
    with Sdf.ChangeBlock():
        for prim, targets in pending_filters.values():
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(list(dict.fromkeys(targets)))
    if len(set(counts)) != 1: raise ValueError('Cloned robots have different collision counts')
    groups = author_collision_groups(stage,channels,box_roots,terrain_colliders)
    return dict(collision_count=counts[0], num_envs=len(counts), shared_prototypes=len(prototypes),
                representation='external=(f+1)/2 multiply; self=max(0,f) average; opposing material=1',**groups)


def author_collision_groups(stage, channels, box_roots, terrain_colliders,
                            old_collision_root='/World/collisions'):
    """One group per shape, using the existing PhysX inverted-group mechanism.

    No large terrain mesh receives shape-by-shape FilteredPairs relationships.
    The old environment grouping is replaced, not combined with another set.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1: raise ValueError('Expected the single active physics scene')
    group_root = '/World/DeepWBCContactGroups'
    UsdGeom.Scope.Define(stage, group_root)
    groups = [UsdPhysics.CollisionGroup.Define(stage, group_root+'/Terrain')]
    includes = [list(terrain_colliders)]
    allowed = [{0}]
    for i, ((external, internal), box_root) in enumerate(zip(channels, box_roots, strict=True)):
        e, s, b = len(groups), len(groups)+1, len(groups)+2
        for role in ('External','Self','Box'):
            groups.append(UsdPhysics.CollisionGroup.Define(stage, f'{group_root}/{role}_{i}'))
        includes.extend([list(external), list(internal), [box_root]])
        allowed[0].update((e,b))
        allowed.extend([{0,b}, {s}, {0,e,b}])
    collections = [Usd.CollectionAPI.Apply(g.GetPrim(),'colliders') for g in groups]
    old = stage.GetPrimAtPath(old_collision_root)
    with Sdf.ChangeBlock():
        if old: old.SetActive(False)
        scenes[0].CreateAttribute('physxScene:invertCollisionGroupFilter',Sdf.ValueTypeNames.Bool).Set(True)
        for i, (group, collection) in enumerate(zip(groups, collections, strict=True)):
            collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
            collection.CreateIncludesRel().SetTargets(includes[i])
            group.CreateFilteredGroupsRel().SetTargets([groups[j].GetPath() for j in sorted(allowed[i])])
    # The standard USD table is normal filtering. PhysX's existing scene-wide
    # invert flag complements it; do not label the raw USD table PhysX evidence.
    table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            enabled = not table.IsCollisionEnabled(gi.GetPath(), gj.GetPath())
            if enabled != (j in allowed[i]): raise ValueError('Composed collision-group matrix differs from required pairs')
    queries = [c.ComputeMembershipQuery() for c in collections]
    actual_shapes = set()
    for external, internal in channels: actual_shapes.update(map(str,external)); actual_shapes.update(map(str,internal))
    actual_shapes.update(map(str,terrain_colliders))
    for root in box_roots:
        actual_shapes.update(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root),Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI))
    for path in actual_shapes:
        if sum(q.IsPathIncluded(path) for q in queries) != 1:
            raise ValueError('Collision shape must belong to exactly one active contact group: '+path)
    return dict(group_count=len(groups),single_membership_shape_count=len(actual_shapes),
                matrix_check='standard USD table complemented by current PhysX scene invert flag; physical validation separate')
