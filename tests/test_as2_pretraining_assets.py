"""CPU checks for the finite family delivered to the existing USD consumer."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pawcerto.as2_pretraining_assets import build_family, inertial_properties, merge_payload, transform_variant
from pawcerto.mujoco.as2_piper_asset import CONFIG


@pytest.fixture(scope='module')
def family(tmp_path_factory):
    return build_family(tmp_path_factory.mktemp('as2_family'))


def test_all_variants_compile_and_match_static_fk(family):
    assert len(family['joint_names']) == 18
    assert len(family['body_names']) == 28
    assert family['collision_count'] == 42
    for variant in family['variants']:
        assert variant['merged_urdf_path'] == variant['urdf_path']
        assert variant['usd_path'] is None
        model = mujoco.MjModel.from_xml_path(variant['source_urdf_path'])
        data = mujoco.MjData(model)
        assert model.nq == 25 and model.nv == 24
        data.qpos[:3] = 0
        data.qpos[3:7] = [1,0,0,0]
        for name,q in zip(family['joint_names'],family['default_joint_positions']):
            data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_kinematics(model,data)
        np.testing.assert_allclose(model.body_mass.sum(),variant['total_mass_kg'],atol=1e-8)
        com = sum(v['mass_kg'] * data.xipos[model.body(n).id] for n,v in variant['per_link'].items()) / variant['total_mass_kg']
        np.testing.assert_allclose(com,variant['nominal_fk']['whole_com_m'],atol=1e-7)
        for name,xyz in variant['nominal_fk']['feet_m'].items():
            np.testing.assert_allclose(data.body(name).xpos,xyz,atol=1e-7)
        b=data.body(family['tcp']['body'])
        tcp=b.xpos+b.xmat.reshape(3,3)@np.array(family['tcp']['xyz'])
        np.testing.assert_allclose(tcp,np.array(variant['nominal_fk']['tcp_pose'])[:3,3],atol=1e-6)
        for name,record in variant['per_link'].items():
            b=model.body(name)
            rot=Rotation.from_quat(b.iquat[[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(rot@np.diag(b.inertia)@rot.T,record['inertia_com_link_kg_m2'],atol=1e-7)


def test_density_geometry_and_point_payload(family):
    variants={v['name']:v for v in family['variants']}
    nominal=variants['nominal']
    np.testing.assert_allclose(nominal['total_mass_kg'],22.347,atol=1e-12)
    for name,s in [('legs_095',.95),('legs_105',1.05)]:
        v=variants[name]
        np.testing.assert_allclose(v['total_mass_kg'],22.347+8.84*(s**3-1),atol=1e-12)
        for part in ['hip','thigh','calf','foot']:
            n=nominal['per_link']['FL_'+part]; r=v['per_link']['FL_'+part]
            np.testing.assert_allclose(r['mass_kg'],n['mass_kg']*s**3)
            np.testing.assert_allclose(r['com_local_m'],np.array(n['com_local_m'])*s)
            np.testing.assert_allclose(r['inertia_com_link_kg_m2'],np.array(n['inertia_com_link_kg_m2'])*s**5)
        original=ET.parse(nominal['urdf_path']); altered=ET.parse(v['urdf_path'])
        for joint in ['FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint']:
            assert ET.tostring(original.find(f"joint[@name='{joint}']")) == ET.tostring(altered.find(f"joint[@name='{joint}']"))
        # Rotated collision origins/axes retain rotation under isotropic scaling.
        for a,b in zip(original.findall("link[@name='FL_thigh']/collision"),altered.findall("link[@name='FL_thigh']/collision")):
            assert a.find('origin').get('rpy') == b.find('origin').get('rpy')
            for key in ['radius','length']:
                x,y=a.find('geometry')[0],b.find('geometry')[0]
                if x.get(key) is not None: np.testing.assert_allclose(float(y.get(key)),float(x.get(key))*s)
    payload=variants['tcp_point_1kg']
    c=np.array(nominal['nominal_fk']['whole_com_m']);tcp=np.array(nominal['nominal_fk']['tcp_pose'])[:3,3]
    np.testing.assert_allclose(payload['nominal_fk']['whole_com_m'],(22.347*c+tcp)/23.347,atol=1e-12)
    assert payload['total_mass_kg']==pytest.approx(23.347)


def test_offset_rigid_payload_parallel_axis_and_nominal_copy():
    root=ET.fromstring('<robot><link name="test"><inertial><origin xyz=".1 .2 .3" rpy=".2 .3 .4"/><mass value="2"/><inertia ixx=".02" ixy="0" ixz="0" iyy=".03" iyz="0" izz=".04"/></inertial></link></robot>')
    link=root.find('link');m,c,I=inertial_properties(link)
    tcp=dict(xyz=[.3,-.1,.2],rpy=[0,0,np.pi/2]); offset=np.array([.1,.2,.3]);R=Rotation.from_euler('xyz',tcp['rpy']).as_matrix();Ip=np.diag([.01,.015,.02]);r=np.array(tcp['xyz'])+R@offset
    merge_payload(link,tcp,dict(mass_kg=1,model='rigid',com_tcp_m=offset.tolist(),inertia_com_tcp_kg_m2=Ip.tolist()))
    mt,ct,It=inertial_properties(link)
    # Compare second moment about the original link origin, independently of merged COM.
    shift=lambda x:np.dot(x,x)*np.eye(3)-np.outer(x,x)
    np.testing.assert_allclose(It+mt*shift(ct),I+m*shift(c)+R@Ip@R.T+shift(r),atol=1e-12)
    np.testing.assert_allclose(ct,(m*c+r)/3,atol=1e-12)
    with pytest.raises(ValueError,match='Point payload'):
        merge_payload(link,tcp,dict(mass_kg=1,model='point',inertia_com_tcp_kg_m2=Ip.tolist()))


def test_invalid_parameters_and_nominal_identity(family):
    nominal=json.loads(CONFIG.read_text());root=ET.parse(family['variants'][0]['source_urdf_path']).getroot();before=ET.tostring(root)
    assert ET.tostring(transform_variant(root,{},nominal)) == before
    assert ET.tostring(root) == before
    for parameters in [{'leg_scale':0},{'base_mass_ratio':float('nan')},{'piper_mass_ratios':{'wrong_link':1.1}},{'payload':{'mass_kg':-1}}]:
        with pytest.raises(ValueError): transform_variant(root,parameters,nominal)


def test_target_as2_density_preserves_piper_and_composes_geometry(tmp_path):
    from pawcerto.as2_pretraining_assets import ROOT
    target = build_family(tmp_path/'target', ROOT/'configs/as2_pretraining_target_domain.json')
    variants = {v['name']:v for v in target['variants']}
    nominal = variants['nominal']
    density = variants['as2_density_20kg']
    factor = 20/17.64
    for name,record in nominal['per_link'].items():
        actual = density['per_link'][name]
        ratio = 1 if name.startswith('piper_') else factor
        np.testing.assert_allclose(actual['mass_kg'],record['mass_kg']*ratio,atol=1e-12)
        np.testing.assert_allclose(actual['inertia_com_link_kg_m2'],np.array(record['inertia_com_link_kg_m2'])*ratio,atol=1e-12)
        np.testing.assert_array_equal(actual['com_local_m'],record['com_local_m'])
    for variant in target['variants']:
        expected=target['parameter_ranges']['candidate_mass_expectations_kg'][variant['name']]
        actual_as2=sum(v['mass_kg'] for n,v in variant['per_link'].items() if not n.startswith('piper_'))
        np.testing.assert_allclose(actual_as2,expected['as2'],atol=1e-12)
        np.testing.assert_allclose(variant['total_mass_kg'],expected['total'],atol=1e-12)
        scale=variant['parameters'].get('leg_scale',1)
        ratio=variant['parameters'].get('as2_mass_ratio',1)
        np.testing.assert_allclose(variant['per_link']['FL_thigh']['inertia_com_link_kg_m2'],np.array(nominal['per_link']['FL_thigh']['inertia_com_link_kg_m2'])*ratio*scale**5,atol=1e-12)
    test_all_variants_compile_and_match_static_fk(target)


def test_mujoco_family_uses_original_ground_motors_without_integration(tmp_path, monkeypatch):
    from pawcerto.as2_pretraining_assets import ROOT, build_mujoco_family
    from pawcerto.mujoco.as2_piper_asset import build
    def forbidden(*args, **kwargs):
        raise AssertionError('Static conversion must not execute dynamics')
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    target = build_family(tmp_path/'family', ROOT/'configs/as2_pretraining_target_domain.json')
    protected = [tmp_path/'family/manifest.json']
    for v in target['variants']:
        protected.extend([Path(v[k]) for k in ('source_urdf_path','merged_urdf_path','preparation_path')])
    original = {p:p.read_bytes() for p in protected}
    converted = build_mujoco_family(tmp_path/'family')
    for p,content in original.items(): assert p.read_bytes() == content
    for source, output in zip(target['variants'], converted['variants']):
        model = mujoco.MjModel.from_xml_path(output['mujoco_path'])
        report = json.loads(Path(output['validation_path']).read_text())
        assert report['physics_steps'] == 0
        assert (model.nq,model.nv,model.nu) == (25,24,18)
        assert model.geom('ground').type[0] == mujoco.mjtGeom.mjGEOM_PLANE
        assert model.key('default').id == 0
        assert model.site('tcp').id >= 0
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert model.opt.timestep == .005
        np.testing.assert_allclose(model.body_mass.sum(), source['total_mass_kg'], atol=1e-10)
        for name,entry in source['per_link'].items():
            np.testing.assert_allclose(model.body(name).mass[0],entry['mass_kg'],atol=1e-12,rtol=0)
    # Supplied-tree support must not alter historical default model construction.
    build(tmp_path/'default', static_only=True)
    actual = ET.parse(tmp_path/'default/robot.xml').getroot()
    saved = ET.parse(ROOT/'reference/as2_piper/robot.xml').getroot()
    # MuJoCo serializer versions differ only in redundant inferred STL MIME tags.
    for tree in (actual, saved):
        for mesh in tree.findall('asset/mesh'):
            if mesh.get('content_type') == 'model/stl' and mesh.get('file', '').endswith('.stl'):
                del mesh.attrib['content_type']
    assert ET.tostring(actual) == ET.tostring(saved)
