"""Local COM uncertainty, fixed plate separation and frozen design semantics."""
import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pawcerto.as2_pretraining_assets import inertial_properties, transform_variant
from pawcerto.mujoco.as2_piper_asset import CONFIG, assemble, merge_mount_plate_inertia, transform


@pytest.fixture
def source():
    return assemble(), json.loads(CONFIG.read_text())


def shift_tensor(mass, displacement):
    return mass*(np.dot(displacement,displacement)*np.eye(3)-np.outer(displacement,displacement))


def combine(m,c,i,p,pc,pi):
    com=(m*c+p*pc)/(m+p)
    return m+p,com,i+pi+shift_tensor(m,c-com)+shift_tensor(p,pc-com)


@pytest.mark.parametrize('density,base_ratio', [(1,1),(20/17.64,.9),(1.1,1.1)])
def test_base_offset_keeps_plate_mass_com_and_tensor(source,density,base_ratio):
    root,config=source
    base=root.find("link[@name='base_link']")
    original=inertial_properties(base)
    plate=dict(size_m=[.16,.18,.006],density_kg_m3=2700,com_base_m=[-.005,0,.089254550594])
    config['rail_mount']={'plate':plate}
    merge_mount_plate_inertia(base,plate)
    before=ET.tostring(root)
    delta=np.array([.01,-.006,.004])
    result=transform_variant(root,dict(as2_mass_ratio=density,base_mass_ratio=base_ratio,
                                      link_com_offsets_m={'base_link':delta.tolist()}),config)
    assert ET.tostring(root)==before
    pm=.46656;pc=np.array(plate['com_base_m'])
    pi=pm/12*np.diag([.18**2+.006**2,.16**2+.006**2,.16**2+.18**2])
    m,c,i=original;ratio=density*base_ratio
    expected=combine(m*ratio,c+delta,i*ratio,pm,pc,pi)
    actual=inertial_properties(result.find("link[@name='base_link']"))
    for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=2e-13,rtol=0)
    # Recover the plate's complete moments by removing the expected original base.
    total,centre,tensor=actual
    np.testing.assert_allclose(total-m*ratio,pm,atol=1e-14)
    np.testing.assert_allclose((total*centre-m*ratio*(c+delta))/pm,pc,atol=1e-13)
    residual=tensor+shift_tensor(total,centre)-i*ratio-shift_tensor(m*ratio,c+delta)
    np.testing.assert_allclose(residual,pi+shift_tensor(pm,pc),atol=1e-13)


def test_link_local_offset_preserves_rotated_central_tensor_and_geometry(source):
    root,config=source
    link=root.find("link[@name='piper_link2']")
    link.find('inertial/origin').set('rpy','.2 -.3 .4')
    m,c,i=inertial_properties(link)
    delta=np.array([.002,-.004,.005])
    out=transform_variant(root,dict(piper_mass_ratios={'piper_link2':1.1},
                                   link_com_offsets_m={'piper_link2':delta.tolist()}),config)
    changed=out.find("link[@name='piper_link2']")
    actual=inertial_properties(changed)
    for a,e in zip(actual,(m*1.1,c+delta,i*1.1)):np.testing.assert_allclose(a,e,atol=1e-14)
    assert [ET.tostring(x) for x in changed.findall('collision')]==[ET.tostring(x) for x in link.findall('collision')]
    assert [ET.tostring(x) for x in out.findall('joint')]==[ET.tostring(x) for x in root.findall('joint')]


def test_payload_is_merged_after_link_offset_in_tcp_axes(source):
    root,config=source
    config['tcp']['rpy']=[.2,-.3,.4]
    name=config['tcp']['body'];link=root.find(f"link[@name='{name}']")
    m,c,i=inertial_properties(link);delta=np.array([.004,-.003,.002])
    payload_offset=np.array([.02,-.01,.05]);pose=transform(config['tcp']['xyz'],config['tcp']['rpy'])
    pc=pose[:3,3]+pose[:3,:3]@payload_offset
    result=transform_variant(root,dict(piper_mass_ratios={name:1.1},link_com_offsets_m={name:delta.tolist()},
                                      payload=dict(mass_kg=1.3,model='point',com_tcp_m=payload_offset.tolist())),config)
    expected=combine(m*1.1,c+delta,i*1.1,1.3,pc,np.zeros((3,3)))
    for a,e in zip(inertial_properties(result.find(f"link[@name='{name}']")),expected):
        np.testing.assert_allclose(a,e,atol=1e-14)


@pytest.mark.parametrize('offsets', [{'no_such_link':[0,0,0]},{'world':[0,0,0]},
    {'base_link':[0,0]},{'base_link':[[0,0,0]]},{'base_link':[0,float('nan'),0]},
    {'base_link':[0,0,float('inf')]},[],{'base_link':1}])
def test_invalid_com_names_and_values_rejected(source,offsets):
    root,config=source
    with pytest.raises(ValueError):transform_variant(root,{'link_com_offsets_m':offsets},config)


def test_legacy_nominal_and_leg_scaling_semantics(source):
    root,config=source
    assert ET.tostring(transform_variant(root,{},config))==ET.tostring(root)
    name='FL_thigh';m,c,i=inertial_properties(root.find(f"link[@name='{name}']"))
    delta=np.array([.001,.002,.003])
    changed=transform_variant(root,dict(leg_scale=.95,as2_mass_ratio=1.1,link_com_offsets_m={name:delta.tolist()}),config)
    actual=inertial_properties(changed.find(f"link[@name='{name}']"))
    for a,e in zip(actual,(m*1.1*.95**3,c*.95+delta,i*1.1*.95**5)):
        np.testing.assert_allclose(a,e,atol=1e-14)


def test_finite_design_keeps_six_regressions_and_paired_priors():
    folder=Path(__file__).resolve().parents[1]/'configs'
    old=json.loads((folder/'as2_pretraining_target_domain.json').read_text())
    new=json.loads((folder/'as2_pretraining_target_distribution.json').read_text())
    assert len(new['variants'])==22
    for old_v,new_v in zip(old['variants'],new['variants']):
        assert old_v['name']==new_v['name'] and old_v['parameters']==new_v['parameters']
    assert new['variants'][0]['parameters']=={}  # source17.64kg, not20kg nominal
    for a,b in zip(new['variants'][6::2],new['variants'][7::2]):
        a,b=a['parameters'],b['parameters']
        assert a['as2_mass_ratio']==b['as2_mass_ratio']==20/17.64
        for field,bounds in [('base_mass_ratio',(.9,1.1)),('leg_scale',(.95,1.05))]:
            assert bounds[0]<=a[field]<=bounds[1] and bounds[0]<=b[field]<=bounds[1]
            np.testing.assert_allclose(a[field]+b[field],2,atol=1e-15)
        for name in a['piper_mass_ratios']:
            assert .9<=a['piper_mass_ratios'][name]<=1.1
            np.testing.assert_allclose(a['piper_mass_ratios'][name]+b['piper_mass_ratios'][name],2)
        for name in a['link_com_offsets_m']:
            bound=.01 if name=='base_link' else .005
            assert np.max(np.abs(a['link_com_offsets_m'][name]))<=bound
            np.testing.assert_allclose(np.array(a['link_com_offsets_m'][name])+b['link_com_offsets_m'][name],0,atol=1e-17)
        assert 0<=a['payload']['mass_kg']<=2
        np.testing.assert_allclose(a['payload']['mass_kg']+b['payload']['mass_kg'],2)
        assert np.all(np.abs(a['payload']['com_tcp_m'])<=[.02,.02,.05])
        np.testing.assert_allclose(np.array(a['payload']['com_tcp_m'])+b['payload']['com_tcp_m'],0,atol=1e-17)


def test_distribution_rejects_silent_historical_mount(tmp_path):
    from pawcerto.as2_pretraining_assets import build_family
    distribution=Path(__file__).resolve().parents[1]/'configs/as2_pretraining_target_distribution.json'
    with pytest.raises(ValueError,match='matching --nominal-config'):
        build_family(tmp_path/'assets',distribution)
    assert not (tmp_path/'assets').exists()


def test_distribution_accepts_selected_assembly_by_content(tmp_path):
    from pawcerto.as2_pretraining_assets import build_family
    assembly=tmp_path/'assembly.json'
    cfg=json.loads(CONFIG.read_text());cfg['mount']['xyz'][2]=.095
    assembly.write_text(json.dumps(cfg))
    recipe=dict(schema_version=1,description='Assembly identity test',
        parameter_ranges=dict(source=dict(assembly_sha256=hashlib.sha256(assembly.read_bytes()).hexdigest())),
        variants=[dict(name='nominal',assumption='Selected assembly',parameters={})])
    path=tmp_path/'family.json';path.write_text(json.dumps(recipe))
    result=build_family(tmp_path/'assets',path,nominal_config_path=assembly)
    assert result['source']['nominal_config_sha256']==recipe['parameter_ranges']['source']['assembly_sha256']
    tree=ET.parse(result['variants'][0]['source_urdf_path'])
    xyz=tree.find("joint[@name='piper_mount']/origin").get('xyz')
    np.testing.assert_allclose(np.fromstring(xyz,sep=' '),[0,0,.095],atol=1e-15)
