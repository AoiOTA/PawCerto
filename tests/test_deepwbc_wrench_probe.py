"""Independent analytic mechanics checks, no simulator import."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('wrench_probe',Path(__file__).parents[1]/'scripts/analyze_deepwbc_wrench.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_static_and_freefall():
    args=(np.array(2.),np.eye(3),np.array([0.,0.,0.,1.]),np.zeros(3))
    np.testing.assert_array_equal(m.net_wrench(*args,np.zeros(6),np.zeros(3)),np.zeros(6))
    np.testing.assert_allclose(m.net_wrench(*args,np.array([0,0,-9.81,0,0,0]),np.array([1,0,0])),[0,0,-19.62,0,19.62,0])

def test_rotated_asymmetric_inertia_and_gyroscopic_term():
    # Body rotated 90deg around z: world omega (-2,1,3) = body (1,2,3).
    q=np.array([0,0,2**-.5,2**-.5]); w=np.array([-2.,1.,3.])
    got=m.net_wrench(np.array(2.),np.diag([2.,3.,4.]),q,w,np.array([0,2,0,-2,1,3]),np.array([0.,1.,0.]))
    # body F=(4,0,0), I alpha=(2,6,12), w x Iw=(6,-6,2), r x F=(0,0,-4)
    np.testing.assert_allclose(got,[4,0,0,8,0,10],atol=1e-12)
