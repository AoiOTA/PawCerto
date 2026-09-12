"""Analytic six-axis inertia/frame/lever checks, independent of simulator."""
import unittest
import numpy as np
import torch
from tests.test_solver_force import model, inputs

class SixAxisTest(unittest.TestCase):
    def test_nonzero_lever_and_offdiagonal_inertia(self):
        inertia=np.array([[2.,.3,.1],[.3,3.,.2],[.1,.2,4.]])
        fixture=model(inertia=inertia)
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        com=np.array([.2,-.3,.4]); da=np.array([2.,3.,4.,5.,6.,7.]); applied=np.array([0.,0.,-9.81])
        tensor=lambda a:torch.tensor(a,dtype=torch.float64)
        free=tensor([[[.7,.3,-2.,.2,.1,-.4]]]); reported=free+tensor(da)
        got=fixture.approximate_sensor_wrench(reported,free,link_rotation=tensor(r)[None,None],com_offset_b=tensor(com)[None,None],applied_force_w=tensor(applied)[None,None])
        force=da[:3]+applied
        expected=np.r_[r.T@force,r.T@((r@inertia@r.T)@da[3:]+np.cross(r@com,force))]
        np.testing.assert_allclose(got[0,0].numpy(),expected,atol=1e-12)
        torch.testing.assert_close(fixture.residual_force(reported,free),tensor(da[:3])[None,None],atol=1e-12,rtol=0)

    def test_equal_reported_and_free_leaves_only_applied_gravity(self):
        fixture=model(inertia=np.diag([2.,3.,4.])); data=inputs()
        acceleration=torch.tensor([[[4.,5.,6.,7.,8.,9.]]],dtype=torch.float64)
        gravity=torch.tensor([[[0.,0.,-9.81]]],dtype=torch.float64)
        got=fixture.approximate_sensor_wrench(acceleration,acceleration,
            link_rotation=data['link_rotation'],com_offset_b=torch.zeros(1,1,3,dtype=torch.float64),applied_force_w=gravity)
        expected=torch.cat((gravity,torch.zeros_like(gravity)),-1)
        torch.testing.assert_close(got,expected,atol=0,rtol=0)

if __name__=='__main__': unittest.main()
