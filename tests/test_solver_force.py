"""CPU analytic checks for PhysX velocity preprocessing and inertia frames."""
import unittest
import numpy as np
import torch

from pawcerto.isaac.solver_force import StockSolverForce
from scripts.analyze_stock_solver_force import com_bias_acceleration


def model(parents=(-1,), inertia=None, dofs=0, dof_limits=None):
    bodies=len(parents)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64)
    return StockSolverForce(parents,torch.ones(1,bodies,dtype=torch.float64),
        body_inertias=(torch.eye(3,dtype=torch.float64).repeat(1,bodies,1,1) if inertia is None else tensor(inertia).reshape(1,bodies,3,3)),
        dt=.005,dof_max_velocity=(torch.full((1,dofs),1000.,dtype=torch.float64) if dof_limits is None else tensor(dof_limits).reshape(1,dofs)),
        body_max_linear_velocity=torch.full((1,bodies),1000.,dtype=torch.float64),
        body_max_angular_velocity=torch.full((1,bodies),1000.,dtype=torch.float64),
        body_linear_damping=torch.zeros(1,bodies),body_angular_damping=torch.zeros(1,bodies),
        armature=torch.zeros(1,dofs),disable_gravity=torch.zeros(1,bodies),
        retain_accelerations=torch.zeros(1,bodies),gyroscopic_forces=torch.ones(1,bodies),
        external_forces_every_iteration=False)


def inputs(bodies=1,dofs=0):
    dim=dofs+6
    return dict(mass_matrix=torch.eye(dim,dtype=torch.float64)[None],
        coriolis=torch.zeros(1,dim,dtype=torch.float64),gravity=torch.zeros(1,dim,dtype=torch.float64),
        jacobian=torch.eye(6,dim,dtype=torch.float64).repeat(1,bodies,1,1),
        com_position=torch.zeros(1,bodies,3,dtype=torch.float64),
        com_velocity=torch.zeros(1,bodies,6,dtype=torch.float64),
        link_rotation=torch.eye(3,dtype=torch.float64).repeat(1,bodies,1,1),
        dof_velocity=torch.zeros(1,dofs,dtype=torch.float64),actuation=torch.zeros(1,dofs,dtype=torch.float64),
        external_wrench_terms=torch.zeros(1,dtype=torch.float64))


class SolverForceTest(unittest.TestCase):
    def test_below_and_at_gyro_threshold_match_unmodified_solve_exactly(self):
        fixture=model(inertia=np.diag([2.,3.,4.]),dofs=2)
        for speed in (0.,199.,200.):
            with self.subTest(speed=speed):
                data=inputs(dofs=2)
                data['com_velocity'][0,0,3:]=torch.tensor([.6*speed,.8*speed,0.],dtype=torch.float64)
                data['coriolis'][0]=torch.arange(8,dtype=torch.float64)*.25
                data['gravity'][0,2]=9.81
                data['actuation'][0]=torch.tensor([.1,-.1],dtype=torch.float64)
                force=torch.cat((torch.zeros(1,6,dtype=torch.float64),data['actuation']),-1)
                expected=torch.linalg.solve(data['mass_matrix'],(force-data['coriolis']-data['gravity'])[...,None])[...,0]
                prediction,qdd,bias=fixture.predict(**data)
                torch.testing.assert_close(qdd,expected,rtol=0,atol=0)
                torch.testing.assert_close(bias,torch.zeros_like(bias),rtol=0,atol=0)
                torch.testing.assert_close(prediction,(data['jacobian']@expected[:,None,:,None])[...,0],rtol=0,atol=0)

    def test_high_speed_rigid_body_matches_gyro_euler_equation_in_body_frame(self):
        # Full, off-diagonal body-frame inertia and a nonidentity link rotation
        # distinguish body coordinates from COM principal-axis coordinates.
        inertia=np.array([[2.,.3,.1],[.3,3.,.2],[.1,.2,4.]])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        omega=np.array([180.,160.,100.])
        world=rotation@inertia@rotation.T
        data=inputs();data['mass_matrix'][0,3:,3:]=torch.from_numpy(world)
        data['coriolis'][0,3:]=torch.from_numpy(np.cross(omega,world@omega))
        data['com_velocity'][0,0,3:]=torch.from_numpy(omega)
        data['link_rotation'][0,0]=torch.from_numpy(rotation)
        fixture=model(inertia=inertia)
        prediction,qdd,bias=fixture.predict(**data)
        limited=omega*(200./np.linalg.norm(omega))
        expected=np.linalg.solve(world,-np.cross(limited,world@limited))
        np.testing.assert_allclose(qdd[0,3:].numpy(),expected,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(prediction[0,0,3:].numpy(),expected,rtol=1e-12,atol=1e-10)
        self.assertGreater(np.max(np.abs(expected-np.linalg.solve(world,-data['coriolis'][0,3:].numpy()))),100.)
        wrong=dict(data,link_rotation=torch.eye(3,dtype=torch.float64).reshape(1,1,3,3))
        self.assertGreater(float(torch.max(torch.abs(fixture.predict(**wrong)[1]-qdd))),100.)
        torch.testing.assert_close(bias,torch.zeros_like(bias),rtol=0,atol=0)

    def test_high_gyro_speed_does_not_clip_recursive_com_bias(self):
        fixture=model(parents=(-1,0));data=inputs(bodies=2)
        omega=torch.tensor([180.,160.,100.],dtype=torch.float64)
        r=torch.tensor([.4,-.2,.3],dtype=torch.float64)
        data['com_position'][0,1]=r
        data['com_velocity'][0,:,3:]=omega
        data['com_velocity'][0,1,:3]=torch.linalg.cross(omega,r)
        # Isotropic inertias make gyro zero; centripetal CoM bias stays nonzero.
        prediction,qdd,bias=fixture.predict(**data)
        expected=torch.linalg.cross(omega,torch.linalg.cross(omega,r))
        torch.testing.assert_close(bias[0,1,:3],expected,rtol=0,atol=0)
        torch.testing.assert_close(qdd,torch.zeros_like(qdd),rtol=0,atol=0)
        torch.testing.assert_close(prediction,bias,rtol=0,atol=0)
        limited=omega*(200./torch.linalg.vector_norm(omega))
        self.assertGreater(float(torch.max(torch.abs(expected-torch.linalg.cross(limited,torch.linalg.cross(limited,r))))),100.)

    def test_global_dof_scaling_matches_full_inverse_dynamics_for_both_signs(self):
        # Three physical bodies, two revolute coordinates: a fast first DOF
        # also scales the non-over-limit second DOF, while root twist is fixed.
        positions=np.array([[0.,0.,0.],[.4,0.,0.],[.4,.3,0.]])
        inertias=np.array([np.diag([.2,.3,.4]),np.diag([.4,.5,.6]),np.diag([.3,.7,.8])])
        jacobian=np.zeros((3,6,8))
        for b,r in enumerate(positions):
            skew=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
            jacobian[b,:3,:3]=np.eye(3);jacobian[b,:3,3:6]=-skew;jacobian[b,3:,3:6]=np.eye(3)
        for b in (1,2):
            jacobian[b,:3,6]=np.cross([0.,0.,1.],positions[b]);jacobian[b,3:,6]=[0.,0.,1.]
        jacobian[2,:3,7]=np.cross([1.,0.,0.],positions[2]-positions[1]);jacobian[2,3:,7]=[1.,0.,0.]
        mass_matrix=np.zeros((8,8))
        for j,inertia in zip(jacobian,inertias):
            spatial=np.zeros((6,6));spatial[:3,:3]=np.eye(3);spatial[3:,3:]=inertia
            mass_matrix+=j.T@spatial@j
        root_twist=np.array([.3,.4,-.2,.2,-.1,.3])
        def inverse_dynamics(nu):
            velocity=np.einsum('bkd,d->bk',jacobian,nu)
            bias=com_bias_acceleration(positions[None],velocity[None],{1:0,2:1})[0]
            omega=velocity[:,3:]
            gyro=np.cross(omega,np.einsum('bij,bj->bi',inertias,omega))
            wrench=np.concatenate((bias[:,:3],np.einsum('bij,bj->bi',inertias,bias[:,3:])+gyro),axis=-1)
            return np.einsum('bkd,bk->d',jacobian,wrench),velocity,bias
        fixture=model(parents=(-1,0,1),inertia=inertias,dofs=2,dof_limits=[20.,10.])
        for fast in (40.,-40.):
            with self.subTest(fast=fast):
                raw=np.array([fast,4.])
                # Source public-C preprocessing clips only the positive side.
                qd_c=np.array([20. if fast>0 else -40.,4.])
                qd_f=np.array([20. if fast>0 else -20.,2.])
                c_public,velocity_c,_=inverse_dynamics(np.r_[root_twist,qd_c])
                c_fd,velocity_f,bias_fd=inverse_dynamics(np.r_[root_twist,qd_f])
                np.testing.assert_array_equal(velocity_f[0],root_twist)
                data=inputs(bodies=3,dofs=2)
                data.update(mass_matrix=torch.from_numpy(mass_matrix)[None],coriolis=torch.from_numpy(c_public)[None],
                    jacobian=torch.from_numpy(jacobian)[None],com_position=torch.from_numpy(positions)[None],
                    com_velocity=torch.from_numpy(velocity_c)[None],dof_velocity=torch.from_numpy(raw)[None])
                data['actuation'][0]=torch.tensor([.1,-.1],dtype=torch.float64)
                qdd_expected=np.linalg.solve(mass_matrix,np.r_[np.zeros(6),[.1,-.1]]-c_fd)
                acceleration_expected=np.einsum('bkd,d->bk',jacobian,qdd_expected)+bias_fd
                prediction,qdd,bias=fixture.predict(**data)
                np.testing.assert_allclose(qdd[0].numpy(),qdd_expected,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(prediction[0].numpy(),acceleration_expected,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(bias[0].numpy(),bias_fd,rtol=1e-12,atol=1e-10)
                self.assertGreater(np.max(np.abs(c_fd-c_public)),1.)

    def test_body_speed_damping_uses_fd_velocity_after_dof_scaling(self):
        fixture=model(parents=(-1,0),dofs=1,dof_limits=[10.]);data=inputs(bodies=2,dofs=1)
        data['jacobian'][0,1,3,6]=1.
        data['dof_velocity'][0,0]=-1200.
        # Negative overshoot remains raw in public C. FD scales it to -10,
        # so the body's supported FD speed must pass rather than reject -1200.
        data['com_velocity'][0,1,3]=-1200.
        prediction,_,_=fixture.predict(**data)
        self.assertTrue(torch.isfinite(prediction).all())
        # Root twist is unchanged by DOF scaling. Public-C child speed is now
        # only -100, but FD child speed is 1090: damping must use the latter.
        data['com_velocity'][0,:,3]+=1100.
        j=data['jacobian']
        data['mass_matrix']=(j.transpose(-1,-2)@j).sum(1)
        prediction,qdd,bias=fixture.predict(**data)
        expected=torch.zeros_like(prediction)
        expected[0,0,3]=-(1100.-1000.)/.005
        expected[0,1,3]=-(1090.-1000.)/.005
        torch.testing.assert_close(prediction,expected,rtol=1e-12,atol=1e-10)
        self.assertAlmostEqual(float(qdd[0,6]),2000.,places=9)
        torch.testing.assert_close(bias,torch.zeros_like(bias),rtol=0,atol=0)

    def test_body_speed_damping_matches_mass_and_world_inertia_for_all_bodies(self):
        # All three fixed bodies share a COM/twist, but their speed limits,
        # masses and full inertias differ. Each body's damping must contribute
        # through J.T, including both root and children.
        masses=np.array([2.,3.,5.])
        inertias=np.array([[[2.,.3,.1],[.3,3.,.2],[.1,.2,4.]],
                          np.diag([.4,.5,.6]),np.diag([.3,.7,.8])])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        world=rotation@inertias@rotation.T
        linear_limits=np.array([2.,3.,4.]);angular_limits=np.array([210.,220.,250.])
        fixture=model(parents=(-1,0,1),inertia=inertias)
        fixture.masses=torch.from_numpy(masses)[None]
        fixture.body_max_linear_velocity=torch.from_numpy(linear_limits)[None]
        fixture.body_max_angular_velocity=torch.from_numpy(angular_limits)[None]
        for mode in ('linear','angular','both','below','at'):
            with self.subTest(mode=mode):
                velocity=np.array([3.,4.,0.]) if mode in ('linear','both') else np.zeros(3)
                omega=np.array([180.,160.,100.]) if mode in ('angular','both') else np.zeros(3)
                if mode=='below':velocity=np.array([1.,0.,0.]);omega=np.array([50.,0.,0.])
                if mode=='at':velocity=np.array([2.,0.,0.]);omega=np.array([210.,0.,0.])
                data=inputs(bodies=3)
                data['mass_matrix'][0,:3,:3]*=masses.sum()
                data['mass_matrix'][0,3:,3:]=torch.from_numpy(world.sum(0))
                data['link_rotation'][0]=torch.from_numpy(np.tile(rotation,(3,1,1)))
                data['com_velocity'][0,:,:3]=torch.from_numpy(velocity)
                data['com_velocity'][0,:,3:]=torch.from_numpy(omega)
                public_gyro=np.cross(omega,np.einsum('bij,j->bi',world,omega)).sum(0)
                data['coriolis'][0,3:]=torch.from_numpy(public_gyro)
                clipped=omega/max(1.,np.linalg.norm(omega)*.005)
                fd_gyro=np.cross(clipped,np.einsum('bij,j->bi',world,clipped)).sum(0)
                # Independent per-body scalar formula; zero-speed cases do
                # not divide. Angular damping uses full omega, not clipped.
                linear=np.zeros(3);angular=np.zeros(3)
                for mass,inertia,llim,alim in zip(masses,world,linear_limits,angular_limits):
                    if np.linalg.norm(velocity)>llim:linear+=mass*velocity*(1.-llim/np.linalg.norm(velocity))/.005
                    if np.linalg.norm(omega)>alim:angular+=inertia@omega*(1.-alim/np.linalg.norm(omega))/.005
                rhs=-np.r_[linear,fd_gyro+angular]
                expected=np.linalg.solve(data['mass_matrix'][0].numpy(),rhs)
                prediction,qdd,bias=fixture.predict(**data)
                np.testing.assert_allclose(qdd[0].numpy(),expected,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(prediction[0].numpy(),np.tile(expected,(3,1)),rtol=1e-12,atol=1e-10)
                torch.testing.assert_close(bias,torch.zeros_like(bias),rtol=0,atol=0)
                if mode in ('below','at'):
                    # Principal-axis omega in these fixtures gives zero gyro
                    # for children; root has off-diagonal inertia. Reconstruct
                    # the old gyro-only path and demand exact equality.
                    w=data['com_velocity'][...,3:]
                    iw=data['link_rotation']@fixture.body_inertias@data['link_rotation'].transpose(-1,-2)
                    gyro=torch.linalg.cross(w,(iw@w[...,None])[...,0])
                    scale=(.005*torch.linalg.vector_norm(w,dim=-1)).clamp_min(1.).reciprocal()
                    old_delta=torch.cat((torch.zeros_like(w),(scale.square()-1.)[...,None]*gyro),-1)
                    dc=(data['jacobian'].transpose(-1,-2)@old_delta[...,None]).sum(1)[...,0]
                    old=torch.linalg.solve(data['mass_matrix'],(-data['coriolis']-dc)[...,None])[...,0]
                    torch.testing.assert_close(qdd,old,rtol=0,atol=0)
                if mode in ('angular','both'):
                    # A gyro-clipped 200rad/s is below every body limit and
                    # would wrongly erase all angular speed damping.
                    self.assertGreater(np.linalg.norm(angular),1000.)
                    wrong=np.linalg.solve(world.sum(0),-fd_gyro)
                    self.assertGreater(np.max(np.abs(expected[3:]-wrong)),100.)

    def test_nonfinite_inputs_and_external_wrenches_still_rejected(self):
        fixture=model(dofs=1)
        data=inputs(dofs=1);data['external_wrench_terms'][0]=1.
        with self.assertRaisesRegex(ValueError,'external wrench'):fixture.predict(**data)
        for field in ('com_velocity','link_rotation','dof_velocity'):
            with self.subTest(field=field):
                data=inputs(dofs=1);data[field].flatten()[0]=float('nan')
                with self.assertRaises(FloatingPointError):fixture.predict(**data)
        with self.assertRaisesRegex(FloatingPointError,'inertia'):
            model(inertia=np.full((3,3),float('nan')))


if __name__=='__main__':
    unittest.main()
