"""Tensor-only XYZW quaternion algebra used by the original Gym task.

No simulator import. Euler outputs preserve the original [0, 2*pi) convention.
"""
import torch
import math


def normalize(x, eps=1e-9):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def quat_mul(q, r):
    xyz = q[...,3:4]*r[...,:3] + r[...,3:4]*q[...,:3] + torch.cross(q[...,:3],r[...,:3],dim=-1)
    w = q[...,3:4]*r[...,3:4] - (q[...,:3]*r[...,:3]).sum(-1,keepdim=True)
    return torch.cat((xyz,w),-1)


def quat_apply(q, v):
    shape = v.shape
    q = q.reshape(-1,4);v = v.reshape(-1,3)
    cross = 2*torch.cross(q[:,:3],v,dim=-1)
    return (v + q[:,3:4]*cross + torch.cross(q[:,:3],cross,dim=-1)).reshape(shape)


def quat_rotate_inverse(q,v):
    return quat_apply(quat_conjugate(q),v)


def quat_from_euler_xyz(roll,pitch,yaw):
    cr,sr=torch.cos(roll/2),torch.sin(roll/2)
    cp,sp=torch.cos(pitch/2),torch.sin(pitch/2)
    cy,sy=torch.cos(yaw/2),torch.sin(yaw/2)
    return torch.stack((sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy),-1)


def get_euler_xyz(q):
    x,y,z,w=q.unbind(-1)
    roll=torch.atan2(2*(w*x+y*z), w*w-x*x-y*y+z*z)
    pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1))
    yaw=torch.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z)
    return roll%(2*math.pi),pitch%(2*math.pi),yaw%(2*math.pi)


def torch_rand_float(lower,upper,shape,device):
    return (upper-lower)*torch.rand(*shape,device=device)+lower


def to_torch(value,device='cpu',dtype=torch.float,requires_grad=False):
    return torch.tensor(value,dtype=dtype,device=device,requires_grad=requires_grad)


def get_axis_params(value,axis_idx,x_value=0.,dtype=float,n_dims=3):
    out=[0.]*n_dims;out[axis_idx]=value
    if axis_idx != 0:out[0]=x_value
    return out
