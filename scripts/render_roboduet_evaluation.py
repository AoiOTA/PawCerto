"""CPU saved-state skeleton replay of RoboDuet physical evaluation, no new physics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left',type=Path,required=True,help='Per-seed substeps.pt')
    parser.add_argument('--right',type=Path,required=True)
    parser.add_argument('--case',default='simultaneous_commands')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--speed',type=float,default=.25,help='Playback speed relative to physical time')
    args = parser.parse_args()
    if args.speed <= 0 or args.output.exists():
        parser.error('speed must be positive and output must be new')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pawcerto.artifacts import file_identity
    from pawcerto.robots.go1_arx5 import URDF
    from pawcerto.methods.roboduet.observations import quat_apply
    torch.set_num_threads(1)
    left,right = [torch.load(path,map_location='cpu',weights_only=False) for path in (args.left,args.right)]
    if left['metadata']['seed'] != right['metadata']['seed']:
        raise ValueError('Compare the same fixed evaluation seed')
    if left['metadata']['protocol_sha256'] != right['metadata']['protocol_sha256']:
        raise ValueError('Compare the same frozen protocol')
    sequences = [data['cases'][args.case] for data in (left,right)]
    if not all(sequences):
        raise ValueError('No substep samples')
    model_names = [data['metadata']['export']['next_iteration'] for data in (left,right)]
    all_pos = np.concatenate([row['body_pos'].numpy() for rows in sequences for row in rows])
    center = (all_pos.min(0)+all_pos.max(0))/2
    radius = max(float(np.ptp(all_pos,axis=0).max())/2,.45)
    tree = ET.parse(URDF)
    connections = [(joint.find('parent').get('link'),joint.find('child').get('link')) for joint in tree.findall('joint')]
    figure = plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f4f6f8')
    axes = [figure.add_subplot(1,2,index+1,projection='3d') for index in range(2)]
    duration = max(rows[-1]['time_s'] for rows in sequences)
    fps = 25
    frames = max(2,int(np.ceil(duration/args.speed*fps))+fps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgba','-s','1280x720',
           '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',
           '-crf','19','-movflags','+faststart',str(args.output)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for frame in range(frames):
            t = min(max(frame/fps*args.speed,min(rows[0]["time_s"] for rows in sequences)),duration)
            for ax,data,rows,iteration in zip(axes,(left,right),sequences,model_names):
                ax.clear()
                times = np.array([row['time_s'] for row in rows])
                row = rows[max(0,np.searchsorted(times,t,side='right')-1)]
                positions = row['body_pos'].numpy()
                names = data['metadata']['body_names']
                for parent,child in connections:
                    if parent in names and child in names:
                        points = positions[[names.index(parent),names.index(child)]]
                        color = '#3182bd' if 'zarx' in child else '#415466'
                        ax.plot(*points.T,color=color,lw=3)
                forces = row['contact_forces'].norm(dim=-1).numpy()
                ax.scatter(*positions.T,c=np.log1p(forces),cmap='magma',vmin=0,vmax=7,s=24)
                l,p,y = row['commands_arm'][:3]
                root = row['root_pos']
                forward = quat_apply(row['root_quat_xyzw'][None],torch.tensor([[1.,0.,0.]]))[0]
                yaw = torch.atan2(forward[1],forward[0])
                target = np.array([root[0]+l*p.cos()*(y+yaw).cos(),root[1]+l*p.cos()*(y+yaw).sin(),.38+l*p.sin()])
                ax.scatter(*target,c='#e69f00',s=85,marker='x')
                for axis,value in zip(('x','y','z'),center):
                    getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
                ax.set_zlim(min(-.05,center[2]-radius),center[2]+radius)
                ax.set_box_aspect((1,1,1))
                ax.view_init(elev=24,azim=130)
                status = 'prefix ended' if t >= rows[-1]['time_s'] else 'recording'
                ax.set_title(f'Checkpoint {iteration} | {status}\nheight={float(row["root_height"]):.3f} m | up={float(row["up_dot"]):.3f}',fontsize=12)
                ax.set_xlabel('World x (m)'); ax.set_ylabel('World y (m)'); ax.set_zlabel('World z (m)')
            figure.suptitle(f'RoboDuet official_play Stage 2 | {args.case} | seed {left["metadata"]["seed"]}\nphysical t={t:.3f} s | playback {args.speed:g}x',fontsize=15)
            if frame == 0:
                figure.text(.5,.035,'Recorded link positions and contact magnitude; orange cross = grasp target. CPU replay, no physics rerun.\nEarly prefixes remain visible. Checkpoint 1600 is an early diagnostic, not a Stage 1 causal ablation.',ha='center',fontsize=10)
            figure.subplots_adjust(top=.79,bottom=.12,left=.02,right=.96,wspace=.05)
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            if frame in (0,frames//2,frames-1):
                from PIL import Image
                Image.fromarray(rgba).save(args.output.with_name(args.output.stem+f'-frame{frame}.png'))
            proc.stdin.write(rgba.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        plt.close(figure)
    if code:
        raise RuntimeError(f'ffmpeg failed: {code}')
    metadata = dict(left=file_identity(args.left),right=file_identity(args.right),case=args.case,
                    physical_duration_s=duration,frames=frames,fps=fps,speed=args.speed,
                    renderer='matplotlib Agg CPU',evidence='saved actual body-position skeleton; no new simulation')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
