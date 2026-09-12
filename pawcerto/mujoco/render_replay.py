"""Software-render a fixed case from two saved physical rollouts; no policy/sim rerun.

Run with xvfb-run and MUJOCO_GL=glfw, LIBGL_ALWAYS_SOFTWARE=1,
GALLIUM_DRIVER=llvmpipe. Images show collision geometry plus true kinematic links.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from OpenGL import GL
from .runtime import Go2Arx5Mujoco
from .asset import ROOT
from .contacts import FEET

WIDTH, VIEW_HEIGHT, HEADER, FOOTER, FPS = 720, 540, 110, 170, 25
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ORANGE, CYAN = (255, 167, 49), (12, 202, 232)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)


def add_geom(scene, kind, pos, size, color, end=None):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array([size, size, size]), np.asarray(pos), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    if end is not None:
        mujoco.mjv_connector(g, kind, size, np.asarray(pos), np.asarray(end))
    scene.ngeom += 1


class Replay:
    def __init__(self, path):
        self.path = Path(path)
        self.raw = np.load(self.path)
        self.meta = json.loads(self.path.with_suffix('.json').read_text())
        source = Path(self.meta['checkpoint'])
        if not source.is_absolute():
            source = ROOT / source
        config = (source.parent if source.is_file() else source) / 'config.json'
        self.sim = Go2Arx5Mujoco(config)
        m = self.sim.model
        m.vis.global_.offwidth, m.vis.global_.offheight = WIDTH, VIEW_HEIGHT
        m.vis.headlight.ambient[:] = .65
        m.vis.headlight.diffuse[:] = .8
        m.vis.headlight.specular[:] = .15
        m.geom_rgba[:] = [.24, .38, .51, 1.]
        m.geom_rgba[m.geom('ground').id] = [.83, .87, .91, 1.]
        for i in range(m.ngeom):
            if m.body(m.geom_bodyid[i]).name.startswith('link'):
                m.geom_rgba[i] = [.64, .69, .75, 1.]
        self.renderer = mujoco.Renderer(m, height=VIEW_HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [-.17, 0., .31]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.65, 122, -29
        self.contact_by_time = {}
        for c in self.meta['nonfoot_contact_records']:
            if c['kind'] == 'external' and c['force_norm_gt_1N']:
                self.contact_by_time.setdefault(round(c['time_s'], 6), set()).update(n for n in c['body_names'] if n != 'world')

    def frame(self, t):
        s, m, d = self.sim, self.sim.model, self.sim.data
        if t == 0:
            s.reset()
            target = self.raw['sampled_target_positions'][0, 0]
            target_r = self.raw['sampled_target_rotations'][0, 0]
            actual, actual_r = d.body('end_effector').xpos.copy(), d.body('end_effector').xmat.reshape(3, 3).copy()
            row = None
            ground = None
        else:
            index = int(np.argmin(abs(self.raw['metrics'][:, 0] - t)))
            row = self.raw['metrics'][index]
            p = self.raw['physical'][index]
            d.qpos[:3] = p[24:27]
            d.qpos[3:7] = Rotation.from_matrix(p[27:36].reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
            d.qpos[s.qadr], d.qvel[s.vadr] = p[36:54], p[54:72]
            mujoco.mj_forward(m, d)  # pose reconstruction only; never mj_step
            actual, actual_r = p[:3], p[3:12].reshape(3, 3)
            target, target_r = p[12:15], p[15:24].reshape(3, 3)
            ground = self.raw['foot_ground_force_z_N'][index]
        for i, name in enumerate(FEET):
            color = [.16, .65, .37, 1.] if ground is not None and ground[i] > 1 else [.22, .29, .34, 1.]
            m.geom_rgba[m.geom_bodyid == m.body(name).id] = color
        self.renderer.update_scene(d, self.camera)
        scene = self.renderer.scene
        for x in np.arange(-1., 1.01, .2):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, [x, -.8, .001], 1., [.61, .67, .73, 1.], [x, .8, .001])
        for y in np.arange(-.8, .81, .2):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, [-1., y, .001], 1., [.61, .67, .73, 1.], [1., y, .001])
        names = ['base_arm_link'] + [f'link{i}' for i in range(1, 7)]
        for a, b in zip(names[:-1], names[1:]):
            if np.linalg.norm(d.body(a).xpos-d.body(b).xpos) > 1e-6:
                add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, d.body(a).xpos, .009, [.33, .45, .56, 1.], d.body(b).xpos)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, target, .019, [1., .61, .1, .38])
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, actual, .009, [0., .78, .93, 1.])
        for pos, rotation, color, length in [(target, target_r, [1., .56, .08, 1.], .065), (actual, actual_r, [0., .73, .88, 1.], .047)]:
            for axis in range(3):
                add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, pos, 3., color, pos + rotation[:, axis] * length)
        image = Image.fromarray(self.renderer.render())
        if row is None:
            info = {'position_mm': float(np.linalg.norm(actual-target)*1000), 'orientation_deg': float(np.degrees(np.arccos(np.clip((np.trace(actual_r.T@target_r)-1)/2,-1,1)))), 'feet': '--', 'up': 1., 'contacts': '--'}
        else:
            info = {'position_mm': row[1]*1000, 'orientation_deg': np.degrees(row[2]), 'feet': int(self.raw['ground_supported_feet'][index]), 'up': row[4], 'contacts': ', '.join(sorted(self.contact_by_time.get(round(float(row[0]),6),set()))) or 'none'}
        return image, info


def render(left_path, right_path, output, left_label, right_label, preview_time=None, group_note=None):
    if os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1' or os.environ.get('MUJOCO_GL') != 'glfw':
        raise RuntimeError('Run under the documented Xvfb software-render command')
    left, right = Replay(left_path), Replay(right_path)
    np.testing.assert_array_equal(left.raw['sampled_target_positions'], right.raw['sampled_target_positions'])
    np.testing.assert_array_equal(left.raw['sampled_target_rotations'], right.raw['sampled_target_rotations'])
    renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
    if 'llvmpipe' not in renderer_name.lower():
        raise RuntimeError(f'Expected CPU llvmpipe renderer, got {renderer_name}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def compose(t):
        canvas = Image.new('RGB', (2*WIDTH, HEADER+VIEW_HEIGHT+FOOTER), '#101d2b')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), 'PawCerto | Go2 + ARX5 | fixed-weight MuJoCo replay', font=font(27, True), fill='white')
        draw.text((24, 52), 'Same preselected sample(16, 0), case 0  |  full 17 s  |  1x time', font=font(20), fill='#bdcdda')
        draw.text((1190, 18), f't = {t:05.2f} s', font=font(27, True), fill='white')
        for offset, obj, title in [(0, left, left_label), (WIDTH, right, right_label)]:
            picture, info = obj.frame(t)
            canvas.paste(picture, (offset, HEADER))
            draw.text((offset+24, 82), title, font=font(19, True), fill='white')
            y = HEADER+VIEW_HEIGHT+10
            draw.text((offset+24, y), f'EE error: {info["position_mm"]:6.1f} mm  |  {info["orientation_deg"]:5.1f} deg', font=font(22, True), fill='white')
            draw.text((offset+24, y+34), f'Ground feet: {info["feet"]}/4   Root up: {info["up"]:.3f}', font=font(19), fill='#c5d4df')
            draw.text((offset+24, y+64), f'Other ground contact (>1 N): {info["contacts"]}', font=font(16), fill='#c5d4df')
        y=HEADER+VIEW_HEIGHT+112
        draw.ellipse((26,y+2,42,y+18),fill=ORANGE)
        draw.text((50,y),'Target EE',font=font(17),fill=ORANGE)
        draw.ellipse((170,y+2,186,y+18),fill=CYAN)
        draw.text((194,y),'Actual EE',font=font(17),fill=CYAN)
        draw.text((344,y),'Collision geometry + true kinematic links; saved-state replay, not a new rollout.',font=font(16),fill='#a8bfce')
        draw.text((26,y+27),group_note or 'Training-data case; no holdout claim. Contacts are sampled at 20 ms policy endpoints.',font=font(16),fill='#a8bfce')
        return canvas
    if preview_time is not None:
        compose(preview_time).save(output)
    else:
        # Include both t=0 and the endpoint t=17.0: 426 frames at25fps=17.04s container.
        command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{2*WIDTH}x{HEADER+VIEW_HEIGHT+FOOTER}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for t in np.arange(426)/FPS:
                process.stdin.write(compose(float(t)).tobytes())
        finally:
            process.stdin.close()
            code=process.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed with exit code{code}')
        output.with_suffix('.json').write_text(json.dumps({'left_source':str(left_path),'right_source':str(right_path), 'simulation_seconds':17.,'container_seconds':17.04,'frames':426,'fps':FPS,'renderer':renderer_name,'replay_only':True,'group_note':group_note,'case_selection':'preselected sample(16,0) case0; no visual selection'},indent=2))
    left.renderer.close()
    right.renderer.close()
    print(json.dumps({'output':str(output),'renderer':renderer_name,'replay_only':True}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,default=ROOT/'outputs/mujoco/contact-audit/scratch_500/case_00.npz')
    p.add_argument('--right',type=Path,default=ROOT/'outputs/mujoco/contact-audit/official_ours/case_00.npz')
    p.add_argument('--left-label',default='Scratch training: iteration 500')
    p.add_argument('--right-label',default='Official released ours')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/videos/scratch500_vs_official_case0.mp4')
    p.add_argument('--preview-time',type=float)
    p.add_argument('--group-note')
    a=p.parse_args();render(a.left,a.right,a.output,a.left_label,a.right_label,a.preview_time,a.group_note)
