"""Fixed-base grasp check: weld the trunk in the air, hand a cube to the open beak, shut it,
then nod / swing / point the head down and see whether the cube stays in the beak.

usage: python scripts/grasp_fixed.py [size_m=0.020] [mass_kg=0.010] [open_frac=0.8] [out.mp4]
(position actuators, no BAM: this checks contact geometry, not servo fidelity)"""
import os, sys; os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mujoco, numpy as np, imageio
from microduck_tidy.beak import load_spec_with_beak, BeakServo, PROP_CONTYPE, PROP_CONAFFINITY
from microduck_tidy.sim import SCENE_FLAT as X
OPEN=float(sys.argv[3]) if len(sys.argv)>3 else 0.8
SIZE=float(sys.argv[1]) if len(sys.argv)>1 else 0.020; MASS=float(sys.argv[2]) if len(sys.argv)>2 else 0.010
spec=load_spec_with_beak(X)
spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="trunk_base", objtype=mujoco.mjtObj.mjOBJ_BODY)
cube=spec.worldbody.add_body(name="cube",pos=[0.055,0,0.32])
cube.add_freejoint(name="cube_free")
cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=[SIZE/2]*3,mass=MASS,rgba=[0.9,0.2,0.2,1],condim=4,friction=[1.0,0.02,0.002],
              contype=PROP_CONTYPE,conaffinity=PROP_CONAFFINITY)
m=spec.compile(); m.opt.timestep=0.002; d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"STAND"))
d.ctrl[:]=m.key_ctrl[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"STAND")]
mujoco.mj_forward(m,d)
beak=BeakServo(m,d)
cg=None
gid=lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)
cq=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"cube_free")]
tip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"mouth_tip")
jb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"beak_lower")
act={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i):i for i in range(m.nu)}
base=d.ctrl.copy()
cg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cube')][0]
r=mujoco.Renderer(m,360,480); cam=mujoco.MjvCamera(); cam.lookat[:]=[0.04,0,0.22]; cam.distance=0.3; cam.azimuth=120; cam.elevation=-15
OUT=sys.argv[4] if len(sys.argv)>4 else None
w=imageio.get_writer(OUT,fps=25,codec="libx264",macro_block_size=1) if OUT else None
def cube_in_head():
    R=d.xmat[jb-1].reshape(3,3) if False else d.site_xmat[tip].reshape(3,3)
    return R.T@(d.qpos[cq:cq+3]-d.site_xpos[tip])
T=8.0; phase_log=[]
for k in range(int(T/m.opt.timestep)):
    t=k*m.opt.timestep
    if t<0.6:
        beak.set_open(OPEN if t<0.5 else 0.0)
        if t>0.3:                          # park the cube between the open pads
            lo=d.geom_xpos[gid("beak_lower_pad_collision")]; up=d.geom_xpos[gid("beak_upper_pad_collision")]
            d.qpos[cq:cq+3]=0.5*(lo+up)+np.array([0.004,0,0]); d.qpos[cq+3:cq+7]=d.xquat[jb]
    elif t<2.5: beak.set_open(0.0)        # closed: hold by squeeze only
    else:
        s=t-2.5; c=base.copy()           # shake: head pitch nod, yaw swing, then point beak down
        c[act["head_yaw"]]+=0.6*np.sin(2*np.pi*0.8*s)
        c[act["neck_pitch"]]+=0.4*np.sin(2*np.pi*0.5*s)
        if s>3.5: c[act["head_pitch"]]+=min(1.0,(s-3.5))*1.0
        d.ctrl[:]=c
    if t<0.6 and t>0.3: d.qvel[cq:cq+6]=0
    beak.update(); mujoco.mj_step(m,d)
    if w and k%20==0:
        cam.lookat[:]=d.site_xpos[tip]+[-0.02,0,0]; r.update_scene(d,camera=cam); w.append_data(r.render())
    if k%250==0:
        rel=cube_in_head()
        nc=sum(1 for i in range(d.ncon) if cg in (d.contact[i].geom1,d.contact[i].geom2))
        print(f"ncon_cube={nc} ", end="")
        print(f"t={t:4.1f} mouth={beak.angle:+.3f}rad tau={d.qfrc_applied[beak.vadr]:+.3f} cube_rel_tip(mm)={np.round(rel*1000,1)} cube_z={d.qpos[cq+2]*1000:.0f}mm")
w and w.close()
dist=np.linalg.norm(d.qpos[cq:cq+3]-d.site_xpos[tip])
print(f"RESULT size={SIZE*1000:.0f}mm mass={MASS*1000:.0f}g final dist cube-mouth_tip={dist*1000:.1f}mm -> {'HELD' if dist<0.05 else 'DROPPED'}")
