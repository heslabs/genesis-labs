# Genesis labs

Claude> Provide the example for Genesis, a robotic simulators

Here's a comprehensive reference of Genesis examples drawn from the official documentation and GitHub:

### 1. Installation
```
## bash
pip install torch torchvision torchaudio
pip install genesis-world
# Also requires Python ≥ 3.9 and PyTorch
```

### 2. Hello Genesis — minimal scene  
The simplest Genesis script loads a Franka arm, lets it fall under gravity, and runs for 1000 steps — all in under 10 lines: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html)
```
## python
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()
```

### 3. Scene with physics & camera options

You can configure gravity, simulation timestep, and viewer camera in the scene constructor: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html)

```
## python
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -10.0),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=(0, 0, 0),
        euler=(0, 0, 90),  # extrinsic x-y-z, in degrees
    ),
)

scene.build()

for i in range(1000):
    scene.step()
```

### 4. Robot joint control
After building the scene, you retrieve the joint DOF indices and set control gains before driving the robot: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html)

```
python
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

# Map joint names to DOF indices
jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

# Set PD control gains
franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87,-87,-87,-87,-12,-12,-12,-100,-100]),
    np.array([ 87, 87, 87, 87, 12, 12, 12,  100, 100]),
)

# Drive to a target pose
franka.set_dofs_position(
    np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
    dofs_idx,
)

for i in range(1000):
    scene.step()
```

### 5. Inverse kinematics + motion planning + pick-and-place
IK solving and motion planning are single-method calls on the robot entity. After planning a path, you iterate over waypoints and step the scene: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/v0.3.3/user_guide/getting_started/inverse_kinematics_motion_planning.html)

```
python
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30, max_FPS=60,
    ),
    show_viewer=True,
)

plane  = scene.add_entity(gs.morphs.Plane())
cube   = scene.add_entity(gs.morphs.Box(
    size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02),
))
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
scene.build()

motors_dof  = np.arange(7)
fingers_dof = np.arange(7, 9)
end_effector = franka.get_link('hand')

# --- Move to pre-grasp pose via motion planning ---
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
qpos[-2:] = 0.04  # open gripper

path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
for _ in range(100):
    scene.step()

# --- Reach grasp pose ---
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for _ in range(100):
    scene.step()

# --- Close gripper (force control) ---
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
for _ in range(100):
    scene.step()

# --- Lift ---
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for _ in range(200):
    scene.step()
```

### 6. Soft robot with MPM solver
For soft-body simulation, you enable the MPM solver alongside the rigid solver and load a URDF with a soft material: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/soft_robots.html)

```
python
import genesis as gs

gs.init(seed=0, precision='32', logging_level='info')

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=3e-3, substeps=10),
    rigid_options=gs.options.RigidOptions(
        gravity=(0, 0, -9.8),
        enable_collision=True,
        enable_self_collision=False,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(0.0, 0.0, -0.2),
        upper_bound=(1.0, 1.0,  1.0),
        gravity=(0, 0, 0),      # gravity compensation
        enable_CPIC=True,
    ),
    vis_options=gs.options.VisOptions(show_world_frame=True),
    show_viewer=True,
)

scene.add_entity(morph=gs.morphs.Plane())
robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file='urdf/simple/two_link_arm.urdf',
        pos=(0.5, 0.5, 0.3),
    ),
    material=gs.materials.MPM.Muscle(),
)

scene.build()

for i in range(2000):
    scene.step()
```

### 7. Suction / weld constraint (pick with rigid coupling)
You can weld two links together at runtime to simulate suction or magnetic grippers using rigid.add_weld_constraint: 
[[Readthedocs]](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html)

```
python
import numpy as np
import genesis as gs

# ... (scene, franka, cube setup as above) ...

rigid        = scene.sim.rigid_solver
end_effector = franka.get_link('hand')
cube_link    = cube.base_link

# Reach pre-grasp
q_pregrasp = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.13]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_pregrasp[:-2], np.arange(7))
for _ in range(50):
    scene.step()

# Activate suction — weld cube to end-effector
rigid.add_weld_constraint(
    np.array([cube_link.idx],    dtype=gs.np_int),
    np.array([end_effector.idx], dtype=gs.np_int),
)

# Lift
for _ in range(100):
    scene.step()
```

---
### Key concepts summary

| Concept	| API |
|:-|:-|
| Init | gs.init(backend=gs.cpu/gpu) | 
| Scene	| gs.Scene(sim_options=..., viewer_options=...) |
| Load robot (MJCF)	| scene.add_entity(gs.morphs.MJCF(file=...)) |
| Load robot (URDF)	| scene.add_entity(gs.morphs.URDF(file=..., fixed=True)) |
| Primitives| gs.morphs.Plane(), gs.morphs.Box(size=..., pos=...) |
| Build	| scene.build() |
| Step | scene.step() |
| IK | robot.inverse_kinematics(link, pos, quat) |
| Motion plan	| robot.plan_path(qpos_goal, num_waypoints) |
| Position control | robot.control_dofs_position(qpos, dofs_idx) |
| Force control	| robot.control_dofs_force(forces, dofs_idx) |

Full docs at [genesis-world.readthedocs.io](https://genesis-world.readthedocs.io/) and source at [github.com/Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis).



