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
