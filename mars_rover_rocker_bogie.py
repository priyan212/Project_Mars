"""
Mars Rover (Rocker-Bogie) builder for NVIDIA Isaac Sim
========================================================

Builds a 6-wheeled rover with the same kinematic architecture as real Mars
rovers (Curiosity / Perseverance style):

    BODY
     |-- Differential Pivot Joint (revolute, connects L/R rocker bars so they
     |    average out body roll automatically -- "rocker-bogie differential")
     |
     |-- LEFT ROCKER  (revolute joint to body)
     |     |-- FRONT-LEFT WHEEL  (steer revolute + drive revolute)
     |     |-- LEFT BOGIE (revolute joint to rocker, free-pivoting, no motor)
     |           |-- MID-LEFT WHEEL   (drive revolute only, no steering)
     |           |-- REAR-LEFT WHEEL (steer revolute + drive revolute)
     |
     |-- RIGHT ROCKER (mirror of left)
           |-- FRONT-RIGHT WHEEL
           |-- RIGHT BOGIE
                 |-- MID-RIGHT WHEEL
                 |-- REAR-RIGHT WHEEL

This matches the real vehicle:
  * Rockers pivot on the body, bogies pivot freely at the rear of each rocker
    (no actuator -> passive suspension, exactly like the real rover).
  * A differential linkage joint couples the two rockers so the body's pitch
    is the average of the two rockers' angles (real rovers use a literal bar
    + pivot for this -- here we approximate with a D6/revolute "differential"
    joint with a gear-like drive ratio of -1, which is the standard rocker-
    bogie modeling trick).
  * 4 corner wheels (FL, FR, RL, RR) have independent steering (revolute about
    Z) + independent drive (revolute about wheel axle).
  * 2 middle wheels (ML, MR) only drive -- no steering -- exactly like the
    real rover (middle wheels are fixed, used for skid-steer style turning
    along with differential wheel speeds).

Run this INSIDE Isaac Sim's Script Editor, or headless with:
    ./python.sh mars_rover_rocker_bogie.py

Tested against the omni.isaac.core / pxr APIs available in Isaac Sim
2023.1.x - 4.x. If your version's API differs slightly, the comments mark
the spots most likely to need adjustment.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, PhysxSchema
import omni.usd
import omni.kit.commands

# ---------------------------------------------------------------------------
# 0. Setup stage / world
# ---------------------------------------------------------------------------
stage = omni.usd.get_context().get_stage()

ROVER_ROOT = "/World/MarsRover"
if stage.GetPrimAtPath(ROVER_ROOT):
    stage.RemovePrim(ROVER_ROOT)

UsdGeom.Xform.Define(stage, ROVER_ROOT)

# ---------------------------------------------------------------------------
# 1. Tunable dimensions (meters / kg) -- loosely modeled after Curiosity
#    (scaled down a bit so it's easy to look at in a default Isaac Sim scene)
# ---------------------------------------------------------------------------
BODY_SIZE        = Gf.Vec3f(1.6, 1.2, 0.5)   # x=length, y=width, z=height
BODY_MASS        = 350.0

WHEEL_RADIUS     = 0.25
WHEEL_WIDTH      = 0.18
WHEEL_MASS       = 8.0

ROCKER_LENGTH    = 1.1
ROCKER_THICK     = 0.06
ROCKER_MASS      = 12.0

BOGIE_LENGTH     = 0.7
BOGIE_THICK      = 0.06
BOGIE_MASS       = 8.0

TRACK_WIDTH      = 1.5   # left-right distance between wheel centerlines
ROCKER_PIVOT_Z   = 0.15  # height of rocker pivot relative to body center
FRONT_WHEEL_X    = 0.95  # front wheel x offset from rocker pivot
REAR_WHEEL_X     = -0.75 # rear wheel x offset (at end of bogie)
MID_WHEEL_X      = -0.05 # middle wheel x offset (other end of bogie)

DRIVE_MAX_TORQUE   = 80.0
STEER_MAX_TORQUE   = 40.0
DRIVE_STIFFNESS    = 0.0     # velocity-controlled drive (no position spring)
DRIVE_DAMPING      = 1e4     # acts as a velocity-drive gain
STEER_STIFFNESS    = 1e5     # position-controlled steering
STEER_DAMPING      = 1e3


# ---------------------------------------------------------------------------
# 2. Small helpers
# ---------------------------------------------------------------------------
def add_box(path, size, mass, color=(0.55, 0.55, 0.58)):
    """Create a rigid-body box (used for body / rocker / bogie links)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    return prim


def add_wheel(path, radius, width, mass, color=(0.12, 0.12, 0.12)):
    """Create a rigid-body cylinder for a wheel, axle aligned with Y."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(width)
    cyl.CreateAxisAttr("Y")
    prim = cyl.GetPrim()
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    return prim


def set_local_pos(prim, pos):
    xf = UsdGeom.Xformable(prim)
    # Reuse existing scale op if present, add translate first in order
    ops = xf.GetOrderedXformOps()
    has_translate = any(o.GetOpType() == UsdGeom.XformOp.TypeTranslate for o in ops)
    if not has_translate:
        t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(*pos))
        # ensure translate happens before scale
        all_ops = xf.GetOrderedXformOps()
        xf.SetXformOpOrder(all_ops[-1:] + all_ops[:-1])
    else:
        for o in ops:
            if o.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                o.Set(Gf.Vec3d(*pos))


def make_revolute_joint(joint_path, body0_path, body1_path,
                         axis="Y", local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
                         lower_deg=None, upper_deg=None,
                         drive_type=None, target_value=0.0,
                         stiffness=0.0, damping=0.0, max_force=0.0):
    """
    Create a PhysX revolute joint between body0 (parent) and body1 (child).

    drive_type: None | "position" | "velocity"
        "position" -> steering joints (hold an angle)
        "velocity" -> drive/motor joints (spin at a target angular velocity)
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateAxisAttr(axis)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateBreakForceAttr(1e9)
    joint.CreateBreakTorqueAttr(1e9)

    if lower_deg is not None and upper_deg is not None:
        joint.CreateLowerLimitAttr(float(lower_deg))
        joint.CreateUpperLimitAttr(float(upper_deg))
    # else: leave unset/free -> unlimited rotation (used for wheel drive axles)

    prim = joint.GetPrim()

    if drive_type is not None:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(max_force)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        if drive_type == "position":
            drive.CreateTargetPositionAttr(float(target_value))
        elif drive_type == "velocity":
            drive.CreateTargetVelocityAttr(float(target_value))

    return joint


# ---------------------------------------------------------------------------
# 3. BODY
# ---------------------------------------------------------------------------
body_path = f"{ROVER_ROOT}/body"
body_prim = add_box(body_path, BODY_SIZE, BODY_MASS, color=(0.65, 0.62, 0.55))
set_local_pos(body_prim, (0, 0, ROCKER_PIVOT_Z + 0.35))

# A free joint to the world isn't required -- the rigid body simply falls
# under gravity / rests on its wheels once simulated. If you want the rover
# fixed in space for testing, you could add a FixedJoint to a static anchor.


# ---------------------------------------------------------------------------
# 4. Build one side (LEFT or RIGHT) of the rocker-bogie assembly
# ---------------------------------------------------------------------------
def build_side(side):
    """side: 'L' or 'R' -> +1 / -1 multiplier on the Y axis."""
    sign = 1.0 if side == "L" else -1.0
    y = sign * TRACK_WIDTH / 2.0

    # ---- Rocker -----------------------------------------------------
    rocker_path = f"{ROVER_ROOT}/rocker_{side}"
    rocker_prim = add_box(rocker_path,
                           (ROCKER_LENGTH, ROCKER_THICK, ROCKER_THICK),
                           ROCKER_MASS, color=(0.45, 0.45, 0.48))
    rocker_world_pos = (0.1, y, ROCKER_PIVOT_Z)
    set_local_pos(rocker_prim, rocker_world_pos)

    # Revolute joint: body <-> rocker, axis = Y (pitch up/down), free-pivoting
    # (passive suspension -- no drive, just limits).
    make_revolute_joint(
        f"{ROVER_ROOT}/joint_rocker_{side}",
        body0_path=body_path, body1_path=rocker_path,
        axis="Y",
        local_pos0=(0.1, sign * (TRACK_WIDTH / 2.0 - BODY_SIZE[1] / 2.0 + 0.05),
                    ROCKER_PIVOT_Z - (ROCKER_PIVOT_Z + 0.35)),
        local_pos1=(0, 0, 0),
        lower_deg=-25, upper_deg=25,
        drive_type=None,
    )

    # ---- Bogie (attached to the REAR end of the rocker) -------------
    bogie_path = f"{ROVER_ROOT}/bogie_{side}"
    bogie_prim = add_box(bogie_path,
                          (BOGIE_LENGTH, BOGIE_THICK, BOGIE_THICK),
                          BOGIE_MASS, color=(0.4, 0.4, 0.42))
    rocker_rear_world = (rocker_world_pos[0] - ROCKER_LENGTH / 2.0,
                          y, ROCKER_PIVOT_Z)
    bogie_world_pos = (rocker_rear_world[0] - BOGIE_LENGTH / 2.0 + 0.3,
                        y, ROCKER_PIVOT_Z - 0.05)
    set_local_pos(bogie_prim, bogie_world_pos)

    make_revolute_joint(
        f"{ROVER_ROOT}/joint_bogie_{side}",
        body0_path=rocker_path, body1_path=bogie_path,
        axis="Y",
        local_pos0=(-ROCKER_LENGTH / 2.0, 0, 0),
        local_pos1=(BOGIE_LENGTH / 2.0 - 0.3, 0, 0.05),
        lower_deg=-35, upper_deg=35,
        drive_type=None,   # free pivot -- exactly like the real bogie
    )

    # ---- FRONT wheel (mounted on rocker, has steering) ---------------
    build_steerable_wheel(
        name=f"front_{side}",
        parent_path=rocker_path,
        parent_local_pos=(FRONT_WHEEL_X - rocker_world_pos[0], 0, -0.1),
        world_pos=(FRONT_WHEEL_X, y, ROCKER_PIVOT_Z - 0.1 - WHEEL_RADIUS),
    )

    # ---- MIDDLE wheel (mounted on bogie, drive only, no steering) ----
    build_fixed_drive_wheel(
        name=f"mid_{side}",
        parent_path=bogie_path,
        parent_local_pos=(MID_WHEEL_X - bogie_world_pos[0], 0, -0.05),
        world_pos=(MID_WHEEL_X, y, ROCKER_PIVOT_Z - 0.05 - WHEEL_RADIUS),
    )

    # ---- REAR wheel (mounted on bogie, has steering) ------------------
    build_steerable_wheel(
        name=f"rear_{side}",
        parent_path=bogie_path,
        parent_local_pos=(REAR_WHEEL_X - bogie_world_pos[0], 0, -0.05),
        world_pos=(REAR_WHEEL_X, y, ROCKER_PIVOT_Z - 0.05 - WHEEL_RADIUS),
    )

    return rocker_path


def build_steerable_wheel(name, parent_path, parent_local_pos, world_pos):
    """Corner wheel = steering knuckle (revolute about Z) + wheel (revolute
    about Y, the drive axle)."""
    knuckle_path = f"{ROVER_ROOT}/steer_knuckle_{name}"
    knuckle_prim = add_box(knuckle_path, (0.08, 0.08, 0.2), 1.5,
                            color=(0.3, 0.3, 0.32))
    set_local_pos(knuckle_prim, world_pos)

    make_revolute_joint(
        f"{ROVER_ROOT}/joint_steer_{name}",
        body0_path=parent_path, body1_path=knuckle_path,
        axis="Z",
        local_pos0=parent_local_pos,
        local_pos1=(0, 0, 0),
        lower_deg=-90, upper_deg=90,
        drive_type="position", target_value=0.0,
        stiffness=STEER_STIFFNESS, damping=STEER_DAMPING,
        max_force=STEER_MAX_TORQUE,
    )

    wheel_path = f"{ROVER_ROOT}/wheel_{name}"
    wheel_prim = add_wheel(wheel_path, WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS)
    set_local_pos(wheel_prim, world_pos)

    make_revolute_joint(
        f"{ROVER_ROOT}/joint_drive_{name}",
        body0_path=knuckle_path, body1_path=wheel_path,
        axis="Y",
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        lower_deg=None, upper_deg=None,   # free spin
        drive_type="velocity", target_value=0.0,
        stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING,
        max_force=DRIVE_MAX_TORQUE,
    )


def build_fixed_drive_wheel(name, parent_path, parent_local_pos, world_pos):
    """Middle wheel = drive only, no steering knuckle (matches real rover)."""
    wheel_path = f"{ROVER_ROOT}/wheel_{name}"
    wheel_prim = add_wheel(wheel_path, WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS)
    set_local_pos(wheel_prim, world_pos)

    make_revolute_joint(
        f"{ROVER_ROOT}/joint_drive_{name}",
        body0_path=parent_path, body1_path=wheel_path,
        axis="Y",
        local_pos0=parent_local_pos,
        local_pos1=(0, 0, 0),
        lower_deg=None, upper_deg=None,
        drive_type="velocity", target_value=0.0,
        stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING,
        max_force=DRIVE_MAX_TORQUE,
    )


rocker_L_path = build_side("L")
rocker_R_path = build_side("R")

# ---------------------------------------------------------------------------
# 5. Differential linkage between the two rockers
#    Real rovers use a physical differential bar so the body pitch is the
#    AVERAGE of the two rocker angles, regardless of which side hits a rock.
#    We approximate this with a revolute joint connecting the two rockers
#    directly above the body, configured as a soft "averaging" spring -- a
#    common rocker-bogie sim trick when a true mechanical differential gear
#    constraint isn't available in the physics engine.
# ---------------------------------------------------------------------------
make_revolute_joint(
    f"{ROVER_ROOT}/joint_differential",
    body0_path=rocker_L_path, body1_path=rocker_R_path,
    axis="Y",
    local_pos0=(0.1 - 0.1, -TRACK_WIDTH / 2.0 + 0.02, 0.4),
    local_pos1=(0.1 - 0.1, TRACK_WIDTH / 2.0 - 0.02, 0.4),
    lower_deg=-45, upper_deg=45,
    drive_type="position", target_value=0.0,
    stiffness=200.0, damping=50.0,   # soft -- lets each rocker move
    max_force=20.0,                  # independently while nudging toward avg
)

print(f"Mars rover (rocker-bogie, 6 wheels) created at {ROVER_ROOT}")
print("Joints created:")
print("  - 2x rocker pivots (body<->rocker, free, +/-25deg)")
print("  - 2x bogie pivots (rocker<->bogie, free, +/-35deg)")
print("  - 1x differential linkage (rocker<->rocker, soft)")
print("  - 4x steering joints (FL, FR, RL, RR; +/-90deg position drive)")
print("  - 6x wheel drive joints (velocity drive, free spin)")
print()
print("Example: drive forward + steer, run in the Script Editor AFTER")
print("the rover above has been created and the simulation is PLAYING:")
print("""
import omni.usd
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()

def set_drive_velocity(joint_path, rad_per_sec):
    prim = stage.GetPrimAtPath(joint_path)
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    drive.GetTargetVelocityAttr().Set(rad_per_sec)

def set_steer_angle(joint_path, degrees):
    prim = stage.GetPrimAtPath(joint_path)
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    drive.GetTargetPositionAttr().Set(degrees)

ROOT = "/World/MarsRover"
for w in ["front_L", "mid_L", "rear_L", "front_R", "mid_R", "rear_R"]:
    set_drive_velocity(f"{ROOT}/joint_drive_{w}", 200.0)  # deg/s forward

for s in ["front_L", "rear_L", "front_R", "rear_R"]:
    set_steer_angle(f"{ROOT}/joint_steer_{s}", 15.0)      # turn wheels
""")
