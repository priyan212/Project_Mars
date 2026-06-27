"""
Mars Rover - Autonomous Exploration Policy + Segmentation Dataset Collector
============================================================================

Goal: drive the rover around the Mars environment with a simple "expert"
policy (random-walk + obstacle avoidance, no human teleop, no RL needed),
and periodically save single-camera frames as (RGB image, semantic
segmentation mask, metadata) triplets for downstream training.

Pipeline:
  1. Attach semantic labels to every relevant prim in the environment
     (terrain -> "ground", rocks -> "rock", craters -> "crater") and to the
     rover itself ("rover") so the segmentation renderer has classes to
     output. Background/sky pixels come out as the "unlabelled" class.
  2. Mount ONE forward-facing camera on the rover body (this is the only
     sensor used for data collection, as requested -- no lidar/depth output
     is saved, though a raycast IS used internally for obstacle avoidance
     since that's the rover's "eyes" for driving, not for the dataset).
  3. Expert policy: drive forward at a random-ish cruise speed, occasionally
     change heading randomly (Mars rovers don't have a fixed "task," so a
     random walk is a reasonable stand-in for "expert roaming"), and
     override that random heading whenever a forward raycast detects an
     obstacle/rock/crater rim within a safety distance -- steer away from it
     and slow down, like a real autonomous-nav obstacle-avoidance behavior.
  4. Every N simulation steps, capture the camera's RGB + semantic
     segmentation buffers via Replicator/Synthetic Data annotators and write
     them to disk as PNG + label-map metadata, building a labelled dataset.

Run this AFTER both the rover-builder and environment-builder scripts have
been run (it expects /World/MarsRover and /World/MarsEnvironment to exist).
Run it in the Script Editor with the simulation PLAYING, or headless via
./python.sh mars_data_collection_policy.py
"""

import os
import json
import random
import math

import numpy as np
import omni.usd
import omni.kit.app
import omni.timeline
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf, Semantics

import omni.replicator.core as rep
from omni.physx import get_physx_scene_query_interface

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROVER_ROOT       = "/World/MarsRover"
ENV_ROOT         = "/World/MarsEnvironment"

OUTPUT_DIR       = "/home/claude/mars_dataset"   # change to your desired path
CAPTURE_EVERY_N_STEPS = 30      # ~0.5s at 60Hz physics step
MAX_FRAMES       = 2000         # stop collecting after this many frames

CAMERA_RESOLUTION = (640, 480)
CAMERA_LOCAL_POS   = (0.85, 0.0, 0.55)   # mounted at the front of the body, up high
CAMERA_LOCAL_ROT_DEG = (0.0, -10.0, 0.0)  # slight downward tilt to see terrain ahead

# Expert policy tuning
CRUISE_SPEED_DEG_S       = 220.0   # wheel angular velocity target ("forward")
TURN_SPEED_DEG_S         = 90.0    # wheel angular velocity target while turning hard
HEADING_CHANGE_INTERVAL  = (3.0, 8.0)   # seconds between random heading changes
HEADING_CHANGE_RANGE_DEG = 35.0    # max random steering offset per change
OBSTACLE_SAFETY_DIST     = 1.8     # meters -- raycast distance that triggers avoidance
RAYCAST_FAN_DEG          = 30.0    # spread of avoidance raycasts left/right of forward
STEER_LIMIT_DEG          = 35.0    # don't exceed the steering joints' real range

random.seed()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
stage = omni.usd.get_context().get_stage()
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Semantic labeling -- required for segmentation ground truth
# ---------------------------------------------------------------------------
def apply_semantic_label(prim, label):
    if not prim or not prim.IsValid():
        return
    if not prim.HasAPI(Semantics.SemanticsAPI):
        sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem_api.CreateSemanticTypeAttr("class")
        sem_api.CreateSemanticDataAttr(label)
    else:
        sem_api = Semantics.SemanticsAPI(prim, "Semantics")
        sem_api.GetSemanticDataAttr().Set(label)


def label_environment():
    terrain_prim = stage.GetPrimAtPath(f"{ENV_ROOT}/terrain")
    apply_semantic_label(terrain_prim, "ground")

    rocks_root = stage.GetPrimAtPath(f"{ENV_ROOT}/rocks")
    if rocks_root and rocks_root.IsValid():
        for child in rocks_root.GetChildren():
            apply_semantic_label(child, "rock")

    craters_root = stage.GetPrimAtPath(f"{ENV_ROOT}/craters")
    if craters_root and craters_root.IsValid():
        for child in craters_root.GetChildren():
            # rim vs floor still both belong to the "crater" class
            apply_semantic_label(child, "crater")

    haze_prim = stage.GetPrimAtPath(f"{ENV_ROOT}/horizon_haze")
    apply_semantic_label(haze_prim, "sky")

    print("Semantic labels applied: ground, rock, crater, sky")


def label_rover():
    rover_prim = stage.GetPrimAtPath(ROVER_ROOT)
    if rover_prim and rover_prim.IsValid():
        # Label the whole rover subtree as "rover" (wheels occasionally
        # visible at the bottom of frame, useful as a distractor class).
        for prim in Usd.PrimRange(rover_prim):
            if prim.IsA(UsdGeom.Gprim):
                apply_semantic_label(prim, "rover")
    print("Semantic label applied: rover")


label_environment()
label_rover()


# ---------------------------------------------------------------------------
# 2. Camera -- single sensor used for ALL saved data
# ---------------------------------------------------------------------------
def create_camera():
    cam_path = f"{ROVER_ROOT}/body/nav_camera"
    camera = UsdGeom.Camera.Define(stage, cam_path)
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

    prim = camera.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_LOCAL_POS))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAMERA_LOCAL_ROT_DEG))

    print(f"Camera created at {cam_path}")
    return cam_path


CAMERA_PATH = create_camera()


# ---------------------------------------------------------------------------
# 3. Replicator render product + writer (RGB + semantic segmentation only)
# ---------------------------------------------------------------------------
render_product = rep.create.render_product(CAMERA_PATH, CAMERA_RESOLUTION)

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True,
    semantic_segmentation=True,
    colorize_semantic_segmentation=True,   # also saves a human-viewable color mask
    semantic_segmentation_mapping=True,    # saves the label<->color/id mapping json
)
writer.attach([render_product])

print(f"Replicator writer attached. Dataset will be written to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# 4. Expert policy -- random-walk exploration + raycast obstacle avoidance
# ---------------------------------------------------------------------------
DRIVE_WHEEL_JOINTS = [
    f"{ROVER_ROOT}/joint_drive_front_L", f"{ROVER_ROOT}/joint_drive_mid_L", f"{ROVER_ROOT}/joint_drive_rear_L",
    f"{ROVER_ROOT}/joint_drive_front_R", f"{ROVER_ROOT}/joint_drive_mid_R", f"{ROVER_ROOT}/joint_drive_rear_R",
]
STEER_JOINTS = {
    "front_L": f"{ROVER_ROOT}/joint_steer_front_L",
    "rear_L":  f"{ROVER_ROOT}/joint_steer_rear_L",
    "front_R": f"{ROVER_ROOT}/joint_steer_front_R",
    "rear_R":  f"{ROVER_ROOT}/joint_steer_rear_R",
}


def set_drive_velocity(joint_path, deg_per_sec):
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        return
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.GetTargetVelocityAttr().Set(deg_per_sec)


def set_steer_angle(joint_path, degrees):
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        return
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.GetTargetPositionAttr().Set(degrees)


def get_rover_pose():
    body_prim = stage.GetPrimAtPath(f"{ROVER_ROOT}/body")
    xform_cache = UsdGeom.XformCache()
    world_xform = xform_cache.GetLocalToWorldTransform(body_prim)
    pos = world_xform.ExtractTranslation()
    rot = world_xform.ExtractRotation()
    # Forward axis assumed +X in body local frame
    forward_world = world_xform.TransformDir(Gf.Vec3d(1, 0, 0))
    forward_world.Normalize()
    return pos, forward_world


def raycast_distance(origin, direction, max_dist=10.0):
    """Returns hit distance, or max_dist if nothing was hit. Used only for
    the rover's internal driving decisions -- NOT saved as training data."""
    hit_info = get_physx_scene_query_interface().raycast_closest(
        carb.Float3(origin[0], origin[1], origin[2]),
        carb.Float3(direction[0], direction[1], direction[2]),
        max_dist,
    )
    if hit_info["hit"]:
        return hit_info["distance"]
    return max_dist


try:
    import carb
except ImportError:
    carb = None


class ExpertPolicy:
    """Random-walk exploration with reactive obstacle avoidance.

    This is intentionally simple (no learning, no path planning) -- it's an
    "expert" only in the sense that it reliably keeps the rover moving and
    out of trouble well enough to harvest diverse, mostly-unobstructed
    camera views for the dataset.
    """

    def __init__(self):
        self.target_steer_deg = 0.0
        self.time_since_heading_change = 0.0
        self.next_heading_change_at = random.uniform(*HEADING_CHANGE_INTERVAL)
        self.avoiding = False

    def update(self, dt):
        self.time_since_heading_change += dt

        # 1) Random heading change, like a rover picking a new bearing
        #    once it's done exploring in the current direction.
        if self.time_since_heading_change >= self.next_heading_change_at:
            self.target_steer_deg = random.uniform(-HEADING_CHANGE_RANGE_DEG,
                                                     HEADING_CHANGE_RANGE_DEG)
            self.time_since_heading_change = 0.0
            self.next_heading_change_at = random.uniform(*HEADING_CHANGE_INTERVAL)

        # 2) Obstacle check via a small fan of forward raycasts.
        pos, forward = get_rover_pose()
        avoid_left = False
        avoid_right = False
        min_dist = OBSTACLE_SAFETY_DIST + 1.0

        if carb is not None:
            for angle_deg in (-RAYCAST_FAN_DEG, 0.0, RAYCAST_FAN_DEG):
                rad = math.radians(angle_deg)
                dx = forward[0] * math.cos(rad) - forward[1] * math.sin(rad)
                dy = forward[0] * math.sin(rad) + forward[1] * math.cos(rad)
                dist = raycast_distance((pos[0], pos[1], pos[2] + 0.3),
                                         (dx, dy, 0.0), max_dist=OBSTACLE_SAFETY_DIST + 1.0)
                min_dist = min(min_dist, dist)
                if dist < OBSTACLE_SAFETY_DIST:
                    if angle_deg < 0:
                        avoid_right = True   # obstacle on the left -> steer right
                    elif angle_deg > 0:
                        avoid_left = True    # obstacle on the right -> steer left
                    else:
                        # dead ahead -- steer whichever side has more room
                        avoid_left = avoid_right = True

        self.avoiding = (min_dist < OBSTACLE_SAFETY_DIST)

        if self.avoiding:
            steer_cmd = STEER_LIMIT_DEG if avoid_left and not avoid_right else \
                        -STEER_LIMIT_DEG if avoid_right and not avoid_left else \
                        random.choice([-STEER_LIMIT_DEG, STEER_LIMIT_DEG])
            drive_speed = TURN_SPEED_DEG_S * 0.5   # slow down while avoiding
        else:
            steer_cmd = max(-STEER_LIMIT_DEG, min(STEER_LIMIT_DEG, self.target_steer_deg))
            drive_speed = CRUISE_SPEED_DEG_S

        # 3) Apply to joints. Front/rear corner wheels steer together
        #    (simple Ackermann-ish approximation, not exact geometry).
        for name, jpath in STEER_JOINTS.items():
            sign = 1.0 if "rear" in name else 1.0
            set_steer_angle(jpath, steer_cmd * sign)

        for jpath in DRIVE_WHEEL_JOINTS:
            set_drive_velocity(jpath, drive_speed)


policy = ExpertPolicy()


# ---------------------------------------------------------------------------
# 5. Main loop -- step physics, run policy, periodically capture a frame
# ---------------------------------------------------------------------------
frame_count = 0
step_count = 0
metadata_log = []


def on_physics_step(dt):
    global frame_count, step_count

    policy.update(dt)
    step_count += 1

    if step_count % CAPTURE_EVERY_N_STEPS == 0 and frame_count < MAX_FRAMES:
        rep.orchestrator.step(rt_subframes=1)  # render + write this frame via Replicator
        pos, forward = get_rover_pose()
        metadata_log.append({
            "frame": frame_count,
            "step": step_count,
            "rover_position": [pos[0], pos[1], pos[2]],
            "rover_forward": [forward[0], forward[1], forward[2]],
            "avoiding_obstacle": policy.avoiding,
        })
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"[mars_dataset] captured frame {frame_count}/{MAX_FRAMES}")

        if frame_count >= MAX_FRAMES:
            print(f"[mars_dataset] reached MAX_FRAMES={MAX_FRAMES}, saving metadata + stopping capture")
            with open(os.path.join(OUTPUT_DIR, "rover_trajectory_metadata.json"), "w") as f:
                json.dump(metadata_log, f, indent=2)


# Hook into the physics step. omni.isaac.core's World is the most common way
# to do this; fall back to a raw physx subscription if World isn't being used.
try:
    from omni.isaac.core import World
    world = World.instance()
    if world is None:
        world = World()
    world.add_physics_callback("mars_expert_policy_step", on_physics_step)
    print("Registered policy+capture callback via omni.isaac.core World.")
except Exception as e:
    import omni.physx
    physx_interface = omni.physx.get_physx_interface()
    _sub = physx_interface.subscribe_physics_step_events(
        lambda evt: on_physics_step(1.0 / 60.0)
    )
    print("Registered policy+capture callback via raw PhysX step subscription.")
    print(f"(World.instance() path failed with: {e})")

print()
print("Expert policy + segmentation data collection is now ARMED.")
print("Press PLAY in Isaac Sim to start the rover roaming and capturing.")
print(f"RGB + segmentation PNGs and label mapping will appear under: {OUTPUT_DIR}")
print(f"Capturing every {CAPTURE_EVERY_N_STEPS} physics steps, up to {MAX_FRAMES} frames.")
