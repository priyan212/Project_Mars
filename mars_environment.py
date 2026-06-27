"""
Mars Environment Builder for NVIDIA Isaac Sim
================================================

Procedurally builds a Mars-like terrain for rover simulation:

  * Reddish-orange undulating terrain (ground plane + randomized height bumps
    via a subdivided mesh, so it isn't perfectly flat -- the rover suspension
    actually has something to do)
  * Scattered rocks/boulders of varying size, all with PhysX colliders so the
    rover physically interacts with them
  * A few craters (shallow sunken rings) for visual + navigational interest
  * Mars-accurate lighting: dim orange-tinted "sun" (distant light, lower
    intensity than Earth sun since Mars gets ~43% of Earth's solar flux) +
    a butterscotch/dusty-pink dome light for the sky/ambient fill
  * A keep-out zone at the origin so rocks/craters don't spawn on top of
    wherever you place the rover

Run this INSIDE Isaac Sim's Script Editor, or headless with:
    ./python.sh mars_environment.py

Run the rover script first (or after, order doesn't matter) -- this script
only touches /World/MarsEnvironment and leaves /World/MarsRover untouched.
"""

import random
import math

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Sdf, Gf, Vt
import omni.usd

stage = omni.usd.get_context().get_stage()

ENV_ROOT = "/World/MarsEnvironment"
if stage.GetPrimAtPath(ENV_ROOT):
    stage.RemovePrim(ENV_ROOT)
UsdGeom.Xform.Define(stage, ENV_ROOT)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
SEED               = 42
TERRAIN_SIZE        = 60.0     # meters, square terrain
TERRAIN_RES         = 80       # grid subdivisions per side (resolution)
TERRAIN_BUMP_HEIGHT = 0.35     # max random height of terrain noise
TERRAIN_BUMP_SCALE  = 6.0      # "wavelength" of terrain undulation

KEEP_OUT_RADIUS     = 4.0      # no rocks/craters spawn within this of origin

NUM_SMALL_ROCKS     = 140
NUM_MEDIUM_ROCKS    = 40
NUM_LARGE_BOULDERS  = 10
NUM_CRATERS         = 6

SMALL_ROCK_RADIUS   = (0.06, 0.18)
MEDIUM_ROCK_RADIUS  = (0.2, 0.45)
LARGE_BOULDER_RADIUS = (0.6, 1.4)

random.seed(SEED)

# Mars surface colors -- ranges so each rock/terrain patch varies a bit
MARS_COLOR_PALETTE = [
    (0.62, 0.32, 0.18),   # rusty red-orange
    (0.55, 0.28, 0.16),   # darker red-brown
    (0.68, 0.40, 0.24),   # lighter ochre
    (0.45, 0.22, 0.13),   # dark basalt-ish brown
    (0.58, 0.34, 0.22),   # dusty clay
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_mars_color(jitter=0.05):
    base = random.choice(MARS_COLOR_PALETTE)
    return Gf.Vec3f(*[max(0.0, min(1.0, c + random.uniform(-jitter, jitter))) for c in base])


def value_noise(x, y, scale):
    """Cheap deterministic pseudo-noise (sum of sines) -- no numpy/perlin
    dependency required, good enough for terrain variation."""
    n = 0.0
    n += math.sin(x / scale * 1.0 + 0.3) * math.cos(y / scale * 1.3 + 1.7)
    n += 0.5 * math.sin(x / scale * 2.3 + 1.1) * math.cos(y / scale * 2.7 + 0.4)
    n += 0.25 * math.sin(x / scale * 4.1 + 2.2) * math.cos(y / scale * 3.9 + 2.9)
    return n / 1.75   # normalize roughly to [-1, 1]


def distance_from_origin(x, y):
    return math.sqrt(x * x + y * y)


def random_point_outside_keepout(half_size, keepout):
    """Uniform random (x, y) within the terrain bounds, rejecting points
    too close to the origin (where the rover spawns)."""
    for _ in range(50):
        x = random.uniform(-half_size, half_size)
        y = random.uniform(-half_size, half_size)
        if distance_from_origin(x, y) > keepout:
            return x, y
    return half_size * 0.8, half_size * 0.8  # fallback


def set_color(prim, color):
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([color])


def make_static_collider(prim):
    """Rocks/terrain are static (no RigidBodyAPI) but still need collision."""
    UsdPhysics.CollisionAPI.Apply(prim)


# ---------------------------------------------------------------------------
# 1. Terrain: subdivided mesh plane with noise-based height + collider
# ---------------------------------------------------------------------------
def build_terrain():
    terrain_path = f"{ENV_ROOT}/terrain"
    mesh = UsdGeom.Mesh.Define(stage, terrain_path)

    half = TERRAIN_SIZE / 2.0
    res = TERRAIN_RES
    step = TERRAIN_SIZE / res

    points = []
    height_lookup = {}
    for j in range(res + 1):
        for i in range(res + 1):
            x = -half + i * step
            y = -half + j * step
            h = value_noise(x, y, TERRAIN_BUMP_SCALE) * TERRAIN_BUMP_HEIGHT
            # flatten near the keep-out zone so the rover starts on flat ground
            d = distance_from_origin(x, y)
            if d < KEEP_OUT_RADIUS * 1.5:
                blend = max(0.0, min(1.0, (d - KEEP_OUT_RADIUS) / (KEEP_OUT_RADIUS * 0.5)))
                h *= blend
            points.append(Gf.Vec3f(x, y, h))
            height_lookup[(i, j)] = h

    face_vertex_counts = []
    face_vertex_indices = []
    for j in range(res):
        for i in range(res):
            v0 = j * (res + 1) + i
            v1 = j * (res + 1) + (i + 1)
            v2 = (j + 1) * (res + 1) + (i + 1)
            v3 = (j + 1) * (res + 1) + i
            face_vertex_counts.append(4)
            face_vertex_indices.extend([v0, v1, v2, v3])

    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
    mesh.CreateSubdivisionSchemeAttr("none")

    # Vertex colors: slightly vary terrain shade with height + noise
    colors = []
    for j in range(res + 1):
        for i in range(res + 1):
            base = random_mars_color(jitter=0.03)
            colors.append(base)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray(colors))
    mesh.SetDisplayColorPrimvar(UsdGeom.Tokens.vertex)

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcol = UsdPhysics.MeshCollisionAPI.Apply(prim)
    meshcol.CreateApproximationAttr("none")   # use exact triangle mesh collider

    print(f"Terrain built: {TERRAIN_SIZE}m x {TERRAIN_SIZE}m, {res}x{res} grid")
    return terrain_path


# ---------------------------------------------------------------------------
# 2. Rocks -- spheres with randomized non-uniform scale so they don't look
#    like perfect balls, randomized rotation, scattered across the terrain.
# ---------------------------------------------------------------------------
def sample_terrain_height(x, y):
    return value_noise(x, y, TERRAIN_BUMP_SCALE) * TERRAIN_BUMP_HEIGHT


def build_rock(name, position_xy, radius_range, irregular=True):
    rpath = f"{ENV_ROOT}/rocks/{name}"
    sphere = UsdGeom.Sphere.Define(stage, rpath)
    r = random.uniform(*radius_range)
    sphere.CreateRadiusAttr(1.0)

    prim = sphere.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    x, y = position_xy
    z = sample_terrain_height(x, y) + r * 0.4   # partially embed in ground

    translate = xf.AddTranslateOp()
    translate.Set(Gf.Vec3d(x, y, z))

    rot = xf.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)))

    # Irregular scale -> looks like a weathered rock, not a perfect sphere
    if irregular:
        sx = r * random.uniform(0.7, 1.3)
        sy = r * random.uniform(0.7, 1.3)
        sz = r * random.uniform(0.5, 0.9)   # flatter on average, like real rocks
    else:
        sx = sy = sz = r

    scale = xf.AddScaleOp()
    scale.Set(Gf.Vec3f(sx, sy, sz))

    set_color(prim, random_mars_color())

    UsdPhysics.CollisionAPI.Apply(prim)
    collision_api = UsdPhysics.CollisionAPI(prim)
    collision_api.CreateCollisionEnabledAttr(True)
    # Keep rocks static (no RigidBodyAPI) -- they're terrain obstacles, not
    # objects the rover should be able to push around. If you'd rather have
    # the rover able to shove small rocks, apply UsdPhysics.RigidBodyAPI and
    # UsdPhysics.MassAPI to "small" rocks specifically.

    return rpath


def build_all_rocks():
    half = TERRAIN_SIZE / 2.0 - 1.0

    for idx in range(NUM_SMALL_ROCKS):
        pos = random_point_outside_keepout(half, KEEP_OUT_RADIUS)
        build_rock(f"small_{idx:03d}", pos, SMALL_ROCK_RADIUS)

    for idx in range(NUM_MEDIUM_ROCKS):
        pos = random_point_outside_keepout(half, KEEP_OUT_RADIUS)
        build_rock(f"medium_{idx:03d}", pos, MEDIUM_ROCK_RADIUS)

    for idx in range(NUM_LARGE_BOULDERS):
        pos = random_point_outside_keepout(half, KEEP_OUT_RADIUS * 1.2)
        build_rock(f"boulder_{idx:03d}", pos, LARGE_BOULDER_RADIUS, irregular=True)

    total = NUM_SMALL_ROCKS + NUM_MEDIUM_ROCKS + NUM_LARGE_BOULDERS
    print(f"Rocks built: {total} total "
          f"({NUM_SMALL_ROCKS} small, {NUM_MEDIUM_ROCKS} medium, "
          f"{NUM_LARGE_BOULDERS} boulders)")


# ---------------------------------------------------------------------------
# 3. Craters -- shallow torus-like rings made from a flattened, inverted
#    cone/cylinder cutout look using a simple scaled cylinder for the rim.
#    (Kept simple/robust: a slightly raised ring + dark depressed disk.)
# ---------------------------------------------------------------------------
def build_crater(name, position_xy, outer_radius):
    x, y = position_xy
    z = sample_terrain_height(x, y)

    rim_path = f"{ENV_ROOT}/craters/{name}_rim"
    rim = UsdGeom.Cylinder.Define(stage, rim_path)
    rim.CreateRadiusAttr(outer_radius)
    rim.CreateHeightAttr(0.12)
    rim.CreateAxisAttr("Z")
    rim_prim = rim.GetPrim()
    xf = UsdGeom.Xformable(rim_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z + 0.06))
    set_color(rim_prim, random_mars_color(jitter=0.02))
    UsdPhysics.CollisionAPI.Apply(rim_prim)

    floor_path = f"{ENV_ROOT}/craters/{name}_floor"
    floor = UsdGeom.Cylinder.Define(stage, floor_path)
    floor.CreateRadiusAttr(outer_radius * 0.75)
    floor.CreateHeightAttr(0.06)
    floor.CreateAxisAttr("Z")
    floor_prim = floor.GetPrim()
    xf2 = UsdGeom.Xformable(floor_prim)
    xf2.ClearXformOpOrder()
    xf2.AddTranslateOp().Set(Gf.Vec3d(x, y, z - 0.18))
    base = random.choice(MARS_COLOR_PALETTE)
    darker = Gf.Vec3f(base[0] * 0.6, base[1] * 0.6, base[2] * 0.6)
    set_color(floor_prim, darker)
    UsdPhysics.CollisionAPI.Apply(floor_prim)


def build_all_craters():
    half = TERRAIN_SIZE / 2.0 - 2.0
    for idx in range(NUM_CRATERS):
        pos = random_point_outside_keepout(half, KEEP_OUT_RADIUS * 1.5)
        radius = random.uniform(1.0, 2.5)
        build_crater(f"crater_{idx:02d}", pos, radius)
    print(f"Craters built: {NUM_CRATERS}")


# ---------------------------------------------------------------------------
# 4. Lighting -- Mars gets ~43% of Earth's solar irradiance and the sky is
#    a dusty butterscotch color due to suspended dust scattering light.
# ---------------------------------------------------------------------------
def build_lighting():
    light_root = f"{ENV_ROOT}/lighting"
    UsdGeom.Xform.Define(stage, light_root)

    # "Sun" -- distant light, dimmer & warmer than Earth's, low angle for
    # long martian shadows.
    sun_path = f"{light_root}/sun"
    sun = UsdLux.DistantLight.Define(stage, sun_path)
    sun.CreateIntensityAttr(1800.0)          # dimmer than a typical Earth-sun (~3000-5000)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.55))   # warm orange sunlight
    sun.CreateAngleAttr(0.6)                  # slightly soft-edged shadows (dust)
    sun_prim = sun.GetPrim()
    xf = UsdGeom.Xformable(sun_prim)
    xf.ClearXformOpOrder()
    # Low sun angle (~25 deg elevation) coming from the "east"
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-65.0, 25.0, 0.0))

    # Dome light -- the butterscotch/dusty-pink Martian sky provides
    # ambient fill light from all directions.
    dome_path = f"{light_root}/sky_dome"
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.55, 0.42))   # dusty pink-orange sky
    dome.CreateSpecularAttr(0.4)

    print("Lighting built: warm distant 'sun' + dusty dome sky")


# ---------------------------------------------------------------------------
# 5. Background sky color via render settings (best-effort -- some Isaac Sim
#    versions expose this differently; dome light color above is the part
#    that's guaranteed to work everywhere).
# ---------------------------------------------------------------------------
def build_far_horizon_haze():
    """Large, very pale, very distant flattened dome to suggest atmospheric
    dust haze near the horizon -- purely visual, no collider."""
    haze_path = f"{ENV_ROOT}/horizon_haze"
    haze = UsdGeom.Sphere.Define(stage, haze_path)
    haze.CreateRadiusAttr(1.0)
    prim = haze.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddScaleOp().Set(Gf.Vec3f(400.0, 400.0, 220.0))
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -150.0))
    set_color(prim, Gf.Vec3f(0.72, 0.5, 0.38))
    # No collider, no physics -- visual backdrop only.
    print("Horizon haze backdrop added")


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------
build_terrain()
build_all_rocks()
build_all_craters()
build_lighting()
build_far_horizon_haze()

print()
print(f"Mars environment created at {ENV_ROOT}")
print(f"Keep-out zone of radius {KEEP_OUT_RADIUS} m around the origin is clear "
      f"for rover spawn.")
print("Tip: run the rover-builder script too (creates /World/MarsRover) -- "
      "the two are independent and can be run in either order.")
