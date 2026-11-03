"""
Rod between two springs: interactive simulation.

State y = [r (3), v (3), u (3), omega (3)]
  r     : center of mass
  u     : unit vector along the rod (RA = r + L/2 u, RB = r - L/2 u)
  omega : angular velocity

Equations of motion
  m r'' = FA + FB - gamma r'
  I omega' = s * torque - gamma_r * omega,   u' = omega x u
  s = -1 : the rod rotates opposite to the torque,  s = +1 : normal physics

Controls
  t            flip the sign s of the torque
  d            randomize the rod direction
  e            small random perturbation of position and direction
  m            randomize the rod position
  mouse drag   black squares: move the spring anchors
               rod body: move the rod
               rod ends: rotate the rod about its center
  Ctrl+S       save the figure
Click the plot window first so it has keyboard focus.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Keep only Ctrl+S for saving (must run before the figure is created)
plt.rcParams["keymap.save"] = ["ctrl+s"]

# ---------------- Parameters ----------------
L = 1.0
k = 1.0
l0 = 0.0                       # spring rest length (0 -> ideal zero-length spring)
m = 1.0
I = m * L**2 / 12.0            # thin rod about its center
gamma = 1.0                    # translational friction
gamma_r = 1.0                  # rotational friction
s = -1.0                       # torque sign, flipped with 't'

RA_spring = np.array([5.0, 0.0, 0.0])
RB_spring = np.array([-5.0, 0.0, 0.0])

dt = 0.005                     # integration step
substeps = 4                   # integration steps per animation frame
print_every = 10               # print the diagnostic every N frames

# ---------------- Random helpers ----------------
RANDOM_3D = False              # False: stay in the xy plane, True: full 3D randomization
rng = np.random.default_rng()

def random_direction():
    if RANDOM_3D:
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)
    th = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(th), np.sin(th), 0.0])

# Small perturbation ('e')
POS_PERTURB = 0.05             # max displacement of the center of mass
DIR_PERTURB = 0.10             # approx. max rotation of the rod (radians)

def perturb(y):
    """Kick position and direction slightly; velocities are left unchanged."""
    if RANDOM_3D:
        dr = rng.uniform(-POS_PERTURB, POS_PERTURB, size=3)
        du = rng.normal(size=3)
    else:
        dr = np.append(rng.uniform(-POS_PERTURB, POS_PERTURB, size=2), 0.0)
        du = np.append(rng.normal(size=2), 0.0)
    y[0:3] += dr
    u = y[6:9] + DIR_PERTURB * rng.uniform() * du / np.linalg.norm(du)
    y[6:9] = u / np.linalg.norm(u)
    return dr

# Random position ('m')
POS_RANGE_X = 3.0
POS_RANGE_Y = 1.0
POS_RANGE_Z = 1.0              # only used if RANDOM_3D = True

def random_position():
    z = rng.uniform(-POS_RANGE_Z, POS_RANGE_Z) if RANDOM_3D else 0.0
    return np.array([rng.uniform(-POS_RANGE_X, POS_RANGE_X),
                     rng.uniform(-POS_RANGE_Y, POS_RANGE_Y),
                     z])

# ---------------- Physics ----------------
def spring_force(R_end, R_anchor):
    d = R_end - R_anchor
    dist = np.linalg.norm(d)
    if dist < 1e-12:
        return np.zeros(3)
    return -k * (dist - l0) * d / dist

def forces(y):
    r, u = y[0:3], y[6:9]
    u = u / np.linalg.norm(u)
    RA = r + 0.5 * L * u
    RB = r - 0.5 * L * u
    return spring_force(RA, RA_spring), spring_force(RB, RB_spring), RA, RB

def rhs(y):
    r, v, u, w = y[0:3], y[3:6], y[6:9], y[9:12]
    u = u / np.linalg.norm(u)
    FA, FB, RA, RB = forces(y)
    torque = np.cross(RA - r, FA) + np.cross(RB - r, FB)
    return np.concatenate([v,
                           (FA + FB - gamma * v) / m,
                           np.cross(w, u),
                           (s * torque - gamma_r * w) / I])

def rk4_step(y, h):
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    y[6:9] /= np.linalg.norm(y[6:9])        # keep u a unit vector
    return y

def tension(y):
    """T > 0: tension, T < 0: compression.
       T = 1/2 u.(FA - FB)   (static spring contribution only)"""
    u = y[6:9] / np.linalg.norm(y[6:9])
    FA, FB, _, _ = forces(y)
    return 0.5 * np.dot(u, FA - FB)

def state_label(T, tol=1e-9):
    if T > tol:
        return "TENSION"
    if T < -tol:
        return "COMPRESSION"
    return "ZERO"

# ---------------- Initial state ----------------
theta0 = np.deg2rad(30)
y = np.concatenate([np.zeros(3), np.zeros(3),
                    [np.cos(theta0), np.sin(theta0), 0.0], np.zeros(3)])
t = 0.0
trail_x, trail_y = [], []

# ---------------- Visualization ----------------
def coil(p0, p1, n_coils=12, amp=0.25):
    """Zig-zag spring drawn between points p0 and p1 (xy projection)."""
    p0, p1 = p0[:2], p1[:2]
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return np.array([p0[0]]), np.array([p0[1]])
    e = d / length
    nrm = np.array([-e[1], e[0]])
    n = 2 * n_coils + 1
    s_ = np.linspace(0, 1, n)
    pts = p0[None, :] + s_[:, None] * d[None, :]
    off = amp * np.where(np.arange(n) % 2 == 0, 1, -1) * np.sin(np.pi * s_)**0.5
    off[0] = off[-1] = 0
    pts += off[:, None] * nrm[None, :]
    return pts[:, 0], pts[:, 1]

fig, ax = plt.subplots(figsize=(9, 4))
ax.set_aspect("equal")
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 2)
ax.set_title("Rod between two springs  ('t': flip torque, 'd': random direction, "
             "'e': perturb, 'm': random position, drag anchors / rod / rod ends)",
             fontsize=8)
anchorA, = ax.plot([RA_spring[0]], [RA_spring[1]], "s", color="k", ms=12)
anchorB, = ax.plot([RB_spring[0]], [RB_spring[1]], "s", color="k", ms=12)
rod_line, = ax.plot([], [], "-", lw=5)
spA, = ax.plot([], [], "b-", lw=1.5)
spB, = ax.plot([], [], "g-", lw=1.5)
com, = ax.plot([], [], "ko", ms=5)
trail, = ax.plot([], [], "k:", lw=0.8)
txt = ax.text(0.02, 0.80, "", transform=ax.transAxes, family="monospace")

# ---------------- Key handling ----------------
def on_key(event):
    global s, y
    if event.key == "t":
        s = -s
        print(f"\n*** s flipped: s = {s:+.0f} ***\n")
    elif event.key == "d":
        y[6:9] = random_direction()
        y[9:12] = 0.0                  # reset spin so it doesn't carry over
        ang = np.degrees(np.arctan2(y[7], y[6]))
        print(f"\n*** rod direction randomized: angle = {ang:.1f} deg ***\n")
    elif event.key == "e":
        dr = perturb(y)
        ang = np.degrees(np.arctan2(y[7], y[6]))
        print(f"\n*** perturbed: dr = ({dr[0]:+.3f}, {dr[1]:+.3f}, {dr[2]:+.3f}), "
              f"new angle = {ang:.1f} deg ***\n")
    elif event.key == "m":
        y[0:3] = random_position()
        y[3:6] = 0.0                   # reset translational velocity
        trail_x.clear()                # avoid a line jumping across the plot
        trail_y.clear()
        print(f"\n*** position randomized: r = ({y[0]:+.3f}, {y[1]:+.3f}, {y[2]:+.3f}) ***\n")

fig.canvas.mpl_connect("key_press_event", on_key)

# ---------------- Mouse dragging: anchors and rod ----------------
PICK_RADIUS = 0.4              # grab distance for anchors and rod ends (data units)
ROD_PICK_WIDTH = 0.2           # grab distance to the rod body (data units)
dragging = None                # None, "A", "B" (anchors), "rod_move", "rod_rotate"
grab_offset = np.zeros(2)      # rod_move: r - mouse at grab time
grab_end = 1.0                 # rod_rotate: +1 if end A grabbed, -1 if end B

def toolbar_active():
    """True if the zoom/pan tool is on, so we don't fight with it."""
    tb = getattr(fig.canvas, "toolbar", None)
    mode = getattr(tb, "mode", "")
    mode = getattr(mode, "value", mode)
    return str(mode) != ""

def point_segment_distance(p, a, b):
    ab = b - a
    tt = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + tt * ab))

def on_press(event):
    global dragging, grab_offset, grab_end
    if event.inaxes is not ax or event.button != 1 or toolbar_active():
        return
    p = np.array([event.xdata, event.ydata])
    # priority 1: spring anchors
    dA = np.linalg.norm(p - RA_spring[:2])
    dB = np.linalg.norm(p - RB_spring[:2])
    if min(dA, dB) <= PICK_RADIUS:
        dragging = "A" if dA <= dB else "B"
        (anchorA if dragging == "A" else anchorB).set_color("tab:orange")
        return
    # priority 2: rod ends -> rotate; priority 3: rod body -> translate
    _, _, RA, RB = forces(y)
    eA = np.linalg.norm(p - RA[:2])
    eB = np.linalg.norm(p - RB[:2])
    if min(eA, eB) <= PICK_RADIUS:
        dragging = "rod_rotate"
        grab_end = 1.0 if eA <= eB else -1.0
        com.set_color("tab:orange")
    elif point_segment_distance(p, RB[:2], RA[:2]) <= ROD_PICK_WIDTH:
        dragging = "rod_move"
        grab_offset = y[0:2] - p
        com.set_color("tab:orange")

def on_motion(event):
    if dragging is None or event.inaxes is not ax:
        return
    p = np.array([event.xdata, event.ydata])
    if dragging in ("A", "B"):
        target = RA_spring if dragging == "A" else RB_spring
        target[0], target[1] = p                       # in place; z stays 0
    elif dragging == "rod_move":
        y[0:2] = p + grab_offset                       # translate, orientation unchanged
    elif dragging == "rod_rotate":
        d = p - y[0:2]                                 # rotate about the center of mass
        n = np.linalg.norm(d)
        if n > 1e-6:
            y[6:9] = np.array([grab_end * d[0] / n, grab_end * d[1] / n, 0.0])

def on_release(event):
    global dragging
    if dragging is None:
        return
    if dragging in ("A", "B"):
        (anchorA if dragging == "A" else anchorB).set_color("k")
        print(f"\n*** anchors: RAspring = ({RA_spring[0]:+.2f}, {RA_spring[1]:+.2f}), "
              f"RBspring = ({RB_spring[0]:+.2f}, {RB_spring[1]:+.2f}) ***\n")
    else:
        com.set_color("k")
        trail_x.clear()
        trail_y.clear()
        ang = np.degrees(np.arctan2(y[7], y[6]))
        print(f"\n*** rod released: r = ({y[0]:+.3f}, {y[1]:+.3f}), angle = {ang:.1f} deg ***\n")
    dragging = None

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

# ---------------- Animation ----------------
frame_count = 0

def update(_):
    global y, t, frame_count
    if dragging in ("rod_move", "rod_rotate"):
        y[3:6] = 0.0                   # rod is held by the mouse: no motion,
        y[9:12] = 0.0                  # velocities zeroed so there is no kick on release
    else:
        for _ in range(substeps):
            y = rk4_step(y, dt)
            t += dt
    frame_count += 1

    r, u = y[0:3], y[6:9]
    _, _, RA, RB = forces(y)
    T = tension(y)
    label = state_label(T)

    rod_line.set_data([RB[0], RA[0]], [RB[1], RA[1]])
    rod_line.set_color("tab:blue" if T > 0 else "tab:red")   # blue = tension, red = compression
    spA.set_data(*coil(RA_spring, RA))
    spB.set_data(*coil(RB_spring, RB))
    anchorA.set_data([RA_spring[0]], [RA_spring[1]])
    anchorB.set_data([RB_spring[0]], [RB_spring[1]])
    com.set_data([r[0]], [r[1]])
    trail_x.append(r[0])
    trail_y.append(r[1])
    trail.set_data(trail_x, trail_y)

    ang = np.degrees(np.arctan2(u[1], u[0]))
    txt.set_text(f"t = {t:6.2f}   s = {s:+.0f}\n"
                 f"angle = {ang:7.1f} deg\n"
                 f"T = {T:+.4f}  ({label})")

    if frame_count % print_every == 0:
        print(f"t={t:7.2f}  s={s:+.0f}  angle={ang:8.2f} deg  "
              f"T={T:+.5f}  -> {label}")
    return rod_line, spA, spB, anchorA, anchorB, com, trail, txt

ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.show()
