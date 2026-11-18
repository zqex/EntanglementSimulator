"""
Interactive 3D viewer + simulation of polymers in a spherical volume, modeled as piecewise
linear curves with kinks.

Model
-----
Polymer m has kinks R[m,0] ... R[m,N[m]]. R[m,0] and R[m,N[m]] are the two chain ends (beads on the
surface of the sphere); R[m,1] ... R[m,N[m]-1] are the interior kinks. Segment i runs from
R[m,i-1] to R[m,i] and has a contour length L[m,i] > 0 (initially the straight length of the
polymer). Every segment is a spring:
    F[m,i] = (k_spring / L[m,i]) * (R[m,i] - R[m,i-1])       (force of segment i)
    E[m,i] = 0.5 * (k_spring / L[m,i]) * |R[m,i] - R[m,i-1]|^2
Dynamics
  * Kinks follow damped Langevin dynamics:  m dv/dt = F - gamma v + sqrt(2 gamma kT) xi(t), with
    F the total force on the kink: the net force of the two strands on either side of it,
    F[m,i+1] - F[m,i] (F[m,i] = k_spring / L[m,i] * (R_i - R_{i-1})), plus the repulsive forces
    below. Every substep: a kick by F, the exact friction + thermal noise update (Ornstein-
    Uhlenbeck) and a drift step. The step per substep is limited to MAX_KINK_STEP.
  * Excluded volume by forces (WCA): the interaction is the Weeks-Chandler-Andersen potential
    U(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps for r < MIN_DISTANCE, zero beyond, with
    MIN_DISTANCE = 2^(1/6) sigma (the WCA cutoff). (1) Every line segment of another polymer
    repels a kink that lies within MIN_DISTANCE of it (its closest point inside the segment) with
    the WCA force, directed perpendicular to the line; the reaction is shared over the two end
    points of the segment (weights 1 - s and s). (2) Any two kinks, on the same polymer or on different polymers, that are
    closer than MIN_DISTANCE repel each other with the WCA force along the line between them (on
    top of the capture attraction between neighbouring kinks of one polymer). Chain
    ends on the surface are not moved by these forces.
  * Slip: contour length flows along the chain between neighbouring segments of the same polymer
    so that the spring energy decreases (L_i proportional to d_i in the minimum), while the
    total contour length of every polymer is exactly conserved.
  * Creation: when two segments of different polymers approach closer than MIN_DISTANCE (closest
    points in the interior of both segments) a kink is inserted in each of them at the closest
    points; the two new kinks are placed exactly MIN_DISTANCE apart (symmetrically about the
    contact point). The contour length of the divided segment is split in proportion to where
    the kink sits on it.
  * Capture: neighbouring kinks of the same polymer (consecutive interior kinks) that are closer
    than DISTANCE_CAPTURE attract each other with F(r) = -30 r - 40 r^3 / 3 - 160 r^5 / 27
    (negative = attractive), along the line joining them; zero beyond DISTANCE_CAPTURE.
  * The removal criterion is only checked every CHECK_KINK_REMOVAL_EACH_STEP steps (creation: every step).
  * Spacing: no kink is created on a line if another kink of the same polymer lies within a
    contour length CONTOUR_KINK_MIN of that point, or within a spatial distance SPATIAL_KINK_MIN
    (= 1) of the new kink's position. Close segment pairs that fail this on both
    lines are skipped (other close pairs are still handled); if it fails on one line only, a
    single kink is created on the other line, MIN_DISTANCE from the closest point of the first.
  * Destruction: a kink that is further than MIN_DISTANCE (+ DESTROY_TOL) from all segments of
    other polymers, that does not interact with kinks of other polymers, and that has at most one
    other kink of its own polymer within a contour length INTRA_CONTOUR_DIST, is removed, provided the single straight segment that replaces its
    two segments is not closer than MIN_DISTANCE to any line segment of another polymer (closest
    points anywhere on both segments, no margin). Otherwise the kink stays. The contour lengths of the two segments add up.
Segments of a polymer also push its own kinks (WCA), but only if the contour distance from the kink to
the nearer end of the segment exceeds SELF_KINK_DISTANCE; kink creation only considers pairs of segments
of different polymers.

Mouse
- Left-drag on empty space  -> rotate the view
- Right-drag / mouse wheel  -> zoom
- Middle-drag (or Shift+left-drag) -> pan
- Left-drag on a chain-end sphere -> move it along the surface of the volume
- Hold Shift while dragging a chain end and move the cursor out past the circular edge of the
  projected sphere -> the bead crosses to the other side (front <-> back) of the volume

Keyboard: see help_text() below (press h, H or ? in the window, or run with --help).

Data flow
    simulation  --positions / network-->  viewer      (timer, every ~30 ms)
    viewer (user input)  --callbacks-->  simulation   (see SphereDragger.__init__)

Bead ordering: molecule by molecule, each as [start, end]; kinks are not beads and are reported
separately by `network()`. Deleting a molecule shifts the indices of all later beads.

Install:  pip install pyvista numpy
Movies:   pip install imageio imageio-ffmpeg
"""
import argparse
import colorsys
import locale
import os
import platform
import re
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import vtk

ACTIVE_COLOR = (0.0, 0.0, 0.0)  # highlight color while a sphere is being dragged

CONTOUR_STEP = 2.0              # contour length added/removed per press of Right/Left arrow
INITIAL_ZOOM = 1.6              # camera zoom when the window opens
START_GAP = 1.25                # initial distance between the two starting polymers, in units of MIN_DISTANCE
WCA_SIGMA = 1.0                 # length scale of the WCA interaction
WCA_EPSILON = 1.0               # strength of the WCA interaction
WCA_FORCE_CAP = 200.0           # largest WCA force (avoids blow-up when things overlap deeply)
MIN_DISTANCE = 2.0 ** (1.0 / 6.0) * WCA_SIGMA   # = the WCA cutoff: kinks are created when two lines come
                                # closer than this; the repulsive force is zero beyond it
DISTANCE_CAPTURE = 1.5             # neighbouring kinks of a polymer closer than this attract each other
CHECK_KINK_REMOVAL_EACH_STEP = 10   # the kink removal criterion is checked once per this many steps
SELF_KINK_DISTANCE = 10.0         # a segment pushes a kink of its own polymer only if the contour distance exceeds this
INTRA_CONTOUR_DIST = 1.3        # a kink is not removed if more than one other kink of its polymer is within this contour length
SPATIAL_KINK_MIN = 1.0           # ... nor within this spatial distance of one
CONTOUR_KINK_MIN = 1.0          # no new kink within this contour length of an existing kink of the same polymer
KINK_MARGIN = 0.1               # contact points must lie at least this far from both ends of a segment
CREATE_TOL = 0.01               # a kink pair is created below MIN_DISTANCE - CREATE_TOL
DESTROY_TOL = 0.01              # a kink is destroyed above MIN_DISTANCE + DESTROY_TOL
MAX_KINK_STEP = 0.25 * MIN_DISTANCE     # largest displacement of a kink in one substep
KINK_MASS = 1.0                 # mass of a kink (Langevin dynamics)
KINK_FRICTION = 5.0             # friction coefficient gamma of a kink
TEMPERATURE = 0.0               # kT of the thermal noise on the kinks (0: no noise, deterministic)
TIME_STEP = 0.02                # Langevin time step per substep
MAX_NEW_KINKS_PER_STEP = 20     # safety limit for the number of kink pairs created in one step
THIN_LINE_RADIUS = 0.05         # tube radius of all lines after pressing t
TENSION_THICKNESS = 0.01        # tube radius of a segment per unit of spring tension
MAX_LINE_RADIUS = 1.0           # thickest tube drawn
MIN_VOLUME_RADIUS = 5.0
VOLUME_RADIUS_STEP = 1.0

# Shift + digit produces a layout-specific symbol; which key name (and character) the window
# system reports for it depends on the keyboard layout. The layout is detected from the
# environment (see detect_keyboard_layout) and selects one of these tables.
SHIFTED_DIGIT_KEYSYMS = {
    "us": ("exclam", "at", "numbersign", "dollar", "percent",
           "asciicircum", "ampersand", "asterisk", "parenleft"),
    "uk": ("exclam", "quotedbl", "sterling", "dollar", "percent",
           "asciicircum", "ampersand", "asterisk", "parenleft"),
    "de": ("exclam", "quotedbl", "section", "dollar", "percent",
           "ampersand", "slash", "parenleft", "parenright"),
    "nordic": ("exclam", "quotedbl", "numbersign", "currency", "percent",
               "ampersand", "slash", "parenleft", "parenright"),
}
SHIFTED_DIGIT_CHARS = {
    "us": "!@#$%^&*(",
    "uk": '!"\u00a3$%^&*(',
    "de": '!"\u00a7$%&/()',
    "nordic": '!"#\u00a4%&/()',
}

# words that identify a layout: language / country codes and the names used by the platforms
_LAYOUT_WORDS = {
    "us": "us", "usa": "us", "en": "us", "abc": "us", "u": "us", "nordic": "nordic",
    "gb": "uk", "uk": "uk", "british": "uk",
    "de": "de", "at": "de", "german": "de",
    "dk": "nordic", "da": "nordic", "danish": "nordic", "se": "nordic", "sv": "nordic",
    "swedish": "nordic", "no": "nordic", "nb": "nordic", "nn": "nordic", "norwegian": "nordic",
    "fi": "nordic", "finnish": "nordic", "is": "nordic", "icelandic": "nordic",
}
_keyboard_layout = None


def _layout_from_text(text):
    """Layout name for a description such as 'dk', 'da_DK', 'com.apple.keylayout.Danish'
    (the most specific word, the last one, wins: 'en_GB' is British), or None."""
    for word in reversed(re.findall(r"[a-z]+", str(text).lower())):
        if word in _LAYOUT_WORDS:
            return _LAYOUT_WORDS[word]
    return None


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=3).stdout


def _layout_windows():
    import ctypes
    langid = ctypes.windll.user32.GetKeyboardLayout(0) & 0xFFFF
    if langid == 0x0809:
        return "gb"
    return {0x06: "da", 0x1D: "sv", 0x14: "nb", 0x0B: "fi", 0x07: "de"}.get(langid & 0x3FF, "us")


def _layout_macos():
    return _run(["defaults", "read", "com.apple.HIToolbox",
                 "AppleCurrentKeyboardLayoutInputSourceID"])


def _layout_unix():
    """X11 / Wayland / console: the XKB layout (first of a comma separated list)."""
    for command in (["setxkbmap", "-query"], ["localectl", "status"]):
        try:
            match = re.search(r"(?:layout|Layout):\s*([\w-]+)", _run(command))
        except (OSError, subprocess.SubprocessError):
            continue
        if match:
            return match.group(1)
    if os.environ.get("XKB_DEFAULT_LAYOUT"):
        return os.environ["XKB_DEFAULT_LAYOUT"].split(",")[0]
    try:
        with open("/etc/default/keyboard") as f:
            match = re.search(r'XKBLAYOUT="?([\w-]+)', f.read())
        return match.group(1) if match else None
    except OSError:
        return None


def _layout_locale():
    return locale.getlocale()[0] or os.environ.get("LANG", "")


def detect_keyboard_layout():
    """Keyboard layout ('us', 'uk', 'de' or 'nordic') taken from the environment: the variable
    LINEPPA_KEYBOARD_LAYOUT if it is set, else what the operating system reports, else the
    language of the locale; 'us' if nothing is recognised."""
    probes = []
    if os.environ.get("LINEPPA_KEYBOARD_LAYOUT"):
        probes.append(lambda: os.environ["LINEPPA_KEYBOARD_LAYOUT"])
    system = platform.system()
    probes.append(_layout_windows if system == "Windows"
                  else _layout_macos if system == "Darwin" else _layout_unix)
    probes.append(_layout_locale)
    for probe in probes:
        try:
            layout = _layout_from_text(probe() or "")
        except Exception:        # a failing probe must never stop the program
            continue
        if layout:
            return layout
    return "us"


def get_keyboard_layout():
    global _keyboard_layout
    if _keyboard_layout is None:
        _keyboard_layout = detect_keyboard_layout()
    return _keyboard_layout


def set_keyboard_layout(name):
    """Use layout `name` ('us', 'uk', 'de', 'nordic') or, for 'auto'/None, detect it."""
    global _keyboard_layout
    _keyboard_layout = None if name in (None, "auto") else name


def help_text():
    """Summary of the program, the mouse actions and the key shortcuts (--help, h, H, ?)."""
    return f"""\
LinePPA - polymers in a spherical volume

Every polymer is a piecewise linear curve between two chain ends (spheres) on the surface of the
sphere; initially a straight line. Each segment is a spring with spring constant
spring_constant / contour_length (the thickness of a line shows its tension).
  * When two polymers come closer than MIN_DISTANCE ({MIN_DISTANCE:g}) a kink is inserted in both
    (small spheres in the color of the polymer), the two kinks exactly MIN_DISTANCE apart.
  * Kinks move by damped Langevin dynamics under the net force of their two strands plus WCA
    repulsion: every segment of another polymer pushes a kink within MIN_DISTANCE (= 2^(1/6), the
    WCA cutoff) away perpendicular to the line, and any two kinks (same or different polymer) repel each other.
  * A kink that is further than MIN_DISTANCE from all other polymers is destroyed: the two
    segments become one straight segment (their contour lengths add up).
  * Contour length slips along the chain, lowering the spring energy, while the total contour
    length of a polymer stays constant.

Mouse
  left-drag on empty space        rotate the view
  right-drag, mouse wheel         zoom
  middle-drag, Shift+left-drag    pan
  left-drag on a chain-end sphere move it along the surface of the sphere
  ... and Shift, cursor out past the edge of the sphere: the end moves to the other side

Keys while a chain end is held
  d                               delete the polymer the end belongs to
  Right / Left arrow              add / remove contour length of the segment at the end
Keys
  a                               add a polymer (a straight line between two free spots)
  r / R                           shrink / grow the sphere by 1 (minimum radius 5)
  t / T                           thin lines of equal radius /
                                  thickness shows the tension (kink spheres always shown)
  Shift+1..9 (or Ctrl+1..9)       store the positions of all chain ends in slot 1..9
  1..9                            move all chain ends to the positions stored in the slot
  i                               print the selected options and parameters
  k                               print the minimal distances: two kinks of one polymer, two line
                                  segments of different polymers, two non-adjacent segments of one polymer
  m / M                           start / stop recording a movie (movie00000.avi, ...)
  p                               save a PNG of the view (img00000.png, ...)
  s                               save all positions (conf00000.dump, ...)
  h, H, ?                         print this text
Command line
  --help                          print this text
  --keyboard-layout LAYOUT        us, uk, de, nordic or auto (default: detected from the
                                  environment; the variable LINEPPA_KEYBOARD_LAYOUT overrides)
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="cylppa", description=help_text(), add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="show this help and exit")
    parser.add_argument("--keyboard-layout", default="auto",
                        choices=["auto", *sorted(SHIFTED_DIGIT_KEYSYMS)],
                        help="keyboard layout used for Shift+digit (default: detect)")
    return parser.parse_args(argv)


def color_for_id(moleculeid):
    """Deterministic, well-separated RGB color (0..1) for an unsigned integer id."""
    hue = (int(moleculeid) * 0.618033988749895) % 1.0  # golden-ratio hue spacing
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


class SphereDragger:
    """Lets the user drag chain ends along a spherical surface, add/delete molecules,
    resize the volume, store/recall chain-end configurations, and save images / movies /
    bead positions; lets a simulation push new positions for all beads. Keeps the
    molecule lines, the kinks and the volume shell in sync."""

    IMAGE_PATTERN = re.compile(r"^img(\d{5})\.png$")
    CONF_PATTERN = re.compile(r"^conf(\d{5})\.dump$")
    MOVIE_PATTERN = re.compile(r"^movie(\d{5})\.avi$")
    # characters whose built-in VTK meaning (reset camera, surface mode, pick, ...) is suppressed
    OWN_CHARS = frozenset("aAdDhHiIkKpPsSmMrRtT?123456789")

    def __init__(self, plotter, template, actors, positions, moleculeids,
                 center, radius, sphere_radius, recall_speed=0.5, movie_fps=30, rng=None,
                 kink_radius=0.5, get_network=None, get_diagnostics=None,
                 on_drag=None, on_release=None,
                 on_add_molecule=None, on_delete_molecule=None,
                 on_radius_change=None, on_ends_driven=None, on_ends_released=None,
                 on_contour_change=None, get_options=None,
                 initial_zoom=INITIAL_ZOOM):
        self.plotter = plotter
        self.renderer = plotter.renderer
        self.template = template        # sphere mesh used for newly created beads
        self.actors = list(actors)
        self.positions = np.array(positions, dtype=float).reshape(-1, 3)   # (n, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)             # radius of the spherical volume
        self.sphere_radius = float(sphere_radius)   # radius of one bead
        self.kink_radius = float(kink_radius)   # radius of the kink spheres
        self.recall_speed = float(recall_speed)     # rad/s along great circles
        self.movie_fps = movie_fps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.next_moleculeid = int(self.moleculeids.max()) + 1 if len(self.moleculeids) else 0
        self.image_counter = self._first_free_number(self.IMAGE_PATTERN)
        self.conf_counter = self._first_free_number(self.CONF_PATTERN)
        self.movie_counter = self._first_free_number(self.MOVIE_PATTERN)

        # Pulled from the simulation every tick: (kink positions (K, 3), [moleculeid] per kink,
        # [(moleculeid, [node, ...], [contour_length, ...], [tension, ...])]).
        # Node k < n is bead k, node n + r is kink r. None = no kinks.
        self.get_network = get_network
        self.get_options = get_options  # () -> [(label, value), ...] settings of the simulation (key i)
        self.get_diagnostics = get_diagnostics  # () -> [(label, text), ...] minimal distances (key k)

        # Callbacks into the simulation:
        self.on_drag = on_drag          # on_drag(index, xyz)                    while dragging
        self.on_release = on_release    # on_release(index, xyz)                 when let go
        self.on_add_molecule = on_add_molecule        # (first_index, positions (2, 3), moleculeid)
        self.on_delete_molecule = on_delete_molecule  # (first_index, count)  old indices
        self.on_radius_change = on_radius_change      # (new_volume_radius)
        self.on_ends_driven = on_ends_driven          # (indices, positions) chain ends moved by a recall
        self.on_ends_released = on_ends_released      # (indices)            recall finished/cancelled
        self.on_contour_change = on_contour_change    # (index, delta)  add/remove contour length at chain end

        self.interactor = plotter.iren.interactor
        self.picker = vtk.vtkPropPicker()
        self.dragged = None             # index of the chain end being dragged
        self.uniform_lines = False      # True after t: all lines thin and equally thick
        self._pending_zoom = initial_zoom   # applied on the first timer tick, after the window opened
        self._side = 1                  # +1: dragged bead on the front (camera) side, -1: back
        self._flip_armed = True         # a Shift + "touch the edge" gesture may flip the side
        self.stored = {}                # slot (1..9) -> {(moleculeid, is_start): unit vector}
        self._anim = None               # running recall animation
        self._driven = set()            # indices of chain ends moved by the animation
        self.recording = False
        self.movie_path = None
        self._in_tick = False           # re-entrancy guard for tick()
        self.bond_mesh = None
        self.lines = []
        self._seg_nodes = None          # (S, 2) node numbers of the end points of every segment
        self._seg_tension = []          # spring tension of every segment (drawn as its thickness)
        # Segments are drawn as tubes whose radius follows the spring tension. Every segment
        # has its own two points (so the radius is constant along it).
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetNumberOfSides(12)
        self.tube.SetCapping(True)
        self.tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.tube.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                         "radius")
        bond_mapper = vtk.vtkPolyDataMapper()
        bond_mapper.SetInputConnection(self.tube.GetOutputPort())
        bond_mapper.SetScalarModeToUsePointFieldData()
        bond_mapper.SelectColorArray("rgb")
        bond_mapper.SetColorModeToDirectScalars()
        self.bond_actor = vtk.vtkActor()
        self.bond_actor.SetMapper(bond_mapper)
        self.bond_actor.SetVisibility(False)
        plotter.add_actor(self.bond_actor, reset_camera=False, name="bonds", pickable=False)
        self.kink_template = pv.Sphere(radius=self.kink_radius,
                                       theta_resolution=16, phi_resolution=16)
        self.kink_actors = []
        self.kink_positions = np.zeros((0, 3))
        self.kink_molecules = []        # moleculeid of every kink
        self._line_signature = None

        # Translucent shell showing the volume (unit sphere, scaled to the current radius).
        # Not pickable, so it never blocks grabbing.
        self.shell = plotter.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64),
            color="lightgray", opacity=0.12, smooth_shading=True)
        self.shell.SetPickable(False)
        self.shell.SetPosition(*self.center)
        self.shell.SetScale(self.radius, self.radius, self.radius)
        self._pull_network(force=True)

        # Make sure PyVista's own key shortcuts for the keys we use don't also fire.
        for key in ("Up", "Down", "Left", "Right", "a", "d", "p", "s", "S", "m", "M",
                    "r", "R", "t", "T", "i", "I", "k", "K", "h", "H", "question", *"123456789"):
            try:
                plotter.clear_events_for_key(key)
            except Exception:
                pass

        # Priority -1 => our handlers run AFTER the camera style's handlers.
        # If we grabbed a bead, we then cancel the style's rotation.
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_press, -1.0)
        self.interactor.AddObserver("MouseMoveEvent", self._on_move, -1.0)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._on_release, -1.0)
        self.interactor.AddObserver("KeyPressEvent", self._on_key, -1.0)
        # VTK's interactor style gives some plain characters a built-in meaning on the
        # CharEvent that follows a key press. A higher-priority observer neutralises the
        # characters we use before the style sees them.
        self.interactor.AddObserver("CharEvent", self._on_char, 10.0)

    # ------------------------------------------------------------ timer tick
    def tick(self, step_simulation):
        """Call every ~30 ms: advance a running recall animation, take a simulation
        step (`step_simulation()` returns the (n, 3) positions), draw, record a frame.
        A tick that arrives while one is still running is skipped."""
        if self._in_tick:
            return
        self._in_tick = True
        try:
            if self._pending_zoom is not None:
                self.renderer.GetActiveCamera().Zoom(self._pending_zoom)
                self._pending_zoom = None
            self.animate()
            self.set_positions(step_simulation())
            self.write_movie_frame()
        finally:
            self._in_tick = False

    def _print_options(self):
        """Key i: print the viewer settings, then the settings and parameters of the simulation."""
        n_chains = len(self.positions) // 2
        rows = [("line display", "thin lines of equal radius (t)" if self.uniform_lines
                 else "thickness shows the tension (T)"),
                ("keyboard layout", get_keyboard_layout()),
                ("volume radius", f"{self.radius:g}"),
                ("polymers / kinks", f"{n_chains} / {len(self.kink_positions)}")]
        if self.get_options is not None:
            rows += list(self.get_options())
        width = max(len(label) for label, _ in rows)
        print("Selected options:")
        for label, value in rows:
            print(f"  {label:<{width}} : {value}")

    def _print_diagnostics(self):
        """Key k: print the minimal distances between kinks and between line segments."""
        if self.get_diagnostics is None:
            print("No simulation attached.")
            return
        rows = list(self.get_diagnostics())
        width = max(len(label) for label, _ in rows)
        print("Minimal distances:")
        for label, text in rows:
            print(f"  {label:<{width}} : {text}")

    # ------------------------------------------------------- simulation -> view
    def set_positions(self, positions):
        """Update all bead positions (e.g. from the simulation). Beads held by the user or moved
        by a recall animation keep their viewer-controlled positions."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if len(positions) != len(self.actors):
            raise ValueError(f"Simulation returned {len(positions)} beads, "
                             f"viewer has {len(self.actors)}.")
        for i, actor in enumerate(self.actors):
            if i != self.dragged and i not in self._driven:
                self.positions[i] = positions[i]
                actor.SetPosition(*positions[i])
        self._pull_network()
        self.plotter.render()

    def get_positions(self):
        return self.positions.copy()

    # ------------------------------------------------------ molecule lines, kinks
    def _all_points(self):
        return np.vstack([self.positions, self.kink_positions])

    def _segment_radii(self):
        """Tube radius of every segment (order of self._seg_nodes) from its spring tension."""
        if self.uniform_lines:
            return np.full(len(self._seg_tension), THIN_LINE_RADIUS)
        return np.minimum(TENSION_THICKNESS * np.array(self._seg_tension, dtype=float),
                          MAX_LINE_RADIUS)

    def _refresh_bonds(self):
        """Cheap update when only positions (and with them the tensions) changed."""
        if self.bond_mesh is None:
            return
        points = self._all_points()
        if self._seg_nodes.max() < len(points):
            self.bond_mesh.points = points[self._seg_nodes].reshape(-1, 3)
            self.bond_mesh.point_data["radius"] = np.repeat(self._segment_radii(), 2)
            self.bond_mesh.GetPointData().SetActiveScalars("radius")

    def _rebuild_bonds(self):
        """Needed when the number of beads, kinks or lines changed."""
        nodes, rgb = [], []
        for mol, idx, _contours, _tensions in self.lines:
            color = (np.array(color_for_id(mol)) * 255).astype(np.uint8)
            for a, b in zip(idx[:-1], idx[1:]):
                nodes.append((a, b))
                rgb += [color] * 2
        if not nodes:
            self.bond_mesh, self._seg_nodes = None, None
            self.bond_actor.SetVisibility(False)
            return
        self._seg_nodes = np.array(nodes, dtype=int)
        k = np.arange(len(nodes))
        cells = np.column_stack([np.full(len(nodes), 2), 2 * k, 2 * k + 1]).ravel()
        mesh = pv.PolyData(self._all_points()[self._seg_nodes].reshape(-1, 3),
                           lines=cells.astype(np.int64))
        mesh.point_data["radius"] = np.repeat(self._segment_radii(), 2)
        mesh.point_data["rgb"] = np.array(rgb, dtype=np.uint8)
        mesh.GetPointData().SetActiveScalars("radius")
        self.bond_mesh = mesh
        self.tube.SetInputData(mesh)
        self.bond_actor.SetVisibility(True)

    def _default_network(self):
        """No simulation-provided network: every molecule is a straight line, no kinks."""
        lines = {}
        for i, m in enumerate(self.moleculeids.tolist()):
            lines.setdefault(m, []).append(i)
        return np.zeros((0, 3)), [], [(m, idx, [self.radius] * (len(idx) - 1),
                                       [30.0] * (len(idx) - 1))
                                      for m, idx in lines.items()]

    def _pull_network(self, force=False):
        """Fetch kinks and lines from the simulation and bring the drawing up to date."""
        kpos, kmols, lines = (self.get_network() if self.get_network is not None
                              else self._default_network())
        self.kink_positions = np.asarray(kpos, dtype=float).reshape(-1, 3)
        kmols = [int(m) for m in kmols]
        self.lines = [(int(m), [int(k) for k in idx], list(cs), [float(t) for t in ts])
                      for m, idx, cs, ts in lines]
        self._seg_tension = [t for _m, _idx, _cs, ts in self.lines for t in ts]
        signature = tuple((m, tuple(idx)) for m, idx, _cs, _ts in self.lines)
        changed = force or signature != self._line_signature or self.bond_mesh is None
        self._line_signature = signature

        # a kink sphere is never thinner than the segments meeting at it
        n = len(self.positions)
        radii = np.full(len(kmols), self.kink_radius)
        seg_radius = self._segment_radii()
        k = 0
        for _m, idx, _cs, _ts in self.lines:
            for a, b in zip(idx[:-1], idx[1:]):
                for node in (a, b):
                    if node >= n and node - n < len(radii):
                        radii[node - n] = max(radii[node - n], 1.5 * seg_radius[k])
                k += 1

        if kmols != self.kink_molecules or len(self.kink_actors) != len(kmols):
            self._sync_kink_actors(kmols)
        for actor, p, r in zip(self.kink_actors, self.kink_positions, radii):
            actor.SetPosition(*p)
            scale = r / self.kink_radius
            actor.SetScale(scale, scale, scale)
            actor.SetVisibility(True)       # kink spheres are always shown
        if changed:
            self._rebuild_bonds()
        else:
            self._refresh_bonds()

    def _sync_kink_actors(self, kmols):
        """(Re)create the kink spheres, each in the color of its molecule."""
        for actor in self.kink_actors:
            self.plotter.remove_actor(actor, reset_camera=False)
        self.kink_actors = []
        for m in kmols:
            actor = self.plotter.add_mesh(self.kink_template, color=color_for_id(m),
                                          smooth_shading=True, reset_camera=False)
            actor.SetPickable(False)
            self.kink_actors.append(actor)
        self.kink_molecules = kmols

    # ------------------------------------------------------------------ helpers
    def _ray(self, x, y):
        """World-space ray (near point, far point) under display pixel (x, y)."""
        pts = []
        for z in (0.0, 1.0):
            self.renderer.SetDisplayPoint(x, y, z)
            self.renderer.DisplayToWorld()
            w = np.array(self.renderer.GetWorldPoint())
            pts.append(w[:3] / w[3])
        return pts

    def _is_front(self, p):
        """True if surface point p faces the camera (is on the visible hemisphere)."""
        cam = self.renderer.GetActiveCamera()
        v = np.asarray(p, dtype=float) - self.center
        if cam.GetParallelProjection():
            return float(np.dot(v, -np.array(cam.GetDirectionOfProjection()))) >= 0.0
        to_cam = np.array(cam.GetPosition()) - self.center
        return float(np.dot(v, to_cam)) >= self.radius ** 2   # polar-plane visibility test

    def _surface_point(self, x, y, current):
        """Point on the spherical surface under the mouse cursor, on the side of the
        sphere (front or back) the bead is currently on.

        If the cursor is outside the circular outline of the projected sphere, the bead
        sits on that outline. Holding Shift there flips the bead to the other side
        (once per excursion); it then follows the cursor on the far side of the sphere."""
        near, far = self._ray(x, y)
        d = far - near
        d /= np.linalg.norm(d)
        oc = near - self.center
        b = np.dot(oc, d)
        disc = b * b - (np.dot(oc, oc) - self.radius ** 2)
        if disc >= 0.0:
            self._flip_armed = True                    # cursor is inside the outline again
            s = np.sqrt(disc)
            t = (-b - s) if self._side > 0 else (-b + s)   # near / far intersection
            hit = near + t * d
        else:
            if self.interactor.GetShiftKey() and self._flip_armed:
                self._side = -self._side               # touched the edge with Shift held
                self._flip_armed = False
            hit = near + (-b) * d  # point on the ray closest to the center (the outline)
        v = hit - self.center
        n = np.linalg.norm(v)
        if n < 1e-12:
            return current
        return self.center + self.radius * v / n

    def _cancel_camera_motion(self):
        style = self.interactor.GetInteractorStyle()
        if style is not None:
            style.OnLeftButtonUp()  # ends the rotate that the style just started

    def _add_bead_actor(self, xyz, moleculeid):
        actor = self.plotter.add_mesh(self.template, color=color_for_id(moleculeid),
                                      smooth_shading=True, reset_camera=False)
        actor.SetPosition(*xyz)
        return actor

    def _unit(self, p):
        v = np.asarray(p, dtype=float) - self.center
        return v / max(np.linalg.norm(v), 1e-12)

    # ------------------------------------------------------- add / delete molecule
    def _pick_new_ends(self):
        """Two spots on the surface that are as far as possible from existing chain
        ends (and from each other; the new line is at least one volume radius long)."""
        candidates = self.rng.normal(size=(500, 3))
        candidates = self.center + self.radius * candidates / np.linalg.norm(
            candidates, axis=1, keepdims=True)
        existing = self.positions
        if len(existing):
            clearance = np.linalg.norm(
                candidates[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
        else:
            clearance = np.full(len(candidates), np.inf)

        a = candidates[np.argmax(clearance)]
        to_a = np.linalg.norm(candidates - a, axis=1)
        score = np.minimum(clearance, to_a)
        score[to_a < self.radius] = -1.0  # avoid very short molecules
        b = candidates[np.argmax(score)]
        return a, b

    def _add_molecule(self):
        a, b = self._pick_new_ends()
        chain = np.vstack([a, b])               # [start, end]; the molecule is the line between
        mol = self.next_moleculeid
        self.next_moleculeid += 1

        first = len(self.positions)             # appended at the end: no index shifts
        for p in chain:
            self.actors.append(self._add_bead_actor(p, mol))
        self.positions = np.vstack([self.positions, chain])
        self.moleculeids = np.concatenate(
            [self.moleculeids, np.full(2, mol, dtype=np.uint32)])
        if self.on_add_molecule is not None:
            self.on_add_molecule(first, chain.copy(), mol)
        self._pull_network(force=True)
        self.plotter.render()

    def _delete_molecule(self):
        """Delete the whole molecule that the grabbed chain end belongs to."""
        mol = self.moleculeids[self.dragged]
        idx = np.flatnonzero(self.moleculeids == mol)   # contiguous block
        first, count = int(idx[0]), len(idx)

        for actor in self.actors[first:first + count]:
            self.plotter.remove_actor(actor, reset_camera=False)
        del self.actors[first:first + count]
        keep = np.ones(len(self.positions), dtype=bool)
        keep[first:first + count] = False
        self.positions = self.positions[keep]
        self.moleculeids = self.moleculeids[keep]
        self.dragged = None                              # the grabbed bead no longer exists
        if self.on_delete_molecule is not None:
            self.on_delete_molecule(first, count)        # indices in the OLD ordering
        self._pull_network(force=True)                   # its kinks are gone too
        self.plotter.render()

    # ------------------------------------------------------------ volume radius
    def _change_radius(self, delta):
        new = self.radius + delta
        if new < MIN_VOLUME_RADIUS - 1e-9:
            print(f"Volume radius is already at its minimum ({MIN_VOLUME_RADIUS:g}).")
            return
        self.radius = new
        self.shell.SetScale(new, new, new)

        # all chain ends stay on the (new) surface: project them radially
        if len(self.positions):
            v = self.positions - self.center
            r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            self.positions = self.center + new * v / r
            for k, actor in enumerate(self.actors):
                actor.SetPosition(*self.positions[k])
        if self.on_radius_change is not None:
            self.on_radius_change(new)
        self._pull_network()
        print(f"Volume radius: {new:g}")
        self.plotter.render()

    # --------------------------------------------- store / recall chain-end positions
    def _end_keys(self):
        """{(moleculeid, is_start): bead index} for every chain end. This identifies a
        chain end even when molecules come and go."""
        keys = {}
        for i in range(len(self.positions)):
            is_start = (i == 0) or (self.moleculeids[i - 1] != self.moleculeids[i])
            keys[(int(self.moleculeids[i]), bool(is_start))] = i
        return keys

    def _store_ends(self, slot):
        keys = self._end_keys()
        # stored as directions, so a later change of the volume radius doesn't matter
        self.stored[slot] = {k: self._unit(self.positions[i]) for k, i in keys.items()}
        print(f"Stored the positions of {len(keys)} chain ends in slot {slot}")

    def _recall_ends(self, slot):
        if self.dragged is not None:
            return  # ignore while a bead is being dragged
        target = self.stored.get(slot)
        if not target:
            print(f"Nothing stored in slot {slot} (use Shift+{slot} to store)")
            return
        self._cancel_animation()
        items = []
        for key, i in self._end_keys().items():
            if key not in target:
                continue                         # molecule created after storing: stays put
            u0, u1 = self._unit(self.positions[i]), target[key]
            cosang = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
            omega = float(np.arccos(cosang))     # great-circle angle between the two points
            if omega < 1e-9:
                continue
            w = u1 - cosang * u0                 # direction of travel at u0 (tangent to the sphere)
            if np.linalg.norm(w) < 1e-9:         # exactly opposite: any great circle will do
                w = np.cross(u0, [1.0, 0.0, 0.0])
                if np.linalg.norm(w) < 1e-6:
                    w = np.cross(u0, [0.0, 1.0, 0.0])
            items.append((i, u0, w / np.linalg.norm(w), omega, u1))
        if not items:
            return
        self._anim = {"items": items, "t0": time.perf_counter()}
        self._driven = {it[0] for it in items}
        print(f"Moving {len(items)} chain ends to slot {slot}")

    def animate(self):
        """Advance the recall animation: every chain end moves along a great circle at
        a constant angular speed (`recall_speed`, rad/s) and stops at its stored position."""
        if self._anim is None:
            return
        elapsed = time.perf_counter() - self._anim["t0"]
        arc = self.recall_speed * elapsed
        idxs, pos, finished = [], [], True
        for i, u0, w, omega, u1 in self._anim["items"]:
            if arc >= omega:
                u = u1
            else:
                u = u0 * np.cos(arc) + w * np.sin(arc)
                finished = False
            p = self.center + self.radius * u
            self.positions[i] = p
            self.actors[i].SetPosition(*p)
            idxs.append(i)
            pos.append(p)
        if self.on_ends_driven is not None:
            self.on_ends_driven(list(idxs), np.array(pos))
        if finished:
            self._cancel_animation()

    def _cancel_animation(self):
        if self._anim is None:
            return
        released = sorted(self._driven)
        self._anim = None
        self._driven = set()
        if self.on_ends_released is not None:
            self.on_ends_released(released)

    # ------------------------------------------------------- files: image, dump, movie
    def _first_free_number(self, pattern):
        """Next unused file number for names matching `pattern` in the current folder."""
        numbers = [int(m.group(1)) for f in os.listdir(os.getcwd())
                   if (m := pattern.match(f))]
        return max(numbers) + 1 if numbers else 0   # never overwrite earlier files

    def _save_image(self):
        path = os.path.join(os.getcwd(), f"img{self.image_counter:05d}.png")
        self.plotter.screenshot(path)
        print("Saved", path)
        self.image_counter += 1

    def _save_configuration(self):
        """Write confXXXXX.dump (LAMMPS-dump-like text):

            # n_beads n_molecules
            ITEM: NUMBER OF ATOMS
            n_beads
            ITEM: BOX BOUNDS pp pp pp
            xmin xmax / ymin ymax / zmin zmax
            ITEM: ATOMS id mol type xu yu zu
            id molecule beadtype x y z          (one line per bead)

        Ids and molecule numbers start at 1 (molecules are renumbered 1..M in file order);
        beadtype 1 = chain end; the kinks follow after all chain ends with beadtype 2 and the
        number of the molecule they belong to; the box bounds are the smallest and largest
        coordinates."""
        path = os.path.join(os.getcwd(), f"conf{self.conf_counter:05d}.dump")
        order = {int(m): k + 1 for k, m in enumerate(dict.fromkeys(self.moleculeids.tolist()))}
        points = self._all_points()
        if len(points):
            lo, hi = points.min(axis=0), points.max(axis=0)
        else:
            lo = hi = np.zeros(3)
        n_ends = len(self.positions)
        with open(path, "w") as f:
            f.write(f"# {len(points)} {len(order)}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(points)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for k in range(3):
                f.write(f"{lo[k]:.6f} {hi[k]:.6f}\n")
            f.write("ITEM: ATOMS id mol type xu yu zu\n")
            for k, xyz in enumerate(points, start=1):
                if k <= n_ends:     # chain end: type 1
                    mol, typ = order[int(self.moleculeids[k - 1])], 1
                else:               # kink: type 2, belongs to one molecule
                    mol, typ = order.get(self.kink_molecules[k - n_ends - 1], 0), 2
                f.write(f"{k} {mol} {typ} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
        print("Saved", path)
        self.conf_counter += 1

    def _start_movie(self):
        if self.recording:
            print("Already recording", self.movie_path)
            return
        path = os.path.join(os.getcwd(), f"movie{self.movie_counter:05d}.avi")
        try:
            self.plotter.open_movie(path, framerate=self.movie_fps, quality=8, codec="mpeg4")
        except Exception as exc:
            print("Could not start recording (needs: pip install imageio imageio-ffmpeg):", exc)
            return
        self.movie_path = path
        self.movie_counter += 1
        self.recording = True
        print("Recording", path, "- press M to stop")

    def _stop_movie(self):
        if not self.recording:
            return
        self.recording = False
        writer = getattr(self.plotter, "mwriter", None)
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print("Problem while closing the movie file:", exc)
            try:
                del self.plotter.mwriter     # so PyVista doesn't try to close it again
            except AttributeError:
                pass
        print("Saved", self.movie_path)

    def write_movie_frame(self):
        if not self.recording:
            return
        try:
            self.plotter.write_frame()
        except Exception as exc:
            print("Recording failed:", exc)
            self._stop_movie()

    # ------------------------------------------------------------------- events
    def _on_char(self, obj, event):
        """Hide the characters we use from VTK's built-in character shortcuts."""
        code = self.interactor.GetKeyCode()
        if isinstance(code, int):
            code = chr(code)
        if code and code in self.OWN_CHARS:
            try:
                self.interactor.SetKeyCode("\x00")
            except TypeError:
                self.interactor.SetKeyCode(" ")

    def _on_press(self, obj, event):
        x, y = self.interactor.GetEventPosition()
        self.picker.Pick(x, y, 0, self.renderer)
        picked = self.picker.GetActor()
        if picked is None:
            return
        for i, actor in enumerate(self.actors):
            if picked is actor:                           # every bead is a chain end
                self._cancel_animation()                  # grabbing stops a running recall
                self.dragged = i
                self._side = 1 if self._is_front(self.positions[i]) else -1
                self._flip_armed = True
                actor.GetProperty().SetColor(*ACTIVE_COLOR)
                self._cancel_camera_motion()
                self.plotter.render()
                return

    def _on_move(self, obj, event):
        if self.dragged is None:
            return
        i = self.dragged
        x, y = self.interactor.GetEventPosition()
        pos = self._surface_point(x, y, self.positions[i])
        self.positions[i] = pos
        self.actors[i].SetPosition(*pos)
        self._refresh_bonds()
        if self.on_drag is not None:
            self.on_drag(i, pos)
        self.plotter.render()

    def _on_release(self, obj, event):
        if self.dragged is None:
            return
        i = self.dragged
        self.actors[i].GetProperty().SetColor(*color_for_id(self.moleculeids[i]))  # restore
        self.dragged = None
        if self.on_release is not None:
            self.on_release(i, self.positions[i].copy())
        self.plotter.render()

    @staticmethod
    def _digit_slot(key, shift, char="", ctrl=False):
        """Map a key press to (slot 1..9, store?) or None. Handles the plain digit (with the
        Shift or Ctrl modifier reported separately) and the layout-specific symbol or
        character produced by Shift + digit (layout detected from the environment)."""
        if len(key) == 1 and key in "123456789":
            return int(key), shift or ctrl
        if key.startswith("KP_") and len(key) == 4 and key[3] in "123456789":
            return int(key[3]), shift or ctrl
        layout = get_keyboard_layout()
        symbols = SHIFTED_DIGIT_KEYSYMS.get(layout, ())
        if key in symbols:
            return symbols.index(key) + 1, True
        chars = SHIFTED_DIGIT_CHARS.get(layout, "")
        if len(char) == 1 and char in chars:
            return chars.index(char) + 1, True
        return None

    def _on_key(self, obj, event):
        key = self.interactor.GetKeySym() or ""
        shift = bool(self.interactor.GetShiftKey())
        if len(key) == 1 and key.isalpha() and shift:
            key = key.upper()
        lower = key.lower()
        char = self.interactor.GetKeyCode()
        char = chr(char) if isinstance(char, int) else (char or "")
        digit = self._digit_slot(key, shift, char, bool(self.interactor.GetControlKey()))

        if lower == "h" or key == "question" or char == "?":
            print(help_text())
        elif lower == "p":
            self._save_image()
        elif lower == "s":
            self._save_configuration()
        elif lower == "a":
            self._add_molecule()
        elif key == "r":
            self._change_radius(-VOLUME_RADIUS_STEP)
        elif key == "R":
            self._change_radius(+VOLUME_RADIUS_STEP)
        elif key in ("t", "T"):
            self.uniform_lines = key == "t"        # t: thin, equal lines; T: thickness = tension
            self._pull_network()
            self.plotter.render()
        elif lower == "i":
            self._print_options()
        elif lower == "k":
            self._print_diagnostics()
        elif key == "m":
            self._start_movie()
        elif key == "M":
            self._stop_movie()
        elif digit is not None:
            slot, store = digit
            if store:
                self._store_ends(slot)
            else:
                self._recall_ends(slot)
        elif self.dragged is not None and lower == "d":
            self._delete_molecule()
        elif self.dragged is not None and key in ("Right", "Left"):
            if self.on_contour_change is not None:
                self.on_contour_change(self.dragged,
                                       CONTOUR_STEP if key == "Right" else -CONTOUR_STEP)
                self._pull_network()
        elif shift and key.islower() and len(key) > 1 and "_" not in key:
            print(f"Shift+{key}: not recognised as a shifted digit for keyboard layout "
                  f"'{get_keyboard_layout()}'; use Ctrl+digit, or start with --keyboard-layout")


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def make_surface_points(k, center, radius):
    """k points evenly spread over a sphere surface (Fibonacci lattice)."""
    i = np.arange(k) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / k)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    unit = np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])
    return center + radius * unit


def make_molecules(n_molecules, center, radius, rng):
    """Build M molecules, each a straight line between a start bead and an end bead,
    both on the surface. Returns positions (2M, 3) and moleculeids (2M,) uint32.
    Beads are ordered molecule by molecule as [start, end]."""
    anchors = make_surface_points(2 * n_molecules, center, radius)
    anchors = anchors[rng.permutation(2 * n_molecules)]
    starts, ends = anchors[:n_molecules], anchors[n_molecules:]

    positions = np.empty((2 * n_molecules, 3))
    positions[0::2] = starts
    positions[1::2] = ends
    moleculeids = np.repeat(np.arange(n_molecules), 2).astype(np.uint32)
    return positions, moleculeids


def make_perpendicular_pair(center, radius, gap):
    """Starting state: two straight polymers, perpendicular to each other, that pass each other at
    the distance `gap` without touching: one along x (through the centre), one along y, shifted
    by `gap` along z. Their ends are on the surface. Same layout as make_molecules."""
    center = np.asarray(center, dtype=float)
    half = float(np.sqrt(radius ** 2 - gap ** 2))
    positions = np.array([[-radius, 0.0, 0.0], [radius, 0.0, 0.0],
                          [0.0, -half, gap], [0.0, half, gap]]) + center
    return positions, np.array([0, 0, 1, 1], dtype=np.uint32)


def segment_parameters(P, d, Q, e):
    """Closest points between the segments P + s d and Q + t e (s, t in [0, 1]) for arrays of
    segments. Returns the parameters (s, t)."""
    r = P - Q
    a = np.maximum((d * d).sum(1), 1e-12)
    ee = np.maximum((e * e).sum(1), 1e-12)
    b, c, f = (d * e).sum(1), (d * r).sum(1), (e * r).sum(1)
    den = a * ee - b * b
    ok = den > 1e-9 * a * ee                      # not (nearly) parallel
    s = np.where(ok, np.clip((b * f - c * ee) / np.where(ok, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / ee
    s = np.where(t < 0, np.clip(-c / a, 0.0, 1.0),
                 np.where(t > 1, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return s, t


def point_segment_distances(points, A, B):
    """Distance from each of K points to each of S segments A -> B: returns (K, S) distances,
    the (K, S) parameters of the closest points, and the (K, S, 3) vectors point - closest point."""
    D = B - A
    dd = np.maximum((D * D).sum(1), 1e-12)
    s = np.clip(((points[:, None, :] - A[None]) * D[None]).sum(2) / dd[None], 0.0, 1.0)
    vec = points[:, None, :] - (A[None] + s[..., None] * D[None])
    return np.linalg.norm(vec, axis=2), s, vec


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------
class DummySimulation:
    """Polymers as piecewise linear curves with kinks (see the module docstring).

    Polymer m: chain ends R[m,0] and R[m,N] (beads on the sphere, `x`), interior kinks
    R[m,1..N-1] (`kinks[m]`, an (N-1, 3) array) and contour lengths L[m,1..N] (`contours[m]`).
    Springs: F = k_spring / L * (R_i - R_{i-1}),  E = 0.5 k_spring / L * |R_i - R_{i-1}|^2.

    Chain ends stay on the spherical surface and are pushed apart (excluded volume) until the
    centres of ends of different molecules are at least `2 * bead_radius` apart. Ends that the
    viewer is moving are immovable for the simulation.

    One `step()`: `substeps` times { project, separate chain ends, WCA forces + Langevin step of the kinks,
    slip contour length }, then create and destroy kinks."""

    def __init__(self, positions, moleculeids, center, radius, bead_radius, substeps=5,
                 kink_mass=KINK_MASS, kink_friction=KINK_FRICTION, temperature=TEMPERATURE,
                 time_step=TIME_STEP, min_distance=MIN_DISTANCE, min_contour_length=0.5,
                 spring_constant=30.0, slip_rate=0.02, verbose=True, seed=None, wca_epsilon=WCA_EPSILON,
                 distance_capture=DISTANCE_CAPTURE, check_removal_each=CHECK_KINK_REMOVAL_EACH_STEP):
        self.x = np.array(positions, dtype=float).reshape(-1, 3)   # chain ends (n, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.bead_radius = float(bead_radius)
        self.bead_distance = 2.0 * self.bead_radius  # chain ends closer than this = overlap
        self.substeps = substeps        # dynamics substeps per viewer update
        self.kink_mass = float(kink_mass)           # Langevin dynamics of the kinks
        self.kink_friction = float(kink_friction)
        self.temperature = float(temperature)       # kT of the thermal noise
        self.time_step = float(time_step)           # per substep
        self.rng = np.random.default_rng(seed)
        self.spring_constant = float(spring_constant)   # segment spring constant = this / contour length
        self.slip_rate = float(slip_rate)           # speed of contour length slipping across kinks
        self.min_contour_length = float(min_contour_length)
        self.min_distance = float(min_distance)
        self.wca_epsilon = float(wca_epsilon)
        self.distance_capture = float(distance_capture)
        self.check_removal_each = max(int(check_removal_each), 1)   # steps between kink removal checks
        self.wca_sigma = self.min_distance / 2.0 ** (1.0 / 6.0)    # the WCA cutoff is min_distance
        self.verbose = verbose          # print a line when kinks are created / destroyed
        self.step_count = 0
        self.held = set()               # beads currently controlled by the viewer

        self.kinks = {}                 # molecule -> (N-1, 3) positions of its interior kinks
        self.kvel = {}                  # molecule -> (N-1, 3) velocities of its interior kinks
        self.contours = {}              # molecule -> [L_1 ... L_N]
        for m in np.unique(self.moleculeids).tolist():
            a, b = self.x[self.moleculeids == m][:2]
            self.kinks[m] = np.zeros((0, 3))
            self.kvel[m] = np.zeros((0, 3))
            self.contours[m] = [float(np.linalg.norm(b - a))]

    # --- called by the viewer -------------------------------------------------
    def set_surface_position(self, index, xyz):
        """The user is dragging a chain end."""
        self.x[index] = xyz
        self.held.add(index)

    def release(self, index, xyz):
        """The user let go of the chain end."""
        self.x[index] = xyz
        self.held.discard(index)

    def set_driven_positions(self, indices, positions):
        """The viewer moves several chain ends (recall of a stored configuration)."""
        self.x[indices] = positions
        self.held.update(int(i) for i in indices)

    def release_driven(self, indices):
        self.held.difference_update(int(i) for i in indices)

    def set_radius(self, radius):
        """The volume radius changed; chain ends are projected onto the new surface."""
        self.radius = float(radius)
        self._apply_constraints()

    def add_molecule(self, first_index, positions, moleculeid):
        """The user created a molecule; its beads [start, end] are appended."""
        assert first_index == len(self.x)
        k = len(positions)
        self.x = np.vstack([self.x, positions])
        self.moleculeids = np.concatenate(
            [self.moleculeids, np.full(k, moleculeid, dtype=np.uint32)])
        self.kinks[int(moleculeid)] = np.zeros((0, 3))
        self.kvel[int(moleculeid)] = np.zeros((0, 3))
        self.contours[int(moleculeid)] = [float(np.linalg.norm(positions[1] - positions[0]))]

    def delete_molecule(self, first_index, count):
        """The user deleted the molecule occupying beads [first_index, first_index + count)."""
        mol = int(self.moleculeids[first_index])
        self.kinks.pop(mol, None)
        self.kvel.pop(mol, None)
        self.contours.pop(mol, None)
        block = slice(first_index, first_index + count)
        self.x = np.delete(self.x, block, axis=0)
        self.moleculeids = np.delete(self.moleculeids, block)
        self.held = {h - count if h >= first_index + count else h
                     for h in self.held if not (first_index <= h < first_index + count)}

    def change_contour_length(self, index, delta):
        """Add (delta > 0) or remove (delta < 0) contour length of the segment connected to
        chain end `index`. The length stays positive: removing never goes below
        `min_contour_length` (a segment that is already shorter than that is left alone)."""
        mol = int(self.moleculeids[index])
        is_start = index == 0 or int(self.moleculeids[index - 1]) != mol
        cl = self.contours[mol]
        k = 0 if is_start else len(cl) - 1
        new = cl[k] + delta
        if delta < 0:
            new = max(new, min(cl[k], self.min_contour_length))
        cl[k] = new

    def options(self):
        """[(label, value)] with the parameters of the simulation (key i)."""
        return [("min_distance = WCA cutoff", f"{self.min_distance:g}"),
                ("WCA sigma / epsilon", f"{self.wca_sigma:g} / {self.wca_epsilon:g}"),
                ("distance_capture", f"{self.distance_capture:g}"),
                ("kink removal checked every (steps)", f"{self.check_removal_each}"),
                ("SELF_KINK_DISTANCE / INTRA_CONTOUR_DIST", f"{SELF_KINK_DISTANCE:g} / {INTRA_CONTOUR_DIST:g}"),
                ("spring_constant", f"{self.spring_constant:g}"),
                ("kink mass / friction / kT", f"{self.kink_mass:g} / {self.kink_friction:g} / {self.temperature:g}"),
                ("time step per substep", f"{self.time_step:g}"),
                ("slip_rate", f"{self.slip_rate:g}"),
                ("min_contour_length", f"{self.min_contour_length:g}"),
                ("max kink step per substep", f"{MAX_KINK_STEP:g}"),
                ("substeps per step", f"{self.substeps}"),
                ("diagnostics on creation / destruction", "on" if self.verbose else "off")]

    def diagnostics(self):
        """[(label, text)] with the minimal distances (key k):
        - between two kinks of the same polymer (any polymer),
        - between two line segments of different polymers,
        - between two non-adjacent line segments of the same polymer (adjacent ones share a kink)."""
        rows = []
        best = None                                         # (distance, polymer, i, j) kink pair
        for m, K in self.kinks.items():
            if len(K) < 2:
                continue
            d = np.linalg.norm(K[:, None, :] - K[None, :, :], axis=2)
            iu = np.triu_indices(len(K), 1)
            k = int(np.argmin(d[iu]))
            if best is None or d[iu][k] < best[0]:
                best = (float(d[iu][k]), m, int(iu[0][k]) + 1, int(iu[1][k]) + 1)
        rows.append(("min distance, two kinks on the same polymer",
                     "n/a (no polymer has two kinks)" if best is None else
                     f"{best[0]:.5f}  (polymer {best[1]}, kinks {best[2]} and {best[3]})"))
        flat = self._flatten()
        S = len(flat.sa)
        results = {"different": None, "same": None}
        if S >= 2:
            P0, P1 = flat.pos[flat.sa], flat.pos[flat.sb]
            D = P1 - P0
            I, J = np.triu_indices(S, 1)
            smol = flat.mol[flat.sa]
            s_, t_ = segment_parameters(P0[I], D[I], P0[J], D[J])
            gap = np.linalg.norm((P0[I] + s_[:, None] * D[I]) - (P0[J] + t_[:, None] * D[J]), axis=1)
            for key, mask in (("different", smol[I] != smol[J]),
                              ("same", (smol[I] == smol[J]) & (np.abs(flat.sloc[I] - flat.sloc[J]) >= 2))):
                if mask.any():
                    idx = np.flatnonzero(mask)
                    k = idx[int(np.argmin(gap[idx]))]
                    results[key] = (float(gap[k]), int(smol[I[k]]), int(flat.sloc[I[k]]) + 1,
                                    int(smol[J[k]]), int(flat.sloc[J[k]]) + 1)
        for key, label in (("different", "min distance, two segments of different polymers"),
                           ("same", "min distance, two non-adjacent segments of one polymer")):
            r = results[key]
            rows.append((label, "n/a" if r is None else
                         f"{r[0]:.5f}  (polymer {r[1]} segment {r[2]} and polymer {r[3]} segment {r[4]})"))
        rows.append(("min_distance (WCA cutoff) for comparison", f"{self.min_distance:.5f}"))
        return rows

    # --- topology ---------------------------------------------------------------
    def _ends(self):
        """{moleculeid: (first bead index, last bead index)}, in bead order."""
        ends = {}
        for i, m in enumerate(self.moleculeids.tolist()):
            ends[m] = (ends[m][0], i) if m in ends else (i, i)
        return ends

    def _chain(self, m, ends=None):
        """(N+1, 3) positions R[m,0..N] of polymer m: start bead, kinks, end bead."""
        i0, i1 = (ends or self._ends())[m]
        return np.vstack([self.x[i0], self.kinks[m], self.x[i1]])

    def _flatten(self):
        """All nodes of all polymers in one array, polymer by polymer. Returns a namespace with
        pos (nodes, 3), mol (nodes,), kink (nodes,) bool, segment end nodes sa, sb, the local
        index sloc of every segment in its polymer, the contour coordinate `coord` of every node (distance
        along the contour from the start of its polymer), and blocks {mol: (offset, number of nodes)}."""
        ends = self._ends()
        pos, mol, kink, sa, sb, sloc, blocks, coord = [], [], [], [], [], [], {}, []
        off = 0
        for m in ends:
            pts = self._chain(m, ends)
            count = len(pts)
            pos.append(pts)
            coord.append(np.concatenate([[0.0], np.cumsum(self.contours[m])]))   # contour coordinate of every node
            mol += [m] * count
            kink += [False] + [True] * (count - 2) + [False]
            sa += range(off, off + count - 1)
            sb += range(off + 1, off + count)
            sloc += range(count - 1)
            blocks[m] = (off, count)
            off += count
        return SimpleNamespace(pos=np.vstack(pos), mol=np.array(mol), kink=np.array(kink, dtype=bool),
                               sa=np.array(sa, dtype=int), sb=np.array(sb, dtype=int),
                               sloc=np.array(sloc, dtype=int), blocks=blocks,
                               coord=np.concatenate(coord))

    def _store_kinks(self, flat, pos):
        """Write the (moved) kink positions of `pos` back to the polymers."""
        for m, (off, count) in flat.blocks.items():
            self.kinks[m] = pos[off + 1:off + count - 1].copy()

    def network(self):
        """Everything the viewer needs to draw the molecules and kinks:
        (kink positions (K, 3), [moleculeid of every kink], lines).
        `lines` is [(moleculeid, [node, ...], [contour_length, ...], [tension, ...])], the
        nodes ordered start -> end and one contour length and spring tension (k * d, k = K / L)
        per segment; node k < n is chain end k (viewer bead index), node n + r is kink row r."""
        n = len(self.x)
        ends = self._ends()
        kpos, kmols, lines, row = [], [], [], 0
        for m, (i0, i1) in ends.items():
            K = self.kinks[m]
            P = self._chain(m, ends)
            nodes = [i0] + [n + row + j for j in range(len(K))] + [i1]
            d = np.linalg.norm(P[1:] - P[:-1], axis=1)
            tension = self.spring_constant * d / np.array(self.contours[m])
            lines.append((m, nodes, list(self.contours[m]), tension.tolist()))
            kpos.append(K)
            kmols += [m] * len(K)
            row += len(K)
        return np.vstack(kpos) if kpos else np.zeros((0, 3)), kmols, lines

    # --- energies and forces ------------------------------------------------------
    def segment_forces(self, m):
        """F[m,i] = k_spring / L[m,i] * (R[m,i] - R[m,i-1]) for i = 1..N, shape (N, 3)."""
        P = self._chain(m)
        return self.spring_constant / np.array(self.contours[m])[:, None] * (P[1:] - P[:-1])

    def energy(self):
        """Total spring energy: sum of 0.5 k_spring / L[m,i] |R[m,i] - R[m,i-1]|^2."""
        total = 0.0
        for m in self._ends():
            P = self._chain(m)
            d2 = ((P[1:] - P[:-1]) ** 2).sum(1)
            total += float((0.5 * self.spring_constant * d2 / np.array(self.contours[m])).sum())
        return total

    # --- dynamics ---------------------------------------------------------------
    def _apply_constraints(self):
        """Chain ends stay on the spherical surface, kinks inside the ball."""
        v = self.x - self.center
        r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        self.x = self.center + self.radius * v / r
        for m, K in self.kinks.items():
            if len(K):
                v = K - self.center
                r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                self.kinks[m] = self.center + v * np.minimum(1.0, 0.999 * self.radius / r)

    def _separate_molecules(self, iterations=20, tol=1e-6):
        n = len(self.x)
        if n < 2:
            return
        other_molecule = self.moleculeids[:, None] != self.moleculeids[None, :]
        fixed = np.zeros(n, dtype=bool)
        for h in self.held:
            if h < n:
                fixed[h] = True
        idx = np.arange(n)
        tie_sign = np.sign(idx[:, None] - idx[None, :]).astype(float)
        d_min = self.bead_distance

        for _ in range(iterations):
            diff = self.x[:, None, :] - self.x[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            contact = other_molecule & (dist < d_min - tol)
            if not contact.any():
                break
            direction = diff / np.maximum(dist, 1e-12)[..., None]
            tie = dist < 1e-9                       # coincident centres: pick opposite axes
            direction[tie] = 0.0
            direction[tie, 0] = tie_sign[tie]
            # each bead of a pair takes half of the overlap; a held bead takes none,
            # so its partner takes all of it
            share = np.where(fixed[None, :], 1.0, 0.5)
            weight = np.where(contact, (d_min - dist) * share, 0.0)
            push = (weight[..., None] * direction).sum(axis=1)
            push[fixed] = 0.0
            self.x += push
            self._apply_constraints()

    def capture_forces(self, K):
        """Attraction between neighbouring kinks of one polymer (rows of K, in chain order): if the
        distance r of two consecutive kinks is below `distance_capture` (DISTANCE_CAPTURE) they
        attract each other with a force of magnitude  30 r + 40 r^3 / 3 + 160 r^5 / 27
        (F(r) = -30 r - 40 r^3 / 3 - 160 r^5 / 27, the minus sign meaning attraction), directed
        along the line joining them; zero beyond the capture distance. Returns the (K, 3) forces."""
        force = np.zeros_like(K)
        if len(K) < 2:
            return force
        d = K[1:] - K[:-1]
        r = np.linalg.norm(d, axis=1)
        magnitude = np.where(r < self.distance_capture,
                             30.0 * r + 40.0 * r ** 3 / 3.0 + 160.0 * r ** 5 / 27.0, 0.0)
        pull = (magnitude / np.maximum(r, 1e-12))[:, None] * d      # on the first kink of a pair, towards the second
        force[:-1] += pull
        force[1:] -= pull
        return force

    def _wca_force(self, r):
        """Magnitude of the repulsive WCA force at distance r (array): 24 eps / r
        [2 (sigma/r)^12 - (sigma/r)^6] for r < min_distance = 2^(1/6) sigma, zero beyond
        (there the force is zero, so it is continuous), at most WCA_FORCE_CAP."""
        r = np.maximum(r, 1e-6)
        x6 = (self.wca_sigma / r) ** 6
        magnitude = 24.0 * self.wca_epsilon / r * (2.0 * x6 * x6 - x6)
        return np.where(r < self.min_distance, np.minimum(magnitude, WCA_FORCE_CAP), 0.0)

    def repulsive_forces(self, flat=None):
        """WCA forces on all nodes (rows of `_flatten()`; only kink rows are non-zero):
        (1) every segment of another polymer (and of the same polymer if the contour distance to the
        nearer segment end exceeds SELF_KINK_DISTANCE) pushes a kink whose closest point lies inside the
        segment and within min_distance, perpendicular to the line, the reaction going to the two
        end points of the segment (weights 1 - s, s); (2) any two kinks, on the same or on
        different polymers, closer than min_distance repel each other along the line joining them."""
        flat = self._flatten() if flat is None else flat
        pos, sa, sb = flat.pos, flat.sa, flat.sb
        force = np.zeros_like(pos)
        kidx = np.flatnonzero(flat.kink)
        if len(kidx) == 0:
            return force
        # (1) line - kink
        dist, s, vec = point_segment_distances(pos[kidx], pos[sa], pos[sb])
        other = flat.mol[kidx][:, None] != flat.mol[sa][None, :]
        # segments of the same polymer also push a kink, if the contour distance from the kink to
        # the closer (in the contour sense) end of the segment exceeds SELF_KINK_DISTANCE
        ck = flat.coord[kidx][:, None]
        contour = np.minimum(np.abs(ck - flat.coord[sa][None, :]), np.abs(ck - flat.coord[sb][None, :]))
        usable = other | (contour > SELF_KINK_DISTANCE)
        hit = usable & (s > 1e-9) & (s < 1.0 - 1e-9) & (dist < self.min_distance)
        ki, si = np.nonzero(hit)
        if len(ki):
            d, sv = dist[ki, si], s[ki, si]
            line = pos[sb[si]] - pos[sa[si]]
            normal = vec[ki, si] / np.maximum(d, 1e-12)[:, None]
            tiny = d < 1e-9                         # on the line: any direction perpendicular to it
            if tiny.any():
                perp = np.cross(line[tiny], (0.3, 0.5, 0.8))
                normal[tiny] = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-12)
            f = self._wca_force(d)[:, None] * normal
            np.add.at(force, kidx[ki], f)
            np.add.at(force, sa[si], -(1.0 - sv)[:, None] * f)
            np.add.at(force, sb[si], -sv[:, None] * f)
        # (2) kink - kink (same or different polymer)
        diff = pos[kidx][:, None, :] - pos[kidx][None, :, :]
        r = np.linalg.norm(diff, axis=2)
        pair = (r < self.min_distance) & ~np.eye(len(kidx), dtype=bool)    # all kink pairs, same or different polymer
        magnitude = np.where(pair, self._wca_force(r), 0.0)
        np.add.at(force, kidx, (magnitude[..., None] * diff / np.maximum(r, 1e-12)[..., None]).sum(axis=1))
        force[~flat.kink] = 0.0                     # chain ends are not moved by these forces
        return force

    def _move_kinks(self):
        """Damped Langevin step of every kink:
            m dv = F dt - gamma v dt + sqrt(2 gamma kT) dW,   dx = v dt
        F = net force of the two strands, k_{i+1} (R_{i+1} - R_i) + k_i (R_{i-1} - R_i)
        with k = k_spring / L, plus the repulsive WCA forces (`repulsive_forces`). Integrated as
        kick (F) - exact friction and noise (Ornstein-Uhlenbeck) - drift. The displacement is
        limited to MAX_KINK_STEP."""
        flat = self._flatten()
        repulsion = self.repulsive_forces(flat)
        dt, mass = self.time_step, self.kink_mass
        decay = np.exp(-self.kink_friction * dt / mass)
        noise = np.sqrt(max(self.temperature, 0.0) / mass * (1.0 - decay ** 2))
        for m, (off, count) in flat.blocks.items():
            K = self.kinks[m]
            if len(K) == 0:
                continue
            P = flat.pos[off:off + count]
            k = self.spring_constant / np.array(self.contours[m])
            force = (k[:-1, None] * (P[:-2] - P[1:-1]) + k[1:, None] * (P[2:] - P[1:-1])
                     + repulsion[off + 1:off + count - 1])
            force += self.capture_forces(K)
            v = self.kvel[m] + dt * force / mass                       # kick
            v = decay * v + noise * self.rng.standard_normal(v.shape)  # friction + thermal noise
            dx = v * dt                                                # drift
            size = np.linalg.norm(dx, axis=1, keepdims=True)
            dx *= np.minimum(1.0, MAX_KINK_STEP / np.maximum(size, 1e-12))
            self.kvel[m] = dx / dt
            self.kinks[m] = K + dx

    def _slip(self):
        """One slip step: across every kink, move contour length between the two neighbouring
        segments of a molecule towards the energy minimum L_1 : L_2 = d_1 : d_2. All transfers are
        computed from the old lengths and applied together; every transfer is antisymmetric, so
        the total contour length of each molecule is conserved."""
        f = min(1.0 - np.exp(-self.slip_rate), 0.45)
        ends = self._ends()
        for m in ends:
            old = self.contours[m]
            if len(old) < 2:
                continue                                 # no kink, nothing can slip
            P = self._chain(m, ends)
            d = np.linalg.norm(P[1:] - P[:-1], axis=1)
            delta = [0.0] * len(old)
            for k in range(len(old) - 1):
                dsum = d[k] + d[k + 1]
                if dsum < 1e-9:
                    continue
                target = (old[k] + old[k + 1]) * d[k] / dsum   # minimum-energy share of segment k
                move = f * (target - old[k])                   # > 0: segment k gains length
                if move > 0:
                    move = min(move, 0.5 * max(old[k + 1] - self.min_contour_length, 0.0))
                else:
                    move = -min(-move, 0.5 * max(old[k] - self.min_contour_length, 0.0))
                delta[k] += move
                delta[k + 1] -= move
            self.contours[m] = [float(a + b) for a, b in zip(old, delta)]

    # --- creation and destruction of kinks -----------------------------------------
    def _insert_kink(self, m, local, position, fraction):
        """Insert a kink into segment `local` (0-based, from node `local` to node `local + 1`) of
        polymer m at `fraction` along it; the contour length is divided in that proportion."""
        K = self.kinks[m]
        self.kinks[m] = np.vstack([K[:local], np.asarray(position)[None], K[local:]])
        V = self.kvel[m]
        self.kvel[m] = np.vstack([V[:local], np.zeros((1, 3)), V[local:]])
        length = self.contours[m][local]
        self.contours[m][local:local + 1] = [length * fraction, length * (1.0 - fraction)]

    def _remove_kink(self, m, index):
        """Remove interior kink `index` (0-based) of polymer m: the two segments become one."""
        self.kinks[m] = np.delete(self.kinks[m], index, axis=0)
        self.kvel[m] = np.delete(self.kvel[m], index, axis=0)
        cl = self.contours[m]
        cl[index:index + 2] = [cl[index] + cl[index + 1]]

    @staticmethod
    def _violations(P0, D0, Q0, D1):
        """For arrays of segment pairs: (distance, closest point on the first, on the second,
        contact in the interior of both segments?). A contact counts as interior when the closest
        points are at least KINK_MARGIN from both ends of both segments."""
        s, t = segment_parameters(P0, D0, Q0, D1)
        p, q = P0 + s[:, None] * D0, Q0 + t[:, None] * D1
        l0, l1 = np.linalg.norm(D0, axis=1), np.linalg.norm(D1, axis=1)
        inside = ((s * l0 > KINK_MARGIN) & ((1 - s) * l0 > KINK_MARGIN)
                  & (t * l1 > KINK_MARGIN) & ((1 - t) * l1 > KINK_MARGIN))
        return np.linalg.norm(p - q, axis=1), p, q, s, t, inside

    def _kink_allowed(self, m, local, fraction):
        """May a kink be created at `fraction` along segment `local` of polymer m? Not if another
        kink of the same polymer lies within a contour length CONTOUR_KINK_MIN of that point
        (contour distance = the sum of the contour lengths between the two points)."""
        lengths = np.array(self.contours[m])
        position = lengths[:local].sum() + fraction * lengths[local]    # contour coordinate
        existing = np.cumsum(lengths)[:-1]                              # contour coordinates of the kinks
        return not np.any(np.abs(existing - position) < CONTOUR_KINK_MIN - 1e-9)

    def _spatially_free(self, m, position):
        """True if `position` is at least SPATIAL_KINK_MIN away (in space, not along the contour)
        from every existing kink of polymer m: a bent polymer can have kinks that are far apart
        along the contour but close in space."""
        K = self.kinks[m]
        return len(K) == 0 or bool(np.all(np.linalg.norm(K - position, axis=1) >= SPATIAL_KINK_MIN))

    def _create_kinks(self):
        """Where two segments of different molecules are closer than min_distance (closest points
        in the interior of both), insert kinks. The closest pair goes first and the segments are
        recomputed after every insertion. A kink is only created on a line if no other kink of
        that polymer is within a contour length CONTOUR_KINK_MIN (`_kink_allowed`); pairs where
        that fails for both lines are skipped and the next close pair is tried. If it fails for
        one line only, a single kink is created on the other line, min_distance from the closest
        point of the first line; if both are allowed, two kinks are created, exactly
        min_distance apart, symmetrically about the contact point."""
        for _ in range(MAX_NEW_KINKS_PER_STEP):
            flat = self._flatten()
            S = len(flat.sa)
            if S < 2:
                return
            P0, P1 = flat.pos[flat.sa], flat.pos[flat.sb]
            D = P1 - P0
            I, J = np.triu_indices(S, 1)
            smol = flat.mol[flat.sa]
            keep = smol[I] != smol[J]
            I, J = I[keep], J[keep]
            if len(I) == 0:
                return
            dist, p, q, s, t, inside = self._violations(P0[I], D[I], P0[J], D[J])
            bad = np.flatnonzero(inside & (dist < self.min_distance - CREATE_TOL))
            created = False
            for k in bad[np.argsort(dist[bad])]:            # closest pair first
                i, j = I[k], J[k]
                ma, mb = int(smol[i]), int(smol[j])
                la, lb = int(flat.sloc[i]), int(flat.sloc[j])
                ok_a = self._kink_allowed(ma, la, s[k])
                ok_b = self._kink_allowed(mb, lb, t[k])
                if not (ok_a or ok_b):
                    continue                                # too close to existing kinks on both lines
                if dist[k] > 1e-6:
                    normal = (q[k] - p[k]) / dist[k]        # from the point on A to the point on B
                else:                                       # the lines touch: use the common normal
                    normal = np.cross(D[i], D[j])
                    if np.linalg.norm(normal) < 1e-9:
                        helper = np.array([1.0, 0.0, 0.0]) if abs(D[i][0]) < 0.9 * np.linalg.norm(D[i]) \
                            else np.array([0.0, 1.0, 0.0])
                        normal = np.cross(D[i], helper)
                    normal = normal / np.linalg.norm(normal)
                new_a = new_b = None
                if ok_a and ok_b:                           # two kinks, symmetric about the contact
                    mid = 0.5 * (p[k] + q[k])
                    new_a = mid - 0.5 * self.min_distance * normal
                    new_b = mid + 0.5 * self.min_distance * normal
                    if not self._spatially_free(ma, new_a):
                        ok_a = False
                    if not self._spatially_free(mb, new_b):
                        ok_b = False
                if not (ok_a and ok_b):                     # one kink on the line that is free
                    new_a = new_b = None
                    if ok_a:
                        new_a = q[k] - self.min_distance * normal
                        if not self._spatially_free(ma, new_a):
                            new_a = None
                    elif ok_b:
                        new_b = p[k] + self.min_distance * normal
                        if not self._spatially_free(mb, new_b):
                            new_b = None
                    if new_a is None and new_b is None:
                        continue                            # too close to existing kinks

                if new_a is not None:
                    self._insert_kink(ma, la, new_a, s[k])
                if new_b is not None:
                    self._insert_kink(mb, lb, new_b, t[k])
                if self.verbose:
                    made = [str(m) for m, new in ((ma, new_a), (mb, new_b)) if new is not None]
                    print(f"[step {self.step_count}] kink(s) created in polymer {' and '.join(made)} "
                          f"(segments of polymers {ma}, {mb} at distance {dist[k]:.4g} < "
                          f"{self.min_distance:g})")
                created = True
                break
            if not created:
                break
        self._apply_constraints()

    def _interacting_nodes(self, flat, reach):
        """Bool array over the rows of `_flatten()`: kinks that are still interacting with
        another polymer, i.e. that feel, or exert, a WCA force within `reach` (= the cutoff plus
        a margin): a kink within `reach` of a line of another polymer (closest point inside the
        segment; the two end points of that segment exert the reaction) or of a kink of
        another polymer. Kinks of the same polymer do not count here (see `_crowded`)."""
        busy = np.zeros(len(flat.pos), dtype=bool)
        kidx = np.flatnonzero(flat.kink)
        if len(kidx) == 0:
            return busy
        pos, sa, sb = flat.pos, flat.sa, flat.sb
        dist, s, _vec = point_segment_distances(pos[kidx], pos[sa], pos[sb])
        other = flat.mol[kidx][:, None] != flat.mol[sa][None, :]
        ki, si = np.nonzero(other & (s > 1e-9) & (s < 1.0 - 1e-9) & (dist < reach))
        busy[kidx[ki]] = True
        busy[sa[si]] = True
        busy[sb[si]] = True
        r = np.linalg.norm(pos[kidx][:, None, :] - pos[kidx][None, :, :], axis=2)
        pair = (r < reach) & (flat.mol[kidx][:, None] != flat.mol[kidx][None, :])   # kinks of other polymers
        busy[kidx[pair.any(axis=1)]] = True
        return busy & flat.kink

    @staticmethod
    def _crowded(flat, node):
        """True if more than one other kink of the same polymer lies within a contour length
        INTRA_CONTOUR_DIST of kink `node` (row of `_flatten()`)."""
        same = flat.kink & (flat.mol == flat.mol[node])
        near = np.abs(flat.coord[same] - flat.coord[node]) <= INTRA_CONTOUR_DIST
        return int(near.sum()) - 1 > 1                      # minus the kink itself

    def _straight_segment_too_close(self, flat, m, a, b):
        """Would the straight segment a -> b (replacing the two segments at a kink of polymer m)
        come closer than min_distance to any line segment of another polymer? Plain distance
        between the two segments (closest points anywhere on both, ends included), no margin."""
        mask = flat.mol[flat.sa] != m
        count = int(mask.sum())
        if count == 0:
            return False
        Q0, Q1 = flat.pos[flat.sa][mask], flat.pos[flat.sb][mask]
        P0, D0 = np.tile(a, (count, 1)), np.tile(b - a, (count, 1))
        s, t = segment_parameters(P0, D0, Q0, Q1 - Q0)
        gap = np.linalg.norm((P0 + s[:, None] * D0) - (Q0 + t[:, None] * (Q1 - Q0)), axis=1)
        return bool((gap < self.min_distance).any())

    def _destroy_kinks(self):
        """Remove every kink that is further than min_distance (+ DESTROY_TOL) from all segments
        of other polymers and does not interact with kinks of other polymers (`_interacting_nodes`) and has at most one other kink
        of its own polymer within a contour length INTRA_CONTOUR_DIST (`_crowded`),
        unless the straight segment that would replace its two segments is closer than
        min_distance to any line segment of another polymer (`_straight_segment_too_close`).
        One kink at a time; the distances are recomputed after every removal."""
        for _ in range(sum(len(K) for K in self.kinks.values()) + 1):
            flat = self._flatten()
            kidx = np.flatnonzero(flat.kink)
            if len(kidx) == 0:
                return
            sa, sb = flat.sa, flat.sb
            smol = flat.mol[sa]
            other = flat.mol[kidx][:, None] != smol[None, :]
            dist, _s, _v = point_segment_distances(flat.pos[kidx], flat.pos[sa], flat.pos[sb])
            dmin = np.where(other, dist, np.inf).min(axis=1)
            busy = self._interacting_nodes(flat, self.min_distance + DESTROY_TOL)
            removed = False
            for kk in np.argsort(-dmin):
                if dmin[kk] <= self.min_distance + DESTROY_TOL:
                    break                                   # all the others are closer still
                node = kidx[kk]
                if busy[node] or self._crowded(flat, node):
                    continue                                # interacts with other polymers, or crowded
                m = int(flat.mol[node])
                a, b = flat.pos[node - 1], flat.pos[node + 1]
                if self._straight_segment_too_close(flat, m, a, b):
                    continue                                # the straight segment would be too close
                index = node - flat.blocks[m][0] - 1
                self._remove_kink(m, index)
                if self.verbose:
                    print(f"[step {self.step_count}] kink destroyed in polymer {m} "
                          f"(distance to other polymers {dmin[kk]:.4g} > {self.min_distance:g})")
                removed = True
                break
            if not removed:
                return

    def total_length(self):
        """Total length of all lines (sum of the segment lengths d)."""
        total = 0.0
        for m in self._ends():
            P = self._chain(m)
            total += np.linalg.norm(P[1:] - P[:-1], axis=1).sum()
        return float(total)

    def step(self):
        """Advance the simulation and return an (n, 3) array of chain-end positions
        (the kinks are fetched with `network()`)."""
        self.step_count += 1
        for _ in range(self.substeps):
            self._apply_constraints()
            self._separate_molecules()
            self._move_kinks()
            self._apply_constraints()
            self._slip()
        self._create_kinks()
        if self.step_count % self.check_removal_each == 0:
            self._destroy_kinks()                  # removal is only checked every few steps
        return self.x


def main(argv=None):
    args = parse_args(argv)                       # --help prints the summary and exits
    set_keyboard_layout(args.keyboard_layout)
    rng = np.random.default_rng(42)

    # ---- geometry ----
    center = np.zeros(3)
    volume_radius = 15.0    # radius of the spherical volume (r / R change it at run time)
    sphere_radius = 1.0     # radius of each bead

    # positions: (2 * n_molecules, 3) ordered [start, end, start, end, ...]; moleculeids: (n,) uint32
    points, moleculeids = make_perpendicular_pair(center, volume_radius, START_GAP * MIN_DISTANCE)

    # ---- scene ----
    plotter = pv.Plotter(window_size=(1008, 800))
    plotter.set_background("white")
    plotter.add_axes()

    template = pv.Sphere(radius=sphere_radius, theta_resolution=24, phi_resolution=24)
    actors = []
    for p, mid in zip(points, moleculeids):
        actor = plotter.add_mesh(template, color=color_for_id(mid), smooth_shading=True)
        actor.SetPosition(*p)
        actors.append(actor)

    # ---- simulation + viewer wiring ----
    sim = DummySimulation(points, moleculeids, center, volume_radius,
                          bead_radius=sphere_radius, min_distance=MIN_DISTANCE)

    def on_release(i, xyz):
        print(f"Chain end {i} (moleculeid={dragger.moleculeids[i]}) released at",
              np.round(xyz, 4))
        sim.release(i, xyz)

    dragger = SphereDragger(
        plotter, template, actors, points, moleculeids, center, volume_radius,
        sphere_radius=sphere_radius, rng=rng,
        kink_radius=0.5 * sphere_radius,
        get_network=sim.network,
        get_options=sim.options,
        get_diagnostics=sim.diagnostics,
        on_drag=sim.set_surface_position,
        on_release=on_release,
        on_add_molecule=sim.add_molecule,
        on_delete_molecule=sim.delete_molecule,
        on_radius_change=sim.set_radius,
        on_ends_driven=sim.set_driven_positions,
        on_ends_released=sim.release_driven,
        on_contour_change=sim.change_contour_length,
    )

    plotter.add_timer_event(max_steps=10**9, duration=30,
                            callback=lambda step: dragger.tick(sim.step))   # ~30 ms

    plotter.show()
    return dragger.get_positions(), dragger.moleculeids


if __name__ == "__main__":
    main()
