"""
Interactive 3D viewer for linear bead-spring molecules (polymer chains) in a
spherical volume.

Every molecule is   start bead (on the surface) - internal beads (inside) - end bead (on the surface).
Beads are spheres colored by molecule; lines of the same color connect consecutive
beads along each chain. Molecules may have the same number of internal beads. Around
each chain end a circular gray patch (radius 2.25 along the surface, less transparent than
the volume shell) marks where the chain is attached to the surface.

Mouse
- Left-drag on empty space  -> rotate the view
- Right-drag / mouse wheel  -> zoom
- Middle-drag (or Shift+left-drag) -> pan
- Left-drag on a chain-end sphere on the SURFACE -> move it along the surface of the
  volume (internal beads cannot be grabbed; they are moved by the simulation)
- Hold Shift while dragging a chain end and move the cursor out past the circular
  edge of the projected sphere -> the bead crosses to the other side (front <-> back)
  of the volume; move the cursor back inside to keep dragging it there

Keyboard
- While a chain end is grabbed (mouse button held):
      Right arrow -> insert a new bead midway between the grabbed end and the
                     internal bead connected to it
      Left arrow  -> remove the internal bead connected to the grabbed end
      k           -> delete the whole molecule the grabbed end belongs to
- D  -> pause the dynamics: the simulation is no longer stepped, so bead positions are not
        updated by it (you can still drag chain ends, insert/remove beads, recall views, ...)
- d  -> resume the dynamics
- a  -> create a new molecule: a straight line between two free spots on the surface,
        with (distance between the ends / bead radius) internal beads
- r  -> shrink the spherical volume by 1 (down to a minimum radius of 5) and remove the
        internal bead next to both ends of every molecule
- R  -> grow the spherical volume by 1 and insert a bead next to both ends of every molecule
- Shift + 1..9 -> store the current positions of all chain ends, and the rotation and
                  zoom of the view, in slot 1..9
- 1..9         -> slowly move all chain ends to the positions stored in that slot, each
                  along a great circle of the sphere, while slowly rotating and zooming
                  the view to the stored state (touching the mouse takes the view back)
- c  -> show the coordinate system (always flush with the lower-left corner of the window);
        C -> hide it
- B  -> hide the internal beads (chain lines and chain ends stay visible); b -> show them again
- m  -> start recording a movie (movie00000.avi, movie00001.avi, ...); M -> stop recording.
        The movie keeps the view the camera had when recording started: rotating, zooming or
        panning the view yourself does NOT show up in the movie, but a view recalled with
        1..9 does (the movie view then moves from where it was to the stored view).
- p  -> save a PNG of the current view (img00000.png, img00001.png, ...)
- h or ?  -> print all key shortcuts and mouse actions
- s  -> save all bead positions (conf00000.dump, conf00001.dump, ...), LAMMPS-dump-like:
        # <number of beads> <number of molecules>
        ITEM: NUMBER OF ATOMS / <number of beads>
        ITEM: BOX BOUNDS pp pp pp / xmin xmax / ymin ymax / zmin zmax  (extent of the beads)
        ITEM: ATOMS id mol type xu yu zu / one line per bead
        # volume_radius <radius>   (last line; extra information is always prefixed with '#')
        (ids and molecule numbers start at 1; type 1 = chain end, 2 = internal bead)
All files are written to the current folder; numbering continues after existing files.

Command line
    python beadppa.py                          start with two new molecules
    python beadppa.py --load conf00000.dump    restore molecules, positions and
                                               volume radius from a dump file saved with 's'

Data flow
    simulation (LAMMPS)  --positions-->  viewer      (timer, every ~30 ms)
    viewer (user input)  --callbacks-->  simulation  (see SphereDragger.__init__)

Bead ordering: molecule by molecule, each as [start, internal..., end]. Inserting or
removing a bead, or deleting a molecule, shifts the indices of all later beads; the
callbacks report the indices explained in their comments, and the simulation must
return positions in the new ordering from then on. New molecules are appended at
the end, so they don't shift existing indices.

Install:  pip install pyvista numpy
Movies:   pip install imageio imageio-ffmpeg
"""
import argparse
import colorsys
import os
import re
import time

import numpy as np
import pyvista as pv
import vtk

ACTIVE_COLOR = (0.0, 0.0, 0.0)  # highlight color (black) while a sphere is being dragged

CAP_RADIUS = 2.25       # radius (measured along the surface) of the patch around a chain end
CAP_COLOR = (0.5, 0.5, 0.5)     # patches are gray
AXES_LENGTH = 0.23      # coordinate system: length of an axis, as a fraction of the window height
                        # (half of what it was before)
AXES_FONT_SIZE = 24     # size of the X / Y / Z labels in pixels
AXES_MARGIN_PX = 3      # gap between the coordinate system and the window edges
INITIAL_FILL = 0.95     # at the start the sphere (with its beads) spans this fraction of the window
CAP_OPACITY = 0.45      # the volume shell itself is much more transparent (0.12)
CAP_LIFT = 0.03         # patches sit this far outside the shell, to avoid z-fighting

MIN_VOLUME_RADIUS = 5.0
VOLUME_RADIUS_STEP = 1.0

# Shift + digit produces a layout-specific symbol; which key name VTK reports for it
# depends on the keyboard layout (X11/Qt). Pick the layout matching your keyboard:
# keys for Shift+1 ... Shift+9. (Where VTK reports the plain digit together with the
# Shift modifier, e.g. on Windows, no table is needed.) Unknown shifted keys print a
# hint with the key name, so you can add your own entry.
KEYBOARD_LAYOUT = "nordic"
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


HELP_TEXT = f"""\
Mouse
  left-drag on empty space ........ rotate the view
  right-drag / mouse wheel ........ zoom
  middle-drag / Shift+left-drag ... pan
  left-drag on a chain end ........ move the chain end along the surface of the volume
                                    (only chain ends on the surface can be grabbed; the
                                    internal beads are moved by the simulation)
  Shift + left-drag on a chain end, moving the cursor out past the circular edge of the
  projected sphere ................ the bead crosses to the other side (front <-> back);
                                    move the cursor back inside to keep dragging it there

Keyboard, while a chain end is grabbed (mouse button held)
  Right arrow ..................... insert a bead midway between the grabbed end and the
                                    internal bead connected to it
  Left arrow ...................... remove the internal bead connected to the grabbed end
  k ............................... delete the whole molecule of the grabbed end

Keyboard, any time
  a ............................... create a new molecule: a straight line between two free
                                    spots on the surface, with (distance between its ends /
                                    bead radius) internal beads
  r / R ........................... shrink / grow the spherical volume by 1 (minimum radius
                                    {MIN_VOLUME_RADIUS:g}); one internal bead is removed / inserted next
                                    to both ends of every molecule
  Shift+1 ... Shift+9 ............. store the positions of all chain ends and the rotation and
                                    zoom of the view in slot 1 ... 9 (layout: {KEYBOARD_LAYOUT})
  1 ... 9 ......................... slowly move all chain ends along great circles to the
                                    positions stored in that slot, while rotating and zooming
                                    the view to the stored state (touching the mouse takes
                                    the view back)
  D / d ........................... pause / resume the dynamics (the simulation is not stepped
                                    while paused; dragging, insert/remove, recall still work)
  B / b ........................... hide / show the internal beads (chain lines stay visible)
  C / c ........................... hide / show the coordinate system
  m / M ........................... start / stop recording a movie (movie00000.avi, ...); the
                                    movie keeps the view from when recording started, your own
                                    rotating / zooming does not show up in it
  p ............................... save a PNG of the current view (img00000.png, ...)
  s ............................... save all bead positions (conf00000.dump, ...)
  h / ? ........................... print this help

Files are written to the current folder; numbering continues after existing files.
"""


def color_for_id(moleculeid):
    """Deterministic, well-separated RGB color (0..1) for an unsigned integer id."""
    hue = (int(moleculeid) * 0.618033988749895) % 1.0  # golden-ratio hue spacing
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


def axes_gizmo_layout(frame, length, margin, label_pad, label_half, head_length,
                      head_angle=np.radians(25.0)):
    """Where to draw the 2D coordinate system, in pixels from the lower-left window corner.

    `frame` is the camera frame (columns: right, up, back). The three world axes are drawn
    as arrows of `length` pixels times their foreshortening in the current view. The whole
    drawing (arrows, arrow heads and label boxes) is then moved so that it touches the
    left and the bottom window edge (`margin` pixels away), whatever the orientation.

    Returns origin (2,), tips (3, 2), label centres (3, 2) and, per axis, the two end points
    of its arrow head (or None for an axis pointing at the viewer)."""
    right, up = frame[:, 0], frame[:, 1]
    d = np.array([[np.dot(e, right), np.dot(e, up)] for e in np.eye(3)])   # unit axes on screen
    tips = d * length
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    u = np.where(norm > 1e-3, d / np.maximum(norm, 1e-12), np.array([[0.0, 1.0]]))
    labels = tips + u * label_pad

    heads = []
    for tip in tips:
        reach = float(np.linalg.norm(tip))
        if reach < 2.0:
            heads.append(None)
            continue
        back = -tip / reach
        hl = min(head_length, 0.4 * reach)
        rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ back
        heads.append((tip + hl * rot(head_angle), tip + hl * rot(-head_angle)))

    points = [np.zeros(2), *tips, *(labels - label_half), *(labels + label_half)]
    points += [h for pair in heads if pair is not None for h in pair]
    origin = margin - np.min(points, axis=0)           # moves everything up/right to the margin
    heads = [None if pair is None else (pair[0] + origin, pair[1] + origin) for pair in heads]
    return origin, tips + origin, labels + origin, heads


class AxesGizmo:
    """The coordinate system: three colored arrows with X / Y / Z labels, drawn as a 2D
    overlay that is rebuilt before every render from the camera that is rendering. It is
    always flush with the lower-left corner of the window, and its size is given in pixels."""

    COLORS = ((0.85, 0.15, 0.15), (0.15, 0.65, 0.15), (0.2, 0.3, 0.95))

    def __init__(self, plotter, length=AXES_LENGTH, font_size=AXES_FONT_SIZE,
                 margin=AXES_MARGIN_PX, visible=True):
        self.renderer = plotter.renderer
        self.length, self.font_size, self.margin = length, font_size, margin
        self.visible = bool(visible)
        self.arrows, self.labels = [], []
        for color, name in zip(self.COLORS, "XYZ"):
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(4)                       # origin, tip, two arrow-head ends
            lines = vtk.vtkCellArray()
            for a, b in ((0, 1), (1, 2), (1, 3)):             # shaft and the two head strokes
                lines.InsertNextCell(2)
                lines.InsertCellPoint(a)
                lines.InsertCellPoint(b)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            arrow = vtk.vtkActor2D()
            arrow.SetMapper(mapper)
            arrow.GetProperty().SetColor(*color)
            arrow.GetProperty().SetLineWidth(3)
            label = vtk.vtkTextActor()
            label.SetInput(name)
            text = label.GetTextProperty()
            text.SetColor(*color)
            text.SetFontSize(font_size)
            text.BoldOn()
            text.SetJustificationToCentered()
            text.SetVerticalJustificationToCentered()
            self.renderer.AddViewProp(arrow)
            self.renderer.AddViewProp(label)
            self.arrows.append((poly, points, arrow))
            self.labels.append(label)
        self.set_visible(self.visible)
        self.renderer.AddObserver("StartEvent", self._update)   # before every render

    def set_visible(self, visible):
        self.visible = bool(visible)
        for _, _, arrow in self.arrows:
            arrow.SetVisibility(self.visible)
        for label in self.labels:
            label.SetVisibility(self.visible)

    def _update(self, obj=None, event=None):
        if not self.visible:
            return
        width, height = self.renderer.GetSize()
        cam = self.renderer.GetActiveCamera()
        offset = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        if width <= 0 or height <= 0 or np.linalg.norm(offset) < 1e-12:
            return
        origin, tips, labels, heads = axes_gizmo_layout(
            camera_frame(offset, cam.GetViewUp()), self.length * min(width, height),
            self.margin, label_pad=0.55 * self.font_size + 6.0,
            label_half=np.array([0.4, 0.55]) * self.font_size, head_length=14.0)
        for i, (poly, points, _) in enumerate(self.arrows):
            h1, h2 = heads[i] if heads[i] is not None else (tips[i], tips[i])
            for k, p in enumerate((origin, tips[i], h1, h2)):
                points.SetPoint(k, float(p[0]), float(p[1]), 0.0)
            points.Modified()
            poly.Modified()
            self.labels[i].SetPosition(float(labels[i][0]), float(labels[i][1]))


def set_initial_view(plotter, center, radius, bead_radius, fill=INITIAL_FILL):
    """Fit the scene to the window, then zoom so that the sphere (plus the beads sitting on
    its surface) spans `fill` of the window's smaller dimension. Marking the camera as set
    stops PyVista from resetting it when the window opens."""
    plotter.reset_camera()
    cam = plotter.renderer.GetActiveCamera()
    distance = np.linalg.norm(np.array(cam.GetPosition()) - np.asarray(center, dtype=float))
    r = radius + bead_radius
    sphere = r / np.sqrt(max(distance ** 2 - r ** 2, 1e-12))      # tan of the half angle it spans
    width, height = plotter.window_size
    window = np.tan(np.radians(cam.GetViewAngle()) / 2.0) * min(1.0, width / height)
    plotter.camera.zoom(fill * window / sphere)
    plotter.camera_position = plotter.camera_position


def make_bond_mesh(positions, moleculeids):
    """Line segments between consecutive beads of the same molecule, colored by
    molecule. Returns None when there is nothing to draw."""
    segments = [i for i in range(len(positions) - 1) if moleculeids[i] == moleculeids[i + 1]]
    if not segments:
        return None
    lines = np.hstack([[2, i, i + 1] for i in segments]).astype(np.int64)
    mesh = pv.PolyData(np.array(positions, dtype=float), lines=lines)
    mesh.cell_data["rgb"] = np.array(
        [np.array(color_for_id(moleculeids[i])) * 255 for i in segments]).astype(np.uint8)
    return mesh


def camera_frame(back, up):
    """Right-handed orthonormal camera frame [right, up, back] as the columns of a 3x3
    matrix. `back` points from the focal point to the camera."""
    z = np.asarray(back, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.asarray(up, dtype=float) - np.dot(up, z) * z
    if np.linalg.norm(y) < 1e-9:                      # degenerate view-up: pick any
        y = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(z, [0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def matrix_to_quaternion(m):
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix."""
    t = np.trace(m)
    if t > 0.0:
        k = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * k, (m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        k = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / k, 0.25 * k, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k]
    elif m[1, 1] > m[2, 2]:
        k = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / k, (m[0, 1] + m[1, 0]) / k, 0.25 * k, (m[1, 2] + m[2, 1]) / k]
    else:
        k = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / k, (m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, 0.25 * k]
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp_quaternion(q0, q1, s):
    """Spherical interpolation between unit quaternions (q1 must be on the same
    hemisphere as q0, i.e. dot(q0, q1) >= 0)."""
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:                                    # nearly equal: linear is fine
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(d)
    return (np.sin((1.0 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


def make_cap_template(alpha, n_rings=4, n_seg=48):
    """Triangulated spherical cap on the unit sphere, centred on +z, with angular
    radius `alpha` (radians). Returns vertices (m, 3) and triangles (t, 3)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    verts = [[0.0, 0.0, 1.0]]
    for j in range(1, n_rings + 1):
        a = alpha * j / n_rings
        verts += [[np.sin(a) * np.cos(t), np.sin(a) * np.sin(t), np.cos(a)] for t in theta]
    tris = [[0, 1 + k, 1 + (k + 1) % n_seg] for k in range(n_seg)]       # fan around the centre
    for j in range(1, n_rings):
        inner, outer = 1 + (j - 1) * n_seg, 1 + j * n_seg
        for k in range(n_seg):
            k2 = (k + 1) % n_seg
            tris += [[inner + k, outer + k, outer + k2], [inner + k, outer + k2, inner + k2]]
    return np.array(verts), np.array(tris, dtype=np.int64)


class SphereDragger:
    """Lets the user drag chain ends along a spherical surface, insert/remove beads,
    add/delete molecules, resize the volume, store/recall chain-end configurations,
    and save images / movies / bead positions; lets a simulation push new positions
    for all beads. Keeps the chain lines and the volume shell in sync."""

    IMAGE_PATTERN = re.compile(r"^img(\d{5})\.png$")
    CONF_PATTERN = re.compile(r"^conf(\d{5})\.dump$")
    MOVIE_PATTERN = re.compile(r"^movie(\d{5})\.avi$")
    # characters whose built-in VTK meaning (reset camera, surface mode, pick, ...) is suppressed
    OWN_CHARS = frozenset("aAbBcCdDhH?kKpPsSmMrR123456789")

    def __init__(self, plotter, template, actors, positions, moleculeids, surface_mask,
                 center, radius, sphere_radius, recall_speed=0.5, movie_fps=30, rng=None,
                 show_internal=True, show_axes=True, axes=None,
                 on_drag=None, on_release=None, on_insert=None, on_remove=None,
                 on_add_molecule=None, on_delete_molecule=None,
                 on_radius_change=None, on_ends_driven=None, on_ends_released=None):
        self.plotter = plotter
        self.renderer = plotter.renderer
        self.template = template        # sphere mesh used for newly created beads
        self.actors = list(actors)
        self.positions = np.array(positions, dtype=float).reshape(-1, 3)   # (n, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.surface_mask = np.asarray(surface_mask, dtype=bool)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)             # radius of the spherical volume
        self.sphere_radius = float(sphere_radius)   # radius of one bead
        self.recall_speed = float(recall_speed)     # rad/s along great circles
        self.movie_fps = movie_fps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.next_moleculeid = int(self.moleculeids.max()) + 1 if len(self.moleculeids) else 0
        self.image_counter = self._first_free_number(self.IMAGE_PATTERN)
        self.conf_counter = self._first_free_number(self.CONF_PATTERN)
        self.movie_counter = self._first_free_number(self.MOVIE_PATTERN)

        # Callbacks into the simulation:
        self.on_drag = on_drag          # on_drag(index, xyz)                    while dragging
        self.on_release = on_release    # on_release(index, xyz)                 when let go
        self.on_insert = on_insert      # on_insert(index, xyz, moleculeid)      bead inserted
        self.on_remove = on_remove      # on_remove(index)                       bead removed
        self.on_add_molecule = on_add_molecule        # (first_index, positions (k, 3), moleculeid)
        self.on_delete_molecule = on_delete_molecule  # (first_index, count)  old indices
        self.on_radius_change = on_radius_change      # (new_volume_radius)
        self.on_ends_driven = on_ends_driven          # (indices, positions) chain ends moved by a recall
        self.on_ends_released = on_ends_released      # (indices)            recall finished/cancelled

        self.interactor = plotter.iren.interactor
        self.picker = vtk.vtkPropPicker()
        self.dragged = None             # index of the chain end being dragged
        self._side = 1                  # +1: dragged bead on the front (camera) side, -1: back
        self._flip_armed = True         # a Shift + "touch the edge" gesture may flip the side
        self.stored = {}                # slot (1..9) -> {(moleculeid, is_start): unit vector}
        self._anim = None               # running recall animation
        self._driven = set()            # indices of chain ends moved by the animation
        self.recording = False
        self.movie_path = None
        self._movie_writer = None       # imageio writer while recording
        self._movie_camera = None       # camera used for the movie frames while recording
        self._in_tick = False           # re-entrancy guard for tick()
        self.paused = False             # True: the simulation is not stepped
        self.show_internal = bool(show_internal)    # are the internal beads drawn?
        self.show_axes = bool(show_axes)            # is the coordinate system drawn?
        self.axes = axes                            # AxesGizmo (None: PyVista's axes widget)
        self.bond_mesh = None
        self.cap_mesh = None            # patches around the chain ends
        self._cap_unit = None           # unit-sphere cap template for the current radius

        # Translucent shell showing the volume (unit sphere, scaled to the current radius).
        # Not pickable, so it never blocks grabbing.
        self.shell = plotter.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64),
            color="lightgray", opacity=0.12, smooth_shading=True)
        self.shell.SetPickable(False)
        self.shell.SetPosition(*self.center)
        self.shell.SetScale(self.radius, self.radius, self.radius)
        self._rebuild_bonds()
        self._apply_internal_visibility()

        # Make sure PyVista's own key shortcuts for the keys we use don't also fire.
        for key in ("Up", "Down", "Left", "Right", "a", "b", "B", "c", "C", "d", "D", "h", "H",
                    "question", "k", "K", "p", "s", "S", "m", "M", "r", "R", *"123456789"):
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
        # Touching the camera with the mouse stops a running view animation.
        for ev in ("RightButtonPressEvent", "MiddleButtonPressEvent",
                   "MouseWheelForwardEvent", "MouseWheelBackwardEvent"):
            self.interactor.AddObserver(ev, self._on_view_interaction, -1.0)
        # VTK's interactor style gives some plain characters a built-in meaning on the
        # CharEvent that follows a key press: 'p' = pick + draw an outline box around the
        # prop under the cursor (and warn "no current renderer" before the first mouse
        # interaction), 's' = surface mode, 'r' = reset camera, '3' = stereo, ... A
        # higher-priority observer neutralises the characters we use before the style sees them.
        self.interactor.AddObserver("CharEvent", self._on_char, 10.0)

    # ------------------------------------------------------------ timer tick
    def tick(self, step_simulation):
        """Call every ~30 ms: advance a running recall animation, take a simulation
        step (`step_simulation()` returns the (n, 3) positions; skipped while the dynamics
        are paused), draw, record a frame.

        Re-entrancy guard: grabbing a movie frame renders the window again, and VTK may
        deliver the next timer event while that is still going on. Without the guard each
        nested timer event would start another tick (and grab another frame) inside the
        previous one until Python's recursion limit is hit. A tick that arrives while one
        is still running is simply skipped."""
        if self._in_tick:
            return
        self._in_tick = True
        try:
            self.animate()
            if self.paused:                 # dynamics paused: keep what is on screen, draw changes
                self._refresh_bonds()       # made by the user / a recall animation
                self.plotter.render()
            else:
                self.set_positions(step_simulation())
            self.write_movie_frame()
        finally:
            self._in_tick = False

    # ------------------------------------------------------- simulation -> view
    def set_positions(self, positions):
        """Update all bead positions (e.g. from LAMMPS). Beads held by the user or moved
        by a recall animation keep their viewer-controlled positions."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if len(positions) != len(self.actors):
            raise ValueError(f"Simulation returned {len(positions)} beads, "
                             f"viewer has {len(self.actors)}.")
        for i, actor in enumerate(self.actors):
            if i != self.dragged and i not in self._driven:
                self.positions[i] = positions[i]
                actor.SetPosition(*positions[i])
        self._refresh_bonds()
        self.plotter.render()

    def get_positions(self):
        return self.positions.copy()

    # ------------------------------------------------------------ chain lines
    def _refresh_bonds(self):
        """Cheap update when only positions changed (chain lines and end patches)."""
        if self.bond_mesh is not None:
            self.bond_mesh.points = self.positions.copy()
        self._refresh_caps()

    # ---------------------------------------------------- patches around chain ends
    def _cap_vertices(self, idx):
        """World-space vertices of the patches of the chain ends `idx` (k * m, 3): each
        patch is the template cap rotated so that its centre points at the chain end."""
        u = self.positions[idx] - self.center
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        helper = np.where(np.abs(u[:, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        e1 = np.cross(u, helper)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(u, e1)
        frames = np.stack([e1, e2, u], axis=1)                     # (k, 3, 3), rows e1 e2 u
        unit = np.einsum("mc,kcj->kmj", self._cap_unit, frames)    # (k, m, 3)
        return (self.center + (self.radius + CAP_LIFT) * unit).reshape(-1, 3)

    def _rebuild_caps(self):
        """Needed when the number of chain ends or the volume radius changed."""
        idx = np.flatnonzero(self.surface_mask)
        if len(idx) == 0:
            self.cap_mesh = None
            self.plotter.remove_actor("caps", reset_camera=False)
            return
        alpha = min(CAP_RADIUS / self.radius, 0.5 * np.pi)         # angular radius of a patch
        self._cap_unit, tris = make_cap_template(alpha)
        k, m, t = len(idx), len(self._cap_unit), len(tris)
        faces = np.column_stack(
            [np.full(k * t, 3), (tris[None, :, :] + (np.arange(k) * m)[:, None, None]).reshape(-1, 3)])
        self.cap_mesh = pv.PolyData(self._cap_vertices(idx), faces=faces.ravel().astype(np.int64))
        # same `name` => pyvista replaces the previous patch actor
        actor = self.plotter.add_mesh(
            self.cap_mesh, color=CAP_COLOR, opacity=CAP_OPACITY, lighting=False,
            name="caps", reset_camera=False)
        actor.SetPickable(False)  # patches never block grabbing a sphere

    def _refresh_caps(self):
        """Cheap update when only positions changed."""
        if self.cap_mesh is not None:
            self.cap_mesh.points = self._cap_vertices(np.flatnonzero(self.surface_mask))

    # ---------------------------------------------------- internal bead visibility
    def _apply_internal_visibility(self):
        for actor, is_end in zip(self.actors, self.surface_mask):
            if not is_end:
                actor.SetVisibility(self.show_internal)

    def _set_paused(self, paused):
        self.paused = bool(paused)
        print("Dynamics", "paused" if self.paused else "resumed")

    def _set_axes_visible(self, visible):
        self.show_axes = bool(visible)
        if self.axes is not None:
            self.axes.set_visible(self.show_axes)
        elif self.show_axes:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        print("Coordinate system", "shown" if self.show_axes else "hidden")
        self.plotter.render()

    def _set_internal_visible(self, visible):
        self.show_internal = bool(visible)
        self._apply_internal_visibility()
        print("Internal beads", "shown" if visible else "hidden")
        self.plotter.render()

    def _rebuild_bonds(self):
        """Needed when the number of beads changed."""
        self._rebuild_caps()
        self.bond_mesh = make_bond_mesh(self.positions, self.moleculeids)
        if self.bond_mesh is None:
            self.plotter.remove_actor("bonds", reset_camera=False)
            return
        # same `name` => pyvista replaces the previous line actor
        actor = self.plotter.add_mesh(
            self.bond_mesh, scalars="rgb", rgb=True, line_width=4,
            render_lines_as_tubes=True, name="bonds", reset_camera=False)
        actor.SetPickable(False)  # lines never block grabbing a sphere

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

    def _chain_neighbour(self, i):
        """For chain end i return (is_start, index of the adjacent bead in its molecule)."""
        is_start = (i == 0) or (self.moleculeids[i - 1] != self.moleculeids[i])
        return is_start, (i + 1 if is_start else i - 1)

    def _add_bead_actor(self, xyz, moleculeid, visible=True):
        actor = self.plotter.add_mesh(self.template, color=color_for_id(moleculeid),
                                      smooth_shading=True, reset_camera=False)
        actor.SetPosition(*xyz)
        actor.SetVisibility(visible)
        return actor

    def _unit(self, p):
        v = np.asarray(p, dtype=float) - self.center
        return v / max(np.linalg.norm(v), 1e-12)

    # ------------------------------------------------------- insert / remove bead
    def _insert_bead_at(self, i):
        """Insert a new bead midway between chain end i and the bead connected to it.
        Does not redraw the chain lines; call _rebuild_bonds() afterwards."""
        is_start, nb = self._chain_neighbour(i)
        mid = 0.5 * (self.positions[i] + self.positions[nb])  # between end and its neighbour
        mol = self.moleculeids[i]

        # New bead sits right next to the end, inside the chain sequence:
        #   start: [start, NEW, old first internal, ...]   -> NEW at i + 1
        #   end:   [..., old last internal, NEW, end]      -> NEW at i (end moves to i + 1)
        j = i + 1 if is_start else i

        self.actors.insert(j, self._add_bead_actor(mid, mol, self.show_internal))
        self.positions = np.insert(self.positions, j, mid, axis=0)
        self.moleculeids = np.insert(self.moleculeids, j, mol)
        self.surface_mask = np.insert(self.surface_mask, j, False)
        if self.dragged is not None and self.dragged >= j:
            self.dragged += 1  # the grabbed bead shifted by one
        if self.on_insert is not None:
            self.on_insert(j, mid.copy(), int(mol))

    def _remove_bead_at(self, i):
        """Remove the internal bead connected to chain end i. Returns False (and does
        nothing) if the molecule has no internal bead on that side. Does not redraw the
        chain lines; call _rebuild_bonds() afterwards."""
        is_start, nb = self._chain_neighbour(i)
        if self.surface_mask[nb]:
            return False  # molecule has no internal beads left; the neighbour is the other end

        j = nb  # the internal bead connected to the end
        self.plotter.remove_actor(self.actors.pop(j), reset_camera=False)
        self.positions = np.delete(self.positions, j, axis=0)
        self.moleculeids = np.delete(self.moleculeids, j)
        self.surface_mask = np.delete(self.surface_mask, j)
        if self.dragged is not None and self.dragged > j:
            self.dragged -= 1  # the grabbed bead shifted by one
        if self.on_remove is not None:
            self.on_remove(j)
        return True

    def _insert_bead(self):
        """Right arrow: insert a bead at the grabbed chain end."""
        self._insert_bead_at(self.dragged)
        self._rebuild_bonds()
        self.plotter.render()

    def _remove_bead(self):
        """Left arrow: remove the internal bead connected to the grabbed chain end."""
        if self._remove_bead_at(self.dragged):
            self._rebuild_bonds()
            self.plotter.render()

    def _molecule_blocks(self):
        """[(first, last)] bead index range of every molecule, in bead order."""
        n = len(self.moleculeids)
        starts = [i for i in range(n) if i == 0 or self.moleculeids[i] != self.moleculeids[i - 1]]
        bounds = starts + [n]
        return [(bounds[k], bounds[k + 1] - 1) for k in range(len(starts))]

    # ------------------------------------------------------- add / delete molecule
    def _pick_new_ends(self):
        """Two spots on the surface that are as far as possible from existing chain
        ends (and from each other; the chain is at least one volume radius long)."""
        candidates = self.rng.normal(size=(500, 3))
        candidates = self.center + self.radius * candidates / np.linalg.norm(
            candidates, axis=1, keepdims=True)
        existing = self.positions[self.surface_mask]
        if len(existing):
            clearance = np.linalg.norm(
                candidates[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
        else:
            clearance = np.full(len(candidates), np.inf)

        a = candidates[np.argmax(clearance)]
        to_a = np.linalg.norm(candidates - a, axis=1)
        score = np.minimum(clearance, to_a)
        score[to_a < self.radius] = -1.0  # avoid very short chains
        b = candidates[np.argmax(score)]
        return a, b

    def _add_molecule(self):
        a, b = self._pick_new_ends()
        # number of internal beads = distance between the ends / bead radius
        n_int = max(0, int(round(np.linalg.norm(b - a) / self.sphere_radius)))
        t = np.linspace(0.0, 1.0, n_int + 2)[:, None]
        chain = a + t * (b - a)                 # straight line, evenly spaced beads
        mol = self.next_moleculeid
        self.next_moleculeid += 1

        first = len(self.positions)             # appended at the end: no index shifts
        for k, p in enumerate(chain):
            is_end = k in (0, len(chain) - 1)
            self.actors.append(self._add_bead_actor(p, mol, is_end or self.show_internal))
        self.positions = np.vstack([self.positions, chain])
        self.moleculeids = np.concatenate(
            [self.moleculeids, np.full(len(chain), mol, dtype=np.uint32)])
        self.surface_mask = np.concatenate(
            [self.surface_mask, [True] + [False] * n_int + [True]])
        self._rebuild_bonds()
        if self.on_add_molecule is not None:
            self.on_add_molecule(first, chain.copy(), mol)
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
        self.surface_mask = self.surface_mask[keep]
        self.dragged = None                              # the grabbed bead no longer exists
        self._rebuild_bonds()
        if self.on_delete_molecule is not None:
            self.on_delete_molecule(first, count)        # indices in the OLD ordering
        self.plotter.render()

    # ------------------------------------------------------------ volume radius
    def _change_radius(self, delta):
        new = self.radius + delta
        if new < MIN_VOLUME_RADIUS - 1e-9:
            print(f"Volume radius is already at its minimum ({MIN_VOLUME_RADIUS:g}).")
            return
        self._cancel_animation()        # bead indices change below
        self.radius = new
        self.shell.SetScale(new, new, new)

        # all chain ends stay on the (new) surface: project them radially
        idx = np.flatnonzero(self.surface_mask)
        if len(idx):
            v = self.positions[idx] - self.center
            r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            self.positions[idx] = self.center + new * v / r
            for k in idx:
                self.actors[k].SetPosition(*self.positions[k])
        if self.on_radius_change is not None:
            self.on_radius_change(new)

        # A larger volume gets one more bead at both ends of every molecule, a smaller
        # volume one bead fewer. Molecules are processed from the back so that the
        # indices of the molecules still to be processed don't shift.
        changed = 0
        for first, last in reversed(self._molecule_blocks()):
            if delta > 0:
                self._insert_bead_at(last)      # end bead first: `first` stays valid
                self._insert_bead_at(first)
                changed += 2
            else:
                changed += int(self._remove_bead_at(last))
                changed += int(self._remove_bead_at(first))
        self._rebuild_bonds()
        print(f"Volume radius: {new:g} ({'inserted' if delta > 0 else 'removed'} {changed} beads)")
        self.plotter.render()

    # --------------------------------------------- store / recall chain-end positions
    def _end_keys(self):
        """{(moleculeid, is_start): bead index} for every chain end. This identifies a
        chain end even when beads are inserted/removed or molecules come and go."""
        keys = {}
        for i in np.flatnonzero(self.surface_mask):
            is_start = (i == 0) or (self.moleculeids[i - 1] != self.moleculeids[i])
            keys[(int(self.moleculeids[i]), bool(is_start))] = int(i)
        return keys

    def _get_view(self, cam=None):
        """Rotation and zoom of a camera's view (default: the user's camera). The focal
        point, i.e. panning, is not part of it."""
        if cam is None:
            cam = self.renderer.GetActiveCamera()
        offset = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        distance = max(float(np.linalg.norm(offset)), 1e-12)
        return {"frame": camera_frame(offset / distance, cam.GetViewUp()),
                "distance": distance,                       # perspective zoom (dolly)
                "parallel_scale": float(cam.GetParallelScale()),   # parallel-projection zoom
                "view_angle": float(cam.GetViewAngle())}

    def _set_view(self, frame, distance, parallel_scale, view_angle, cam=None):
        user_camera = cam is None
        if user_camera:
            cam = self.renderer.GetActiveCamera()
        focal = np.array(cam.GetFocalPoint())
        cam.SetPosition(*(focal + frame[:, 2] * distance))
        cam.SetViewUp(*frame[:, 1])
        cam.SetParallelScale(parallel_scale)
        cam.SetViewAngle(view_angle)
        if user_camera:
            self.renderer.ResetCameraClippingRange()

    @staticmethod
    def _plan_view(v0, v1):
        """How to get from view v0 to view v1: rotate along the shortest arc (quaternion
        slerp) and zoom on a log scale. Returns the plan and its size (radians / e-folds)."""
        q0, q1 = matrix_to_quaternion(v0["frame"]), matrix_to_quaternion(v1["frame"])
        if np.dot(q0, q1) < 0.0:
            q1 = -q1                             # same rotation, shorter way round
        rotation = 2.0 * float(np.arccos(min(1.0, np.dot(q0, q1))))
        zoom = max(abs(np.log(max(v1["distance"], 1e-12) / v0["distance"])),
                   abs(np.log(max(v1["parallel_scale"], 1e-12) / max(v0["parallel_scale"], 1e-12))),
                   np.radians(abs(v1["view_angle"] - v0["view_angle"])))
        return {"v0": v0, "v1": v1, "q0": q0, "q1": q1, "f0": 0.0}, max(rotation, zoom)

    def _apply_view_plan(self, plan, f, cam=None):
        """Put a camera at fraction f (0..1) of the whole animation; the plan itself starts
        at fraction plan["f0"] (later than 0 if it was made while the animation was running)."""
        g = min(1.0, max(0.0, (f - plan["f0"]) / max(1.0 - plan["f0"], 1e-9)))
        v0, v1 = plan["v0"], plan["v1"]
        ratio = lambda a, b: max(a, 1e-12) / max(b, 1e-12)
        self._set_view(
            quaternion_to_matrix(slerp_quaternion(plan["q0"], plan["q1"], g)),
            v0["distance"] * ratio(v1["distance"], v0["distance"]) ** g,
            v0["parallel_scale"] * ratio(v1["parallel_scale"], v0["parallel_scale"]) ** g,
            v0["view_angle"] + g * (v1["view_angle"] - v0["view_angle"]),
            cam=cam)

    def _store_ends(self, slot):
        keys = self._end_keys()
        # ends are stored as directions, so a later change of the volume radius doesn't matter
        self.stored[slot] = {
            "ends": {k: self._unit(self.positions[i]) for k, i in keys.items()},
            "view": self._get_view()}
        print(f"Stored the positions of {len(keys)} chain ends and the view in slot {slot}")

    def _recall_ends(self, slot):
        if self.dragged is not None:
            return  # ignore while a bead is being dragged
        stored = self.stored.get(slot)
        if not stored:
            print(f"Nothing stored in slot {slot} (use Shift+{slot} to store)")
            return
        target = stored["ends"]
        self._cancel_animation()
        items = []
        for key, i in self._end_keys().items():
            if key not in target:
                continue                         # molecule created after storing: stays put
            u0, u1 = self._unit(self.positions[i]), target[key]
            cosang = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
            omega = float(np.arccos(cosang))     # great-circle angle between the two points
            if omega < 1e-6:                     # already there (to ~1e-5 length units)
                continue
            w = u1 - cosang * u0                 # direction of travel at u0 (tangent to the sphere)
            if np.linalg.norm(w) < 1e-9:         # exactly opposite: any great circle will do
                w = np.cross(u0, [1.0, 0.0, 0.0])
                if np.linalg.norm(w) < 1e-6:
                    w = np.cross(u0, [0.0, 1.0, 0.0])
            items.append((i, u0, w / np.linalg.norm(w), omega, u1))

        # The view: rotate and zoom at the same angular speed as the chain ends; it takes as
        # long as the slowest of them. While recording, the movie camera makes the same move
        # starting from its own view (the user's own rotating/zooming doesn't move it).
        user_plan, size = self._plan_view(self._get_view(), stored["view"])
        movie_plan = None
        if self.recording and self._movie_camera is not None:
            movie_plan, movie_size = self._plan_view(self._get_view(self._movie_camera),
                                                     stored["view"])
            size = max(size, movie_size)
        camera = None
        if size > 1e-6:
            duration = size / self.recall_speed
            if items:
                duration = max(duration, max(it[3] for it in items) / self.recall_speed)
            camera = {"user": user_plan, "movie": movie_plan, "duration": duration}
        if not items and camera is None:
            return
        self._anim = {"items": items, "camera": camera, "t0": time.perf_counter()}
        self._driven = {it[0] for it in items}
        print(f"Moving {len(items)} chain ends" + (" and the view" if camera else "")
              + f" to slot {slot}")

    def animate(self):
        """Advance the recall animation: every chain end moves along a great circle at
        a constant angular speed (`recall_speed`, rad/s) and stops at its stored position;
        the view rotates and zooms to its stored state."""
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
        if idxs and self.on_ends_driven is not None:
            self.on_ends_driven(list(idxs), np.array(pos))

        cam = self._anim["camera"]
        if cam is not None:
            f = min(1.0, elapsed / cam["duration"])
            self._apply_view_plan(cam["user"], f)
            if self.recording and self._movie_camera is not None:
                if cam["movie"] is None:          # recording started while the view was moving
                    cam["movie"], _ = self._plan_view(self._get_view(self._movie_camera),
                                                      cam["user"]["v1"])
                    cam["movie"]["f0"] = f
                self._apply_view_plan(cam["movie"], f, cam=self._movie_camera)
            if f < 1.0:
                finished = False
        if finished:
            self._cancel_animation()

    def _cancel_camera_animation(self):
        """The user took over the camera: stop rotating/zooming the view (chain ends go on)."""
        if self._anim is None or self._anim["camera"] is None:
            return
        self._anim["camera"] = None
        if not self._anim["items"]:
            self._cancel_animation()

    def _on_view_interaction(self, obj, event):
        self._cancel_camera_animation()

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
            xmin xmax
            ymin ymax
            zmin zmax
            ITEM: ATOMS id mol type xu yu zu
            id molecule beadtype x y z          (one line per bead)
            # volume_radius radius              (extra information, always prefixed with '#')

        Ids and molecule numbers start at 1 (molecules are renumbered 1..M in file order);
        beadtype is 1 for chain ends and 2 for internal beads; the box bounds are the
        smallest and largest bead coordinates."""
        path = os.path.join(os.getcwd(), f"conf{self.conf_counter:05d}.dump")
        # molecule numbers 1..M, in order of appearance (moleculeids may have gaps after deletions)
        order = {int(m): k + 1 for k, m in enumerate(dict.fromkeys(self.moleculeids.tolist()))}
        if len(self.positions):
            lo, hi = self.positions.min(axis=0), self.positions.max(axis=0)
        else:
            lo = hi = np.zeros(3)
        with open(path, "w") as f:
            f.write(f"# {len(self.positions)} {len(order)}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(self.positions)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for k in range(3):
                f.write(f"{lo[k]:.6f} {hi[k]:.6f}\n")
            f.write("ITEM: ATOMS id mol type xu yu zu\n")
            for k, (xyz, mol, is_end) in enumerate(
                    zip(self.positions, self.moleculeids, self.surface_mask), start=1):
                beadtype = 1 if is_end else 2
                f.write(f"{k} {order[int(mol)]} {beadtype} "
                        f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
            f.write(f"# volume_radius {self.radius:.6f}\n")
        print("Saved", path)
        self.conf_counter += 1

    def _start_movie(self):
        """Start recording. The movie is written with imageio directly (not through
        PyVista's open_movie / mwriter attribute, which newer PyVista versions don't let
        us delete and re-create between recordings)."""
        if self.recording:
            print("Already recording", self.movie_path)
            return
        path = os.path.join(os.getcwd(), f"movie{self.movie_counter:05d}.avi")
        try:
            from imageio import get_writer
            self._movie_writer = get_writer(path, fps=self.movie_fps, quality=8, codec="mpeg4")
            # the movie gets its own camera, starting as a copy of the user's camera
            self._movie_camera = vtk.vtkCamera()
            self._movie_camera.DeepCopy(self.renderer.GetActiveCamera())
        except Exception as exc:
            self._movie_writer = None
            self._movie_camera = None
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
        self._movie_camera = None
        writer, self._movie_writer = self._movie_writer, None   # never written to again
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print("Problem while closing the movie file:", exc)
        print("Saved", self.movie_path)

    @staticmethod
    def _same_view(a, b):
        return all(np.allclose(np.array(x(a), dtype=float), np.array(x(b), dtype=float), atol=1e-9)
                   for x in (lambda c: c.GetPosition(), lambda c: c.GetFocalPoint(),
                             lambda c: c.GetViewUp(), lambda c: [c.GetViewAngle()],
                             lambda c: [c.GetParallelScale()]))

    def _grab_movie_frame(self):
        """RGB array of the window as seen by the movie camera. If the user has rotated,
        zoomed or panned the view since the recording started, the window is rendered once
        more with the movie camera into the back buffer only (swapping buffers is switched
        off, so the user never sees that frame) and then given back to the user's camera."""
        user_camera = self.renderer.GetActiveCamera()
        if self._same_view(user_camera, self._movie_camera):
            return self.plotter.image
        window = self.plotter.render_window
        self.renderer.SetActiveCamera(self._movie_camera)
        self.renderer.ResetCameraClippingRange()
        window.SwapBuffersOff()
        try:
            return self.plotter.image
        finally:
            window.SwapBuffersOn()
            self.renderer.SetActiveCamera(user_camera)
            self.renderer.ResetCameraClippingRange()

    def write_movie_frame(self):
        if not self.recording:
            return
        try:
            self._movie_writer.append_data(self._grab_movie_frame())
        except Exception as exc:
            print("Recording failed:", exc)
            self._stop_movie()

    def close(self):
        """Call after the window was closed: finishes a movie that is still being recorded."""
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
        if picked is not None:
            for i, actor in enumerate(self.actors):
                if picked is actor and self.surface_mask[i]:  # chain ends only
                    self._cancel_animation()                  # grabbing stops a running recall
                    self.dragged = i
                    self._side = 1 if self._is_front(self.positions[i]) else -1
                    self._flip_armed = True
                    actor.GetProperty().SetColor(*ACTIVE_COLOR)
                    self._cancel_camera_motion()
                    self.plotter.render()
                    return
        self._cancel_camera_animation()   # clicked elsewhere: the user takes over the camera

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
    def _digit_slot(key, shift):
        """Map a key press to (slot 1..9, store?) or None. Handles the plain digit
        (with the Shift modifier reported separately) and the layout-specific symbol
        produced by Shift + digit."""
        if len(key) == 1 and key in "123456789":
            return int(key), shift
        if key.startswith("KP_") and len(key) == 4 and key[3] in "123456789":
            return int(key[3]), shift
        symbols = SHIFTED_DIGIT_KEYSYMS.get(KEYBOARD_LAYOUT, ())
        if key in symbols:
            return symbols.index(key) + 1, True
        return None

    def _on_key(self, obj, event):
        key = self.interactor.GetKeySym() or ""
        shift = bool(self.interactor.GetShiftKey())
        if len(key) == 1 and key.isalpha() and shift:
            key = key.upper()
        lower = key.lower()
        digit = self._digit_slot(key, shift)
        char = self.interactor.GetKeyCode()
        if isinstance(char, int):
            char = chr(char)

        if lower == "h" or key == "question" or char == "?":
            print(HELP_TEXT)
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
        elif key == "c":
            self._set_axes_visible(True)
        elif key == "C":
            self._set_axes_visible(False)
        elif key == "b":
            self._set_internal_visible(True)
        elif key == "B":
            self._set_internal_visible(False)
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
        elif key == "D":
            self._set_paused(True)
        elif key == "d":
            self._set_paused(False)
        elif self.dragged is not None and lower in ("k", "right", "left"):
            if lower == "k":
                self._delete_molecule()
            elif lower == "right":
                self._insert_bead()
            else:
                self._remove_bead()
        elif shift and key.islower() and len(key) > 1 and "_" not in key:
            print(f"Shift+{key}: not recognised as a shifted digit for KEYBOARD_LAYOUT="
                  f"'{KEYBOARD_LAYOUT}' (see SHIFTED_DIGIT_KEYSYMS at the top of the file)")


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


def make_molecules(internal_counts, center, radius, rng):
    """Build M chains. Chain m has internal_counts[m] beads inside the volume and
    one bead at each end on the surface. Counts may repeat.

    Returns positions (n, 3), moleculeids (n,) uint32, surface_mask (n,) bool.
    Beads are ordered molecule by molecule as [start, internal..., end]."""
    counts = [int(c) for c in internal_counts]
    m_count = len(counts)
    anchors = make_surface_points(2 * m_count, center, radius)
    anchors = anchors[rng.permutation(2 * m_count)]
    starts, ends = anchors[:m_count], anchors[m_count:]

    positions, moleculeids, surface_mask = [], [], []
    for m, (n, a, b) in enumerate(zip(counts, starts, ends)):
        t = np.linspace(0.0, 1.0, n + 2)[:, None]
        chain = a + t * (b - a)                       # chord from start to end
        axis = (b - a) / np.linalg.norm(b - a)
        perp = rng.normal(size=3)
        perp -= np.dot(perp, axis) * axis
        perp /= np.linalg.norm(perp)
        chain = chain + np.sin(np.pi * t) * 0.35 * radius * perp   # bend the chain
        # keep internal beads strictly inside the volume
        v = chain[1:-1] - center
        r = np.linalg.norm(v, axis=1, keepdims=True)
        chain[1:-1] = center + v * np.minimum(1.0, 0.85 * radius / np.maximum(r, 1e-12))

        positions.append(chain)
        moleculeids += [m] * (n + 2)
        surface_mask += [True] + [False] * n + [True]

    return (np.vstack(positions),
            np.array(moleculeids, dtype=np.uint32),
            np.array(surface_mask, dtype=bool))


def load_dump(path, center=(0.0, 0.0, 0.0)):
    """Read a dump file written with the 's' key (any LAMMPS-style dump with the columns
    id, mol and x y z / xu yu zu works). Lines starting with '#' are comments, except
    "# volume_radius <r>" which gives the radius of the spherical volume.

    Returns positions (n, 3), moleculeids (n,) uint32, surface_mask (n,) bool, volume_radius.
    Beads are ordered molecule by molecule (by id within a molecule); the first and last
    bead of every molecule are its chain ends. If the file has no volume radius, the
    distance of the farthest chain end from `center` is used."""
    center = np.asarray(center, dtype=float)
    with open(path) as f:
        lines = [ln.strip() for ln in f]

    volume_radius, columns, rows = None, None, []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            parts = line[1:].split()
            if len(parts) == 2 and parts[0] == "volume_radius":
                volume_radius = float(parts[1])
            i += 1
        elif line.startswith("ITEM:"):
            item = line[len("ITEM:"):].strip()
            if item.startswith("ATOMS"):
                columns = item.split()[1:]
                i += 1
                while i < len(lines) and lines[i] and not lines[i].startswith(("#", "ITEM:")):
                    rows.append(lines[i].split())
                    i += 1
            elif item.startswith("BOX BOUNDS"):
                i += 4                      # header + three bound lines
            elif item.startswith("NUMBER OF ATOMS") or item.startswith("TIMESTEP"):
                i += 2                      # header + one value line
            else:
                i += 1
        else:
            i += 1
    if columns is None or not rows:
        raise ValueError(f"{path}: no 'ITEM: ATOMS ...' section with beads found")

    def column(*names):
        for name in names:
            if name in columns:
                return columns.index(name)
        raise ValueError(f"{path}: column {'/'.join(names)} missing in 'ITEM: ATOMS {' '.join(columns)}'")

    c_id, c_mol = column("id"), column("mol")
    c_x, c_y, c_z = column("xu", "x"), column("yu", "y"), column("zu", "z")
    table = np.array(rows, dtype=float)
    ids, mol = table[:, c_id].astype(int), table[:, c_mol].astype(int)
    xyz = table[:, [c_x, c_y, c_z]]

    # molecule by molecule, in order of first appearance (by id); by id within a molecule
    order = np.argsort(ids, kind="stable")
    first_seen = {}
    for m in mol[order]:
        first_seen.setdefault(int(m), len(first_seen))
    rank = np.array([first_seen[int(m)] for m in mol[order]])
    order = order[np.argsort(rank, kind="stable")]
    xyz, mol = xyz[order], mol[order]

    surface_mask = np.zeros(len(mol), dtype=bool)
    start = 0
    for end in list(np.flatnonzero(np.diff(mol) != 0) + 1) + [len(mol)]:
        if end - start < 2:
            raise ValueError(f"{path}: molecule {mol[start]} has fewer than two beads")
        surface_mask[start] = surface_mask[end - 1] = True
        start = end
    if "type" in columns:
        bead_type = table[order, columns.index("type")].astype(int)
        if not np.array_equal(bead_type == 1, surface_mask):
            print(f"Warning: bead types in {path} don't match the chain ends "
                  "(first and last bead of each molecule); using the molecule structure.")

    if volume_radius is None:
        volume_radius = float(np.linalg.norm(xyz[surface_mask] - center, axis=1).max())
    return xyz, (mol - mol.min()).astype(np.uint32), surface_mask, volume_radius


# --------------------------------------------------------------------------
# Stand-in for the simulation. Replace with your LAMMPS wrapper (see below).
# --------------------------------------------------------------------------
class DummySimulation:
    """Toy dynamics, no randomness:

    1. Chain contraction. Each internal bead relaxes toward the midpoint of its two
       chain neighbours (gradient descent on the sum of squared bond lengths). With
       the chain ends held on the surface this converges to a straight, evenly
       spaced chain, i.e. the shortest possible molecule.
    2. Excluded volume between molecules. Beads of DIFFERENT molecules are pushed
       apart until their centres are at least `2 * bead_radius` apart (position-based
       constraint, so it holds regardless of how strongly the chains contract).
       Beads of the SAME molecule may overlap freely.

    Internal beads stay inside the volume; surface beads are moved only tangentially
    (they are re-projected onto the sphere). Surface beads that the viewer is moving
    (grabbed by the user, or travelling to a stored position) are immovable for the
    simulation, so they push other beads out of the way instead."""

    def __init__(self, positions, moleculeids, surface_mask, center, radius, bead_radius,
                 chain_rate=0.5, substeps=5):
        self.x = np.array(positions, dtype=float).reshape(-1, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.surface_mask = np.array(surface_mask, dtype=bool)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.bead_radius = float(bead_radius)
        self.min_distance = 2.0 * self.bead_radius  # centres closer than this = overlap
        self.chain_rate = chain_rate    # fraction of the way toward the neighbour midpoint
        self.substeps = substeps        # dynamics substeps per viewer update
        self.held = set()               # beads currently controlled by the viewer

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

    def insert_bead(self, index, xyz, moleculeid):
        """The user inserted an internal bead at `index` (new ordering)."""
        self.x = np.insert(self.x, index, xyz, axis=0)
        self.moleculeids = np.insert(self.moleculeids, index, moleculeid)
        self.surface_mask = np.insert(self.surface_mask, index, False)
        self.held = {h + 1 if h >= index else h for h in self.held}

    def remove_bead(self, index):
        """The user removed the internal bead at `index` (old ordering)."""
        self.x = np.delete(self.x, index, axis=0)
        self.moleculeids = np.delete(self.moleculeids, index)
        self.surface_mask = np.delete(self.surface_mask, index)
        self.held = {h - 1 if h > index else h for h in self.held if h != index}

    def add_molecule(self, first_index, positions, moleculeid):
        """The user created a molecule; its beads [start, internal..., end] are appended."""
        assert first_index == len(self.x)
        k = len(positions)
        self.x = np.vstack([self.x, positions])
        self.moleculeids = np.concatenate(
            [self.moleculeids, np.full(k, moleculeid, dtype=np.uint32)])
        self.surface_mask = np.concatenate(
            [self.surface_mask, [True] + [False] * (k - 2) + [True]])

    def delete_molecule(self, first_index, count):
        """The user deleted the molecule occupying beads [first_index, first_index + count)."""
        block = slice(first_index, first_index + count)
        self.x = np.delete(self.x, block, axis=0)
        self.moleculeids = np.delete(self.moleculeids, block)
        self.surface_mask = np.delete(self.surface_mask, block)
        self.held = {h - count if h >= first_index + count else h
                     for h in self.held if not (first_index <= h < first_index + count)}

    # --- dynamics ---------------------------------------------------------------
    def contour_length(self):
        """Total length of all chains (sum of bond lengths)."""
        same = self.moleculeids[:-1] == self.moleculeids[1:]
        return float(np.linalg.norm(np.diff(self.x, axis=0), axis=1)[same].sum())

    def _contract_chains(self):
        idx = np.flatnonzero(~self.surface_mask)  # internal beads always have 2 neighbours
        mid = 0.5 * (self.x[idx - 1] + self.x[idx + 1])
        self.x[idx] += self.chain_rate * (mid - self.x[idx])

    def _apply_constraints(self):
        v = self.x - self.center
        r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        on_surface = self.surface_mask[:, None]
        limit = self.radius - self.bead_radius  # keep internal beads inside the shell
        scale = np.where(on_surface, self.radius / r, np.minimum(1.0, limit / r))
        self.x = self.center + v * scale

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
        d_min = self.min_distance

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

    def step(self):
        """Advance the 'simulation' and return an (n, 3) array of positions."""
        for _ in range(self.substeps):
            self._contract_chains()
            self._apply_constraints()
            self._separate_molecules()
        return self.x

    def min_intermolecular_distance(self):
        """Smallest distance between beads of different molecules (for checking)."""
        diff = self.x[:, None, :] - self.x[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        other = self.moleculeids[:, None] != self.moleculeids[None, :]
        return float(dist[other].min()) if other.any() else np.inf


# Example of the same interface on top of the LAMMPS Python API.
# Insertion/removal/creation/deletion change the atom count and shift viewer indices, so
# keep a list `self.ids` (viewer index -> LAMMPS atom id) instead of assuming index == id - 1.
# In LAMMPS the physical ingredients of the dummy dynamics are, for example,
#   bond_style harmonic        (chain contraction)
#   pair_style lj/cut 2.24     with pair_modify shift yes, applied only between
#                              different molecule ids (excluded within a molecule via
#                              special_bonds / neigh_modify exclude molecule/intra all)
#   region sph sphere 0 0 0 ${R} side in / fix wall/region ...  (the spherical volume)
#
#   import ctypes
#   from lammps import lammps
#   class LammpsSimulation:
#       def __init__(self, input_file):
#           self.lmp = lammps()
#           self.lmp.file(input_file)
#           self.ids = list(range(1, self.lmp.get_natoms() + 1))  # molecule-by-molecule order
#       def set_surface_position(self, index, xyz):
#           ...  # move atom self.ids[index] and zero its velocity, or drive a fix
#       def release(self, index, xyz):
#           ...  # let the atom move again
#       def set_driven_positions(self, indices, positions):
#           ...  # same as set_surface_position, for many atoms at once
#       def release_driven(self, indices):
#           ...
#       def set_radius(self, radius):
#           # lmp.command(f"variable R equal {radius}") if the wall region uses ${R}, and
#           # project the surface atoms onto the new sphere
#           ...
#       def insert_bead(self, index, xyz, moleculeid):
#           # lmp.command(f"create_atoms 1 single {xyz[0]} {xyz[1]} {xyz[2]}"), set its
#           # molecule id, replace the bond (end - old neighbour) by (end - new) and
#           # (new - old neighbour), then self.ids.insert(index, new_atom_id)
#           ...
#       def remove_bead(self, index):
#           # lmp.command(f"group rm id {self.ids[index]}"), delete_atoms group rm ...,
#           # create a bond between the two former neighbours, then self.ids.pop(index)
#           ...
#       def add_molecule(self, first_index, positions, moleculeid):
#           # create_atoms for every row of `positions`, give them molecule id `moleculeid`,
#           # create bonds between consecutive atoms, then self.ids.extend(new atom ids)
#           ...
#       def delete_molecule(self, first_index, count):
#           # lmp.command(f"group rm molecule {moleculeid}"), delete_atoms group rm bond yes
#           # molecule yes, then del self.ids[first_index:first_index + count]
#           ...
#       def step(self):
#           self.lmp.command("run 10 pre no post no")
#           x = np.array(self.lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)  # ordered by atom id
#           return x[np.array(self.ids) - 1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive viewer for bead-spring molecules in a spherical volume.",
        epilog="Mouse actions and key shortcuts (press h or ? in the window to print them again)\n"
               "\n" + HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", metavar="DUMP_FILE",
                        help="restore molecules, positions and volume radius from a dump "
                             "file written with the 's' key")
    args = parser.parse_args(argv)

    print("Press h or ? in the window for all key shortcuts and mouse actions.")
    rng = np.random.default_rng(42)

    # ---- geometry ----
    center = np.zeros(3)
    sphere_radius = 1.0     # radius of each bead
    if args.load:
        points, moleculeids, surface_mask, volume_radius = load_dump(args.load, center)
        print(f"Loaded {len(points)} beads in {len(np.unique(moleculeids))} molecules "
              f"from {args.load} (volume radius {volume_radius:g})")
    else:
        volume_radius = 15.0    # radius of the spherical volume (r / R change it at run time)
        internal_counts = [24, 40]    # two molecules initially; press 'a' to add more
        # positions: (n, 3); moleculeids: (n,) uint32; surface_mask: True for chain ends
        points, moleculeids, surface_mask = make_molecules(
            internal_counts, center, volume_radius, rng)

    # ---- scene ----
    plotter = pv.Plotter(window_size=(1008, 800))
    plotter.set_background("white")
    axes = AxesGizmo(plotter)       # coordinate system, flush with the lower-left corner

    template = pv.Sphere(radius=sphere_radius, theta_resolution=24, phi_resolution=24)
    actors = []
    for p, mid in zip(points, moleculeids):
        actor = plotter.add_mesh(template, color=color_for_id(mid), smooth_shading=True)
        actor.SetPosition(*p)
        actors.append(actor)

    # ---- simulation + viewer wiring ----
    sim = DummySimulation(points, moleculeids, surface_mask, center, volume_radius,
                          bead_radius=sphere_radius)
    # sim = LammpsSimulation("in.sim")

    def on_release(i, xyz):
        print(f"Chain end {i} (moleculeid={dragger.moleculeids[i]}) released at",
              np.round(xyz, 4))
        sim.release(i, xyz)

    dragger = SphereDragger(
        plotter, template, actors, points, moleculeids, surface_mask, center, volume_radius,
        sphere_radius=sphere_radius, rng=rng, axes=axes,
        on_drag=sim.set_surface_position,
        on_release=on_release,
        on_insert=sim.insert_bead,
        on_remove=sim.remove_bead,
        on_add_molecule=sim.add_molecule,
        on_delete_molecule=sim.delete_molecule,
        on_radius_change=sim.set_radius,
        on_ends_driven=sim.set_driven_positions,
        on_ends_released=sim.release_driven,
    )

    set_initial_view(plotter, center, volume_radius, sphere_radius)   # sphere fills the window

    plotter.add_timer_event(max_steps=10**9, duration=30,
                            callback=lambda step: dragger.tick(sim.step))   # ~30 ms

    plotter.show()
    dragger.close()
    return dragger.get_positions(), dragger.moleculeids


if __name__ == "__main__":
    main()
