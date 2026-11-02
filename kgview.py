"""
Interactive 3D viewer for linear bead-spring molecules (polymer chains), meant for dump
files with millions of beads.

Every molecule is   start bead - internal beads - end bead.  Beads are spheres colored by
molecule; lines of the same color connect consecutive beads along each chain.

Only the beads within a distance `volume_radius` from a center position (xc, yc, zc) are
shown (the translucent shell marks this volume: a sphere to begin with; v / V switch to a cube
or a thin slab, see below), and only those parts of molecules
that have at least `min_length` beads in a row inside it. A molecule may cross the sphere in
many segments (maximal runs of consecutive beads inside); a segment is shown only if it has at
least `min_length` beads. All other beads are kept in memory but are not drawn. A chain line is drawn only when both of its
beads are shown (and the bond is not longer than MAX_BOND_LENGTH). Chain ends that lie on the
surface of the volume are "attached" to it; these are the chain ends that Shift+1..9 / 1..9
store and recall.

Mouse
- Left-drag on empty space  -> rotate the view
- Right-drag / mouse wheel  -> zoom
- Middle-drag (or Shift+left-drag) -> pan

Keyboard
- Arrow keys -> shift the center (xc, yc, zc) of the shown volume by
        10% of the current volume radius in the direction of the arrow (screen right/left/up/down), which pans the
        view the same way: the camera moves along with the center, so the sphere stays in
        place on the screen and the beads inside it change. Chain ends are attached to the
        surface only if they lie on it around the NEW center.
- Shift + Up / Shift + Down -> the same, but the center moves into /
        out of the image, i.e. along the viewing direction.
- w  -> decrease the display scale of the beads (sphere_scale) by 10%; W -> increase it by 10%.
        The scale only changes how large the beads are drawn, not the bead
        radius used by r / R and the simulation. It starts at 0.5 (--scale changes that).
- l  -> increase min_length by one; L -> decrease it by one (not below 1). Only segments of
        molecules with at least min_length beads inside the volume are shown (default 1, i.e.
        every bead inside; --min-length changes the start value).
- d  -> switch depth cueing on; D -> switch it off. With depth cueing the beads and
        chain lines fade toward the white background with their depth in the volume: the
        front of the sphere keeps its color, the back is paler. (--depth-cueing starts with
        it on.)
- E  -> pause the dynamics: the simulation is no longer stepped, so bead positions are not
        updated by it (you can still shift the center, recall views, ...)
- e  -> start/resume the dynamics. THE DYNAMICS ARE OFF WHEN THE PROGRAM STARTS.
- r  -> decrease the volume radius by 10% and increase the bead radius by 10% (refused when
        the new volume radius would be smaller than 5). The view is zoomed so that the
        spherical volume keeps its size in the window.
- R  -> increase the volume radius by 10% and decrease the bead radius by 10% (the volume
        keeps its size in the window). Bead positions are not rescaled.
- Shift + 1..9 -> store the current positions of all surface-attached chain ends, the
                  rotation and zoom of the view, and the center and the radius of the
                  volume, in slot 1..9
                  (1..9 are the nine keys of the number row, whatever characters they type:
                  Shift + the key stores, the key alone recalls. On the French layout (--keyboard-layout
                  fr) these are the keys typing & é " ' ( - è _ ç, and the digits they type with Shift.)
- 1..9         -> slowly move all chain ends stored in that slot to their stored positions,
                  each along a great circle of the sphere, while the center and the radius
                  of the volume move to their stored values (a translation: the camera
                  follows the center) and the view rotates and zooms to the stored state
                  (touching the mouse takes the view back). While a movie is being recorded
                  the motion advances by exactly one movie frame per picture, however slow
                  the drawing is, so the movie plays at the intended speed.
- v  -> switch the shape of the shown volume, around the same center: sphere (radius
        volume_radius) -> cube (side length 2*volume_radius) -> thin slab (two long sides of
        length 2*volume_radius, the thin side 0.5*volume_radius, along the z axis) -> sphere.
        V -> the same in the opposite order. The cube and the slab are aligned with the
        coordinate axes. Chain ends are attached to the surface (and stored / recalled by
        Shift+1..9 / 1..9) only while the volume is a sphere.
- c  -> show the coordinate system (axes widget); C -> hide it
- B  -> hide the internal beads (chain lines and chain ends stay visible); b -> show them again
- m  -> start recording a movie (movie00000.avi, movie00001.avi, ...); M -> stop recording.
        The movie keeps the view the camera had when recording started: rotating, zooming or
        panning the view yourself does NOT show up in the movie, but a view recalled with
        1..9 does (the movie view then moves from where it was to the stored view).
- h or ?  -> print a summary of the mouse and key shortcuts in the terminal
- p  -> save a PNG of the current view (img00000.png, img00001.png, ...)
- s  -> save the beads that are shown, i.e. those inside the volume that belong to segments
        with at least min_length beads (the other beads are not written), to conf00000.dump,
        conf00001.dump, ... in a LAMMPS-dump-like format:
        # <number of beads> <number of segments>
        ITEM: NUMBER OF ATOMS / <number of beads>
        ITEM: BOX BOUNDS pp pp pp / xmin xmax / ymin ymax / zmin zmax  (extent of the beads)
        ITEM: ATOMS id mol type xu yu zu / one line per bead, ending in a comment
          <id> <mol> <type> <x> <y> <z> # <original_bead_id> <original_molecule_number> <index_on_molecule>
        # volume_radius <radius>   (last lines; extra information is always prefixed with '#')
        # center <xc> <yc> <zc>
        # shape <sphere|cube|slab>
        Every segment is saved as one molecule (numbered 1..S, ids start at 1). Type 2 = surface
        bead: the preceding or the following bead of the original molecule is outside the sphere;
        type 1 = internal bead: all its neighbours along the molecule are inside (a chain end
        has only one neighbour). original_bead_id and original_molecule_number are the id and
        molecule number in the input file; index_on_molecule counts the beads along the original
        molecule from 1 (the first bead of the molecule).
All files are written to the current folder; numbering continues after existing files.

Command line
    python beadppa.py --load conf00000.dump
    python beadppa.py --load conf00000.dump --center XC YC ZC --volume-radius R

    --load           dump file (e.g. written with 's', or any LAMMPS-style dump with the columns
                     id, mol, x y z / xu yu zu; only the first frame is read). A file name
                     ending in .gz is read as a gzip-compressed dump.
    --center         center (xc, yc, zc) of the shown volume (default 0 0 0)
    --scale          display scale of the beads, i.e. the drawn bead radius is this factor times
                     the bead radius (default 0.5; w / W change it while running)
    --min-length     show only segments of molecules with at least this many beads inside the
                     volume (default 1; l / L change it while running)
    --depth-cueing   start with depth cueing switched on ('d' / 'D' switch it on / off)
    --keyboard-layout  layout of your keyboard, for the slot keys Shift+1..9 / 1..9: us, uk, de,
                     nordic or fr (default nordic); see KEYBOARD_LAYOUTS to add your own
    --nowrap         do not wrap coordinates into the bounding box of the dump header
                     (the 6 numbers after "ITEM: BOX BOUNDS pp pp pp"); by default beads
                     outside it are wrapped back in (periodic boundary conditions)
    --volume-radius  radius of the shown volume (default: the "# volume_radius" line of the
                     dump file; without it the distance of the farthest chain end from the center)

Data flow
    simulation (LAMMPS)  --positions-->  viewer      (timer, every ~30 ms; only while the
                                                      dynamics are switched on)
    viewer (user input)  --callbacks-->  simulation  (see SphereDragger.__init__)

Bead ordering: molecule by molecule, each as [start, internal..., end], bead id ascending
within a molecule (the lines of the dump file may be in any order). The viewer keeps ALL
beads of the file, shown or not, in this order, and the simulation must return positions for
all of them. Beads are never inserted or removed, so the indices stay fixed.

Install:  pip install pyvista numpy scipy      (scipy: only for the toy DummySimulation)
Movies:   pip install imageio imageio-ffmpeg
"""
import argparse
import gzip
import os
import re
import time

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

AXES_SIZE = 0.8         # coordinate system widget: size as a fraction of the window
# ... placed at the bottom-left edge of the window (fully inside it)
AXES_VIEWPORT = (0.0, 0.0, AXES_SIZE, AXES_SIZE)
INITIAL_ZOOM = 1.6      # the first view shows everything this many times larger than "fit to window"
AXES_LABEL_REFERENCE_SIZE = 0.4   # widget size at which the X/Y/Z labels have the size we keep

MIN_LENGTH = 1          # default: show only segments with at least this many beads inside
# Level of detail of the spheres: with very many beads the finest sphere would hand the GPU
# billions of triangles (this crashed the graphics driver with 4 million beads), so the sphere
# template gets coarser until (beads x triangles per bead) fits into this budget.
MAX_TRIANGLES = 30_000_000
SPHERE_LEVELS = ((16, 16), (10, 10), (6, 6), (4, 4))    # (theta, phi) resolution, finest first
MAX_TUBE_BONDS = 200_000    # with more chain lines than this they are drawn as plain lines
DEPTH_CUE_STRENGTH = 0.9    # depth cueing: the farthest beads are blended this much toward white
SPHERE_SCALE = 0.5      # default display scale of the beads (--scale changes it)
SCALE_STEP = 0.10       # w / W change the display scale by this fraction
CENTER_STEP_FRACTION = 0.10     # an arrow key shifts the center by this fraction of the volume radius
AUTOFOCUS_BEADS = 10000         # initial volume radius: just enough to show this many beads
MAX_BOND_LENGTH = 2.0   # bonds (chain lines) longer than this are not drawn

SHAPES = ("sphere", "cube", "slab")     # v / V cycle through these shapes of the shown volume
SLAB_THICKNESS = 0.5    # slab: thin side = this times the volume radius (long sides 2 * radius)

MIN_VOLUME_RADIUS = 5.0
VOLUME_SCALE_STEP = 0.10    # r / R change the volume radius by this fraction, the bead radius oppositely

# Relative tolerances (fractions of the volume radius):
WINDOW_TOLERANCE = 1e-6     # a bead exactly on the surface still counts as inside
SURFACE_TOLERANCE = 1e-3    # a chain end this close to the surface counts as attached to it

# Keyboard layouts. The slot keys 1..9 are the nine keys of the number row, and which
# CHARACTER such a key types depends on the layout (and on Shift). The viewer works with the
# typed character, which VTK reports on every platform, so a layout is just a table: for each
# layout the characters typed WITHOUT Shift and WITH Shift by the keys of slots 1..9.
# Pressing the key recalls slot n, Shift + the key stores slot n. The French AZERTY layout
# types the symbols without Shift and the digits with Shift. Add your own layout here, or
# select one with --keyboard-layout (default: KEYBOARD_LAYOUT).
KEYBOARD_LAYOUTS = {
    #           unshifted          shifted
    "us":     ("123456789", "!@#$%^&*("),
    "uk":     ("123456789", '!"\u00a3$%^&*('),
    "de":     ("123456789", '!"\u00a7$%&/()'),
    "nordic": ("123456789", '!"#\u00a4%&/()'),
    "fr":     ('&\u00e9"\'(-\u00e8_\u00e7', "123456789"),
}
KEYBOARD_LAYOUT = "nordic"

# X11-style key names that VTK may report instead of a character (some platforms report a
# key name for symbols and leave the typed character empty); used as a fallback.
KEYSYM_TO_CHAR = {
    "exclam": "!", "at": "@", "numbersign": "#", "dollar": "$", "percent": "%",
    "asciicircum": "^", "ampersand": "&", "asterisk": "*", "parenleft": "(", "parenright": ")",
    "quotedbl": '"', "apostrophe": "'", "quoteright": "'", "minus": "-", "underscore": "_",
    "slash": "/", "sterling": "\u00a3", "section": "\u00a7", "currency": "\u00a4",
    "eacute": "\u00e9", "egrave": "\u00e8", "ccedilla": "\u00e7", "agrave": "\u00e0",
}


def colors_for_ids(ids):
    """Deterministic, well-separated RGB colors (k, 3), values 0..1, for an array of
    unsigned integer ids (golden-ratio hue spacing, HSV with s=0.65, v=0.95)."""
    s, v = 0.65, 0.95
    h6 = ((np.asarray(ids, dtype=np.float64) * 0.618033988749895) % 1.0) * 6.0
    sector = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = np.full_like(f, v * (1.0 - s))
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    vv = np.full_like(f, v)
    conds = [sector == k for k in range(6)]
    r = np.select(conds, [vv, q, p, p, t, vv])
    g = np.select(conds, [t, vv, vv, q, p, p])
    b = np.select(conds, [p, p, t, vv, vv, q])
    return np.column_stack([r, g, b])


def color_for_id(moleculeid):
    """Color (r, g, b) in 0..1 for one molecule id."""
    return tuple(float(c) for c in colors_for_ids([int(moleculeid)])[0])


def chain_ends(moleculeids):
    """(first, last) boolean masks: first / last bead of every molecule. Molecules are
    stored as contiguous blocks of beads."""
    n = len(moleculeids)
    first = np.ones(n, dtype=bool)
    last = np.ones(n, dtype=bool)
    if n > 1:
        change = moleculeids[1:] != moleculeids[:-1]
        first[1:] = change
        last[:-1] = change
    return first, last


def segment_ids(inside, first):
    """Segments = maximal runs of consecutive beads of ONE molecule that are inside the
    sphere. `inside` (n,) bool, `first` (n,) bool marks the first bead of every molecule.
    Returns (segment number of every bead (meaningful where inside), number of segments)."""
    prev_inside = np.zeros(len(inside), dtype=bool)
    prev_inside[1:] = inside[:-1] & ~first[1:]
    start = inside & ~prev_inside
    return np.cumsum(start) - 1, int(start.sum())


def shown_mask(inside, first, min_length):
    """Beads inside the sphere that belong to a segment with at least `min_length` beads."""
    if min_length <= 1 or not inside.any():
        return inside
    seg, n_seg = segment_ids(inside, first)
    lengths = np.bincount(seg[inside], minlength=n_seg)
    return inside & (lengths[np.maximum(seg, 0)] >= min_length)


def set_initial_view(plotter, zoom=INITIAL_ZOOM):
    """Fit the scene to the window, then zoom in `zoom` times (the sphere appears `zoom`
    times as large). Marking the camera as set stops PyVista from resetting it at show()."""
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    plotter.camera_position = plotter.camera_position


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


def _points_view(mesh):
    """Zero-copy numpy view of a mesh's point coordinates (writing to it changes the mesh)."""
    return vtk_to_numpy(mesh.GetPoints().GetData())


def _touch_points(mesh):
    """Tell VTK that the points of `mesh` were changed in place."""
    mesh.GetPoints().GetData().Modified()
    mesh.GetPoints().Modified()
    mesh.Modified()


def _locate(sorted_indices, wanted):
    """Positions of `wanted` indices inside the sorted index array `sorted_indices`.
    Returns (k, ok): k[ok] are valid positions, ok marks the wanted indices that are present."""
    wanted = np.asarray(wanted, dtype=np.intp)
    if len(sorted_indices) == 0:
        return np.zeros(len(wanted), dtype=np.intp), np.zeros(len(wanted), dtype=bool)
    k = np.minimum(np.searchsorted(sorted_indices, wanted), len(sorted_indices) - 1)
    return k, sorted_indices[k] == wanted


class SphereDragger:
    """Shows the beads inside a spherical volume (in segments with enough beads),
    lets the user shift the volume, resize it, store/recall chain-end configurations together
    with the view and the volume, and save images / movies / bead positions; lets a
    simulation push new positions for all beads.

    All beads (millions are fine) live in numpy arrays. Only those within `radius` of
    `center` are rendered: the spheres with one instanced glyph mapper (not one actor per
    bead) and the chain lines as a single mesh."""

    IMAGE_PATTERN = re.compile(r"^img(\d{5})\.png$")
    CONF_PATTERN = re.compile(r"^conf(\d{5})\.dump$")
    MOVIE_PATTERN = re.compile(r"^movie(\d{5})\.avi$")
    # characters whose built-in VTK meaning (reset camera, surface mode, pick, ...) is suppressed
    # ('a'/'A' have no function here any more, but VTK's own 'a' = actor mode stays disabled)
    OWN_CHARS = frozenset("aAbBcCdDeEhHlLpPsSmMrRvVwW123456789")

    def __init__(self, plotter, template, positions, moleculeids, surface_mask,
                 center, radius, sphere_radius, recall_speed=0.5, movie_fps=30,
                 show_internal=True, show_axes=True, paused=True, sphere_scale=SPHERE_SCALE,
                 min_length=MIN_LENGTH, depth_cueing=False,
                 original_ids=None, original_molecules=None, keyboard_layout=KEYBOARD_LAYOUT,
                 on_radius_change=None, on_center_change=None,
                 on_ends_driven=None, on_ends_released=None):
        self.plotter = plotter
        self.renderer = plotter.renderer
        self.template = template        # sphere mesh used as the shape of every bead
        self.positions = np.array(positions, dtype=float).reshape(-1, 3)   # (n, 3), ALL beads
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.surface_mask = np.array(surface_mask, dtype=bool)   # chain ends attached to the surface
        n_beads = len(self.positions)
        # id / molecule number of every bead in the input file (written by 's')
        self.original_ids = (np.arange(1, n_beads + 1) if original_ids is None
                             else np.asarray(original_ids, dtype=np.int64))
        self.original_molecules = (self.moleculeids.astype(np.int64) + 1 if original_molecules is None
                                   else np.asarray(original_molecules, dtype=np.int64))
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)             # radius of the spherical volume
        self.sphere_radius = float(sphere_radius)   # radius of one bead
        if keyboard_layout not in KEYBOARD_LAYOUTS:
            raise ValueError(f"unknown keyboard layout '{keyboard_layout}'; "
                             f"choose one of {', '.join(KEYBOARD_LAYOUTS)}")
        self.keyboard_layout = keyboard_layout
        self.depth_cueing = bool(depth_cueing)      # fade distant beads / bonds toward the background
        self._cue_dirty = True
        self._cue_key = None            # viewing direction the colors were last computed for
        self._sphere_base = None        # undimmed uint8 colors of the shown spheres
        self._bond_base = None          # undimmed uint8 colors of the drawn bonds
        self.min_length = max(1, int(min_length))   # a segment needs at least this many beads
        self.sphere_scale = float(sphere_scale)     # drawn bead radius = sphere_scale * sphere_radius
        self.recall_speed = float(recall_speed)     # rad/s along great circles
        self.movie_fps = movie_fps
        self.image_counter = self._first_free_number(self.IMAGE_PATTERN)
        self.conf_counter = self._first_free_number(self.CONF_PATTERN)
        self.movie_counter = self._first_free_number(self.MOVIE_PATTERN)

        # Callbacks into the simulation (all indices refer to the full bead list):
        self.on_radius_change = on_radius_change      # (new_volume_radius, new_bead_radius)
        self.on_center_change = on_center_change      # (new_center, surface_mask (n,) bool)
        self.on_ends_driven = on_ends_driven          # (indices, positions) chain ends moved by a recall
        self.on_ends_released = on_ends_released      # (indices)            recall finished/cancelled

        self.interactor = plotter.iren.interactor
        self.stored = {}                # slot (1..9) -> {"ends": {(moleculeid, is_start): unit vector},
        #                                   "view": ..., "center": xyz, "radius": volume radius}
        self._anim = None               # running recall animation
        self._driven = np.zeros(0, dtype=np.intp)   # indices of chain ends moved by the animation
        self.recording = False
        self.movie_path = None
        self._movie_writer = None       # imageio writer while recording
        self._movie_camera = None       # camera used for the movie frames while recording
        self._in_tick = False           # re-entrancy guard for tick()
        self._need_render = False
        self.paused = bool(paused)      # True: the simulation is not stepped
        self.show_internal = bool(show_internal)    # are the internal beads drawn?
        self.show_axes = bool(show_axes)            # is the coordinate system drawn?
        self.bond_mesh = None
        self._sphere_mesh = None        # point cloud of the shown beads (glyph input)
        self._vis_mask = np.zeros(0, dtype=bool)    # bead is inside the volume
        self._vis = np.zeros(0, dtype=np.intp)      # indices of the beads inside the volume
        self._shown = np.zeros(0, dtype=np.intp)    # indices of the beads drawn as spheres
        self._bond_seg = np.zeros(0, dtype=np.intp) # drawn bonds: segment a joins beads a, a+1
        self.shape = "sphere"           # shape of the shown volume, one of SHAPES
        self._update_structure()

        # Translucent shell showing the volume (unit sphere, scaled to the current radius).
        self.shell = plotter.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64),
            color="lightgray", opacity=0.12, smooth_shading=True)
        self.shell.SetPickable(False)
        # the same for the cube and the slab (a unit cube, stretched to the shape)
        self.box_shell = plotter.add_mesh(
            pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0),
            color="lightgray", opacity=0.12)
        self.box_shell.SetPickable(False)
        self._update_shell()

        # All spheres are drawn by ONE actor: a glyph mapper instancing the template at every
        # point of a point cloud, colored per point (uint8 RGB point data "rgb").
        self._mapper = vtk.vtkGlyph3DMapper()
        self._mapper.SetSourceData(template)
        self._template_radius = self.sphere_radius   # radius of the template sphere
        self._templates = {SPHERE_LEVELS[0]: template}   # level of detail -> sphere mesh
        self._lod = SPHERE_LEVELS[0]
        self._mapper.SetScaling(True)                # bead radius = template radius * scale factor
        self._mapper.SetScaleModeToNoDataScaling()
        self._mapper.SetScaleFactor(self.sphere_scale)
        self._mapper.SetOrient(False)
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.SetColorModeToDirectScalars()
        self._mapper.ScalarVisibilityOn()
        self._glyph_actor = vtk.vtkActor()
        self._glyph_actor.SetMapper(self._mapper)
        plotter.add_actor(self._glyph_actor, name="beads", reset_camera=False, pickable=False)
        self._rebuild_scene()

        # Make sure PyVista's own key shortcuts for the keys we use don't also fire.
        for key in ("Up", "Down", "Left", "Right", "a", "b", "B", "c", "C", "d", "D", "p", "s",
                    "S", "m", "M", "r", "R", "w", "W", "l", "L", "e", "E", "h", "v", "V", *"123456789"):
            try:
                plotter.clear_events_for_key(key)
            except Exception:
                pass

        # Priority -1 => our handlers run AFTER the camera style's handlers.
        self.interactor.AddObserver("KeyPressEvent", self._on_key, -1.0)
        # Touching the camera with the mouse stops a running view animation.
        for ev in ("LeftButtonPressEvent", "RightButtonPressEvent", "MiddleButtonPressEvent",
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
        step (`step_simulation()` returns the (n, 3) positions of all beads; skipped while the
        dynamics are paused, which is the default), draw if something changed, record a frame.

        Re-entrancy guard: grabbing a movie frame renders the window again, and VTK may
        deliver the next timer event while that is still going on. Without the guard each
        nested timer event would start another tick (and grab another frame) inside the
        previous one until Python's recursion limit is hit. A tick that arrives while one
        is still running is simply skipped."""
        if self._in_tick:
            return
        self._in_tick = True
        try:
            animating = self._anim is not None
            self.animate()
            self._update_cueing()           # the view may have been rotated by the user
            if self.paused:                 # dynamics paused: only draw what the user / a
                if self._need_render or animating:      # recall animation has changed
                    self._need_render = False
                    self.plotter.render()
            else:
                self.set_positions(step_simulation())
            self.write_movie_frame()
        finally:
            self._in_tick = False

    # ------------------------------------------------------- simulation -> view
    def set_positions(self, positions):
        """Update the positions of all beads (e.g. from LAMMPS). Chain ends moved by a recall
        animation keep their viewer-controlled positions."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if len(positions) != len(self.positions):
            raise ValueError(f"Simulation returned {len(positions)} beads, "
                             f"viewer has {len(self.positions)}.")
        held = self._driven
        kept = self.positions[held].copy() if len(held) else None
        np.copyto(self.positions, positions)
        if kept is not None:
            self.positions[held] = kept
        self._refresh_scene()
        self._update_cueing()
        self._need_render = False
        self.plotter.render()

    def get_positions(self):
        return self.positions.copy()

    # ------------------------------------------------------------ structure
    def _update_structure(self):
        """Needed whenever beads were inserted/removed (molecules are contiguous blocks)."""
        self.first_mask, self.last_mask = chain_ends(self.moleculeids)
        self.end_mask = self.first_mask | self.last_mask
        self.mol_index = np.cumsum(self.first_mask) - 1     # bead -> molecule number 0..M-1

    def _molecule_blocks(self):
        """(first, last): bead index of the first / last bead of every molecule, in order."""
        return np.flatnonzero(self.first_mask), np.flatnonzero(self.last_mask)

    # ------------------------------------------------------ scene (rendering)
    def _update_shell(self):
        """Position, size and shape of the translucent shell marking the volume."""
        sphere = self.shape == "sphere"
        self.shell.SetVisibility(sphere)
        self.box_shell.SetVisibility(not sphere)
        r = self.radius
        z = r * (0.5 * SLAB_THICKNESS if self.shape == "slab" else 1.0)
        self.shell.SetPosition(*self.center)
        self.box_shell.SetPosition(*self.center)
        self.shell.SetScale(r, r, r)
        self.box_shell.SetScale(r, r, z)

    def _cycle_shape(self, step):
        """v (step=+1) / V (step=-1): sphere -> cube -> slab -> sphere, or backwards."""
        self._cancel_animation()
        self.shape = SHAPES[(SHAPES.index(self.shape) + step) % len(SHAPES)]
        self._update_shell()
        self._attach_ends()             # (surface attachment only exists for the sphere)
        print(f"Shape of the volume: {self.shape}; showing {len(self._vis)} beads")
        self.plotter.render()

    def _visible_mask(self):
        """Beads inside the volume (sphere, cube or slab around `center`, see `shape`) that belong
        to a segment (a run of consecutive beads of one molecule inside) with at least
        `min_length` beads."""
        v = self.positions - self.center
        limit = self.radius * (1.0 + WINDOW_TOLERANCE) + 1e-9
        if self.shape == "sphere":
            inside = np.einsum("ij,ij->i", v, v) <= limit * limit
        else:                           # half side lengths: radius (cube), thin side/2 (slab)
            half = limit * np.array([1.0, 1.0, 0.5 * SLAB_THICKNESS if self.shape == "slab" else 1.0])
            inside = np.all(np.abs(v) <= half, axis=1)
        return shown_mask(inside, self.first_mask, self.min_length)

    def _rebuild_scene(self, mask=None):
        """Needed when the shown set of beads, the number of beads or the radius changed."""
        self._vis_mask = self._visible_mask() if mask is None else mask
        self._vis = np.flatnonzero(self._vis_mask)
        self._update_shown()
        self._rebuild_bonds()
        self._cue_dirty = True
        self._update_cueing()

    def _refresh_scene(self):
        """Positions of (possibly) all beads changed: rebuild if beads entered or left the
        volume, otherwise just move the points."""
        mask = self._visible_mask()
        if np.array_equal(mask, self._vis_mask):
            if not np.array_equal(self._bond_segments(), self._bond_seg):
                self._rebuild_bonds()           # a bond became shorter/longer than the limit
            self._refresh_points()
        else:
            self._rebuild_scene(mask)

    def _refresh_points(self):
        self._cue_dirty = True              # positions changed: depths changed
        if self._sphere_mesh is not None:
            _points_view(self._sphere_mesh)[:] = self.positions[self._shown]
            _touch_points(self._sphere_mesh)
        if self.bond_mesh is not None:
            _points_view(self.bond_mesh)[:] = self.positions[self._vis]
            _touch_points(self.bond_mesh)

    def _update_moved(self, idx):
        """Cheap update when only the beads `idx` moved (and stayed inside the volume)."""
        idx = np.asarray(idx, dtype=np.intp)
        if len(idx) == 0:
            return
        if self._sphere_mesh is not None:
            k, ok = _locate(self._shown, idx)
            if ok.any():
                _points_view(self._sphere_mesh)[k[ok]] = self.positions[idx[ok]]
                _touch_points(self._sphere_mesh)
        if self._bonds_changed_near(idx):
            self._rebuild_bonds()
        elif self.bond_mesh is not None:
            k, ok = _locate(self._vis, idx)
            if ok.any():
                _points_view(self.bond_mesh)[k[ok]] = self.positions[idx[ok]]
                _touch_points(self.bond_mesh)
        self._need_render = True

    def _update_shown(self):
        """Which of the beads inside the volume are drawn as spheres."""
        vis = self._vis
        self._shown = vis if self.show_internal else vis[self.end_mask[vis]]
        self._rebuild_spheres()

    def _sphere_level(self, n):
        """Finest sphere resolution whose n instances fit into MAX_TRIANGLES."""
        for res in SPHERE_LEVELS:
            if res not in self._templates:
                self._templates[res] = pv.Sphere(radius=self._template_radius,
                                                 theta_resolution=res[0], phi_resolution=res[1])
            if self._templates[res].GetNumberOfCells() * n <= MAX_TRIANGLES:
                return res
        return SPHERE_LEVELS[-1]

    def _rebuild_spheres(self):
        shown = self._shown
        if len(shown) == 0:
            self._sphere_mesh = None
            self._glyph_actor.SetVisibility(False)
            return
        mesh = pv.PolyData(np.array(self.positions[shown], dtype=float))
        rgb = (colors_for_ids(self.moleculeids[shown]) * 255.0).round().astype(np.uint8)
        self._sphere_base = rgb.copy()      # (the VTK array may share memory with `rgb`)
        mesh.point_data["rgb"] = rgb
        mesh.GetPointData().SetActiveScalars("rgb")
        self._sphere_mesh = mesh
        level = self._sphere_level(len(shown))
        if level != self._lod:
            self._lod = level
            self._mapper.SetSourceData(self._templates[level])
            print(f"Drawing {len(shown)} beads with {self._templates[level].GetNumberOfCells()} "
                  f"triangles each (level of detail {level[0]}x{level[1]})")
        self._mapper.SetInputData(mesh)
        self._glyph_actor.SetVisibility(True)

    def _bond_segments(self, a=None):
        """Indices a of the bonds (a, a+1) that are drawn: both beads shown, same molecule,
        and not longer than MAX_BOND_LENGTH. `a` restricts the check to those candidates."""
        vm, mol, x = self._vis_mask, self.moleculeids, self.positions
        if a is None:
            a = np.flatnonzero((mol[:-1] == mol[1:]) & vm[:-1] & vm[1:])
        else:
            a = a[(a >= 0) & (a < len(mol) - 1)]
            a = a[(mol[a] == mol[a + 1]) & vm[a] & vm[a + 1]]
        d = x[a + 1] - x[a]
        return a[np.einsum("ij,ij->i", d, d) <= MAX_BOND_LENGTH ** 2]

    def _bonds_changed_near(self, idx):
        """Did a bond next to one of the moved beads `idx` start/stop being drawn?"""
        cand = np.unique(np.concatenate([idx - 1, idx]))
        cand = cand[(cand >= 0) & (cand < len(self.moleculeids) - 1)]
        want = self._bond_segments(cand)
        k, ok = _locate(self._bond_seg, cand)
        have = cand[ok]
        return not np.array_equal(want, have)

    def _rebuild_bonds(self):
        """Chain lines between consecutive shown beads of the same molecule (bonds longer
        than MAX_BOND_LENGTH are skipped)."""
        vm, mol = self._vis_mask, self.moleculeids
        a = self._bond_segments()
        self._bond_seg = a
        if len(a) == 0:
            self.bond_mesh = None
            self.plotter.remove_actor("bonds", reset_camera=False)
            return
        local = np.cumsum(vm) - 1                       # global bead index -> index among shown beads
        lines = np.column_stack([np.full(len(a), 2), local[a], local[a + 1]]).ravel().astype(np.int64)
        self.bond_mesh = pv.PolyData(np.array(self.positions[self._vis], dtype=float), lines=lines)
        rgb = (colors_for_ids(mol[a]) * 255.0).round().astype(np.uint8)
        self._bond_base = rgb.copy()
        self.bond_mesh.cell_data["rgb"] = rgb
        # same `name` => pyvista replaces the previous line actor
        actor = self.plotter.add_mesh(
            self.bond_mesh, scalars="rgb", rgb=True, line_width=4,
            render_lines_as_tubes=len(a) <= MAX_TUBE_BONDS, name="bonds", reset_camera=False)
        actor.SetPickable(False)

    # ---------------------------------------------------------------- geometry helper
    def _unit_rows(self, p):
        v = np.asarray(p, dtype=float) - self.center
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    # ---------------------------------------------------- misc. display switches
    def _set_paused(self, paused):
        self.paused = bool(paused)
        print("Dynamics", "off (paused)" if self.paused else "on")

    def _set_axes_visible(self, visible):
        self.show_axes = bool(visible)
        if self.show_axes:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        print("Coordinate system", "shown" if self.show_axes else "hidden")
        self.plotter.render()

    def _apply_depth_cueing(self):
        """Recolor the spheres and chain lines: blend each toward white by its depth along the
        viewing direction within the volume (0 at the front of the sphere, DEPTH_CUE_STRENGTH
        at the back). With depth cueing off the original colors are put back."""
        self._cue_dirty = False
        d = np.array(self.renderer.GetActiveCamera().GetDirectionOfProjection())
        self._cue_key = tuple(np.round(d, 4))

        def faded(base, points):
            if not self.depth_cueing:
                return base
            t = np.clip(((points - self.center) @ d + self.radius) / (2.0 * self.radius), 0.0, 1.0)
            t = (t * DEPTH_CUE_STRENGTH).astype(np.float32)[:, None]
            return (base * (1.0 - t) + 255.0 * t).round().astype(np.uint8)

        if self._sphere_mesh is not None and self._sphere_base is not None:
            array = self._sphere_mesh.GetPointData().GetArray("rgb")
            vtk_to_numpy(array)[:] = faded(self._sphere_base, self.positions[self._shown])
            array.Modified()
            self._sphere_mesh.Modified()
        if self.bond_mesh is not None and self._bond_base is not None:
            a = self._bond_seg
            mid = 0.5 * (self.positions[a] + self.positions[a + 1])
            array = self.bond_mesh.GetCellData().GetArray("rgb")
            vtk_to_numpy(array)[:] = faded(self._bond_base, mid)
            array.Modified()
            self.bond_mesh.Modified()
        self._need_render = True

    def _update_cueing(self):
        """Recolor if depth cueing is on and the view or the positions changed."""
        if not self.depth_cueing:
            return
        key = tuple(np.round(self.renderer.GetActiveCamera().GetDirectionOfProjection(), 4))
        if self._cue_dirty or key != self._cue_key:
            self._apply_depth_cueing()

    def _set_depth_cueing(self, on):
        self.depth_cueing = bool(on)
        self._apply_depth_cueing()
        print("Depth cueing", "on" if self.depth_cueing else "off")
        self._need_render = False
        self.plotter.render()

    def _set_internal_visible(self, visible):
        self.show_internal = bool(visible)
        self._update_shown()
        print("Internal beads", "shown" if visible else "hidden")
        self.plotter.render()

    # ------------------------------------------------------------ volume radius
    def _change_min_length(self, delta):
        """l / L: change the number of beads a segment needs inside the volume to be shown."""
        new = self.min_length + delta
        if new < 1:
            print("min_length is already 1.")
            return
        self.min_length = new
        self._rebuild_scene()
        print(f"min_length: {new}; showing {len(self._vis)} beads")
        self.plotter.render()

    def _update_bead_size(self):
        """Apply sphere_radius and sphere_scale to the drawn spheres."""
        self._mapper.SetScaleFactor(self.sphere_scale * self.sphere_radius / self._template_radius)

    def _scale_spheres(self, grow):
        """w / W: change the display scale of the beads by 10%."""
        self.sphere_scale *= (1.0 + SCALE_STEP) if grow else (1.0 - SCALE_STEP)
        self._update_bead_size()
        print(f"Bead display scale: {self.sphere_scale:g}")
        self.plotter.render()

    def _scale_volume(self, shrink):
        """r (shrink=True): volume radius -10%, bead radius +10%; R: volume radius +10%, bead
        radius -10%. The cameras are scaled about the center of the volume by the same factor
        as the volume radius, so the volume keeps its size (and place) in the window."""
        f = VOLUME_SCALE_STEP
        factor, bead_factor = (1.0 - f, 1.0 + f) if shrink else (1.0 + f, 1.0 - f)
        new = self.radius * factor
        if new < MIN_VOLUME_RADIUS:
            print(f"Volume radius would become {new:g}, below the minimum ({MIN_VOLUME_RADIUS:g}).")
            return
        self._cancel_animation()
        self.radius = new
        self.sphere_radius *= bead_factor
        self._update_shell()
        self._update_bead_size()

        # keep the apparent size of the volume: scale the cameras about its center
        for cam in (self.renderer.GetActiveCamera(), self._movie_camera):
            if cam is not None:
                cam.SetPosition(*(self.center + factor * (np.array(cam.GetPosition()) - self.center)))
                cam.SetFocalPoint(*(self.center + factor * (np.array(cam.GetFocalPoint()) - self.center)))
                cam.SetParallelScale(cam.GetParallelScale() * factor)
        self.renderer.ResetCameraClippingRange()

        # all surface-attached chain ends stay on the (new) surface: project them radially
        idx = np.flatnonzero(self.surface_mask)
        if len(idx):
            self.positions[idx] = self.center + new * self._unit_rows(self.positions[idx])
        if self.on_radius_change is not None:
            self.on_radius_change(new, self.sphere_radius)
        self._rebuild_scene()           # other beads may have entered / left the smaller or larger volume
        print(f"Volume radius: {new:g}, bead radius: {self.sphere_radius:g}; "
              f"showing {len(self._vis)} beads")
        self.plotter.render()

    # ------------------------------------------------------------ shift the center
    def _attach_ends(self):
        """The center or the radius of the volume changed: the chain ends that lie on the surface
        of the volume around the new center are attached to it. Rebuilds the scene."""
        if self.shape == "sphere":
            dist = np.linalg.norm(self.positions - self.center, axis=1)
            self.surface_mask = self.end_mask & (
                np.abs(dist - self.radius) <= SURFACE_TOLERANCE * self.radius)
        else:
            self.surface_mask = np.zeros(len(self.positions), dtype=bool)
        if self.on_center_change is not None:
            self.on_center_change(self.center.copy(), self.surface_mask.copy())
        self._rebuild_scene()

    def _pan_center(self, arrow, depth=False):
        """Arrow key: move the center of the shown volume by 10% of the volume radius along the screen
        direction of the arrow, and the camera with it (= panning the view). With depth=True
        (Shift + Up / Down) the center moves into (Up) or out of (Down) the image instead."""
        frame = self._get_view()["frame"]               # columns: screen right, up, towards camera
        if depth:
            direction = -frame[:, 2] if arrow == "up" else frame[:, 2]
        else:
            direction = {"right": frame[:, 0], "left": -frame[:, 0],
                         "up": frame[:, 1], "down": -frame[:, 1]}[arrow]
        delta = CENTER_STEP_FRACTION * self.radius * direction
        self._cancel_animation()
        self.center = self.center + delta
        self._update_shell()
        for cam in (self.renderer.GetActiveCamera(), self._movie_camera):
            if cam is not None:
                cam.SetPosition(*(np.array(cam.GetPosition()) + delta))
                cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + delta))
        self.renderer.ResetCameraClippingRange()
        self._attach_ends()
        print("Center of the volume: ({:g}, {:g}, {:g}); showing {} beads".format(
            *self.center, len(self._vis)))
        self.plotter.render()

    # --------------------------------------------- store / recall chain-end positions
    def _end_keys(self, attached_only=True):
        """{(moleculeid, is_start): bead index} for every surface-attached chain end (or, with
        attached_only=False, for every chain end). This identifies a chain end independently
        of its index."""
        idx = np.flatnonzero(self.surface_mask if attached_only else self.end_mask)
        return {(m, s): i for i, m, s in zip(idx.tolist(), self.moleculeids[idx].tolist(),
                                             self.first_mask[idx].tolist())}

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
        idx = np.array(list(keys.values()), dtype=np.intp)
        units = self._unit_rows(self.positions[idx]) if len(idx) else np.zeros((0, 3))
        self.stored[slot] = {"ends": dict(zip(keys.keys(), units)),
                             "view": self._get_view(),
                             "center": self.center.copy(), "radius": self.radius}
        print(f"Stored {len(keys)} chain ends, the view, the center "
              f"({self.center[0]:g}, {self.center[1]:g}, {self.center[2]:g}) and the volume radius "
              f"{self.radius:g} in slot {slot}")

    def _recall_ends(self, slot):
        stored = self.stored.get(slot)
        if not stored:
            print(f"Nothing stored in slot {slot} (use Shift+{slot} to store)")
            return
        target = stored["ends"]
        self._cancel_animation()
        c0, R0 = self.center.copy(), self.radius
        c1, R1 = np.asarray(stored["center"], dtype=float), float(stored["radius"])

        keys = self._end_keys(attached_only=False) if self.shape == "sphere" else {}
        common = [k for k in keys if k in target]
        idx = np.array([keys[k] for k in common], dtype=np.intp)
        p0 = self.positions[idx]
        u0 = self._unit_rows(p0) if len(idx) else np.zeros((0, 3))
        r0 = np.linalg.norm(p0 - c0, axis=1)        # distance from the center now
        u1 = np.array([target[k] for k in common]).reshape(-1, 3)
        cosang = np.clip(np.einsum("ij,ij->i", u0, u1), -1.0, 1.0)
        omega = np.arccos(cosang)                   # great-circle angle between the two directions
        w = u1 - cosang[:, None] * u0               # direction of travel at u0 (tangent to the sphere)
        bad = np.linalg.norm(w, axis=1) < 1e-9      # exactly opposite or equal: any great circle will do
        if bad.any():
            alt = np.cross(u0[bad], [1.0, 0.0, 0.0])
            small = np.linalg.norm(alt, axis=1) < 1e-6
            alt[small] = np.cross(u0[bad][small], [0.0, 1.0, 0.0])
            w[bad] = alt
        if len(w):
            w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        items = (idx, u0, w, omega, u1, r0)
        volume_moves = bool(np.linalg.norm(c1 - c0) > 1e-9 or abs(R1 - R0) > 1e-9)

        # Everything (chain ends, center, radius, view) moves in the same time, at `recall_speed`:
        # angles in rad/s, translations and radius changes measured in units of the volume radius.
        # While recording, the movie camera makes the same move starting from its own view
        # (the user's own rotating/zooming doesn't move it).
        user_plan, size = self._plan_view(self._get_view(), stored["view"])
        movie_plan = None
        if self.recording and self._movie_camera is not None:
            movie_plan, movie_size = self._plan_view(self._get_view(self._movie_camera),
                                                     stored["view"])
            size = max(size, movie_size)
        sizes = [size, float(np.linalg.norm(c1 - c0)) / max(R0, 1e-12),
                 abs(float(np.log(max(R1, 1e-12) / max(R0, 1e-12))))]
        if len(idx):
            sizes += [float(omega.max()), float(np.abs(r0 - R1).max()) / max(R0, 1e-12)]
        duration = max(sizes) / self.recall_speed
        if duration < 1e-6:
            return                                  # already there
        camera = None
        if size > 1e-6 or volume_moves:
            camera = {"user": user_plan, "movie": movie_plan,
                      # the camera follows the center: keep its focal point at the same offset
                      "off_user": np.array(self.renderer.GetActiveCamera().GetFocalPoint()) - c0,
                      "off_movie": (np.array(self._movie_camera.GetFocalPoint()) - c0
                                    if movie_plan is not None else None)}
        self._anim = {"items": items, "camera": camera, "duration": duration,
                      "c0": c0, "c1": c1, "R0": R0, "R1": R1, "moves_volume": volume_moves,
                      "clock": 0.0, "tlast": time.perf_counter()}
        self._driven = idx
        print(f"Moving {len(idx)} chain ends" + (" and the view" if camera else "")
              + (" and the volume" if volume_moves else "") + f" to slot {slot}")

    def animate(self):
        """Advance the recall animation: every chain end moves along a great circle at
        a constant angular speed (`recall_speed`, rad/s) and stops at its stored position;
        the center and radius of the volume interpolate linearly to their stored values; the
        view rotates and zooms to its stored state. While recording, one call is exactly one
        movie frame of animation time, otherwise the animation follows the wall clock."""
        a = self._anim
        if a is None:
            return
        now = time.perf_counter()
        a["clock"] += (1.0 / self.movie_fps) if self.recording else (now - a["tlast"])
        a["tlast"] = now
        elapsed = a["clock"]
        f = min(1.0, elapsed / a["duration"])
        arc = self.recall_speed * elapsed

        if a["moves_volume"]:
            self.center = a["c0"] + f * (a["c1"] - a["c0"])
            self.radius = a["R0"] + f * (a["R1"] - a["R0"])
            self._update_shell()
            if self.on_radius_change is not None:
                self.on_radius_change(self.radius, self.sphere_radius)
            if self.on_center_change is not None:
                self.on_center_change(self.center.copy(), self.surface_mask.copy())
        idx, u0, w, omega, u1, r0 = a["items"]
        if len(idx):
            done = arc >= omega
            u = np.where(done[:, None], u1, u0 * np.cos(arc) + w * np.sin(arc))
            r = r0 + f * (a["R1"] - r0)             # distance from the center (= radius for attached ends)
            p = self.center + r[:, None] * u
            self.positions[idx] = p
            if self.on_ends_driven is not None:
                self.on_ends_driven(idx.copy(), p)
        if a["moves_volume"]:
            self._rebuild_scene()                   # other beads enter / leave the moving volume
            self._need_render = True
        elif len(idx):
            self._update_moved(idx)

        cam = a["camera"]
        if cam is not None:
            self.renderer.GetActiveCamera().SetFocalPoint(*(self.center + cam["off_user"]))
            self._apply_view_plan(cam["user"], f)
            if self.recording and self._movie_camera is not None:
                if cam["movie"] is None:          # recording started while the view was moving
                    cam["movie"], _ = self._plan_view(self._get_view(self._movie_camera),
                                                      cam["user"]["v1"])
                    cam["movie"]["f0"] = f
                    cam["off_movie"] = np.array(self._movie_camera.GetFocalPoint()) - self.center
                self._movie_camera.SetFocalPoint(*(self.center + cam["off_movie"]))
                self._apply_view_plan(cam["movie"], f, cam=self._movie_camera)
        if f >= 1.0:
            self._cancel_animation()

    def _cancel_camera_animation(self):
        """The user took over the camera: stop rotating/zooming/following (the rest goes on)."""
        if self._anim is None or self._anim["camera"] is None:
            return
        self._anim["camera"] = None
        if len(self._anim["items"][0]) == 0 and not self._anim["moves_volume"]:
            self._cancel_animation()

    def _on_view_interaction(self, obj, event):
        self._cancel_camera_animation()

    def _cancel_animation(self):
        """The animation is over or interrupted: release the chain ends. If the volume had been
        moving, the chain ends lying on its surface are attached to it again."""
        if self._anim is None:
            return
        released = self._driven
        moved = self._anim["moves_volume"]
        self._anim = None
        self._driven = np.zeros(0, dtype=np.intp)
        if moved:
            self._attach_ends()
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
        """Write confXXXXX.dump (LAMMPS-dump-like text) with the beads that are shown (inside
        the volume, in segments with at least min_length beads; the internal beads are written
        also while they are hidden with B):

            # n_beads n_segments
            ITEM: NUMBER OF ATOMS
            n_beads
            ITEM: BOX BOUNDS pp pp pp
            xmin xmax
            ymin ymax
            zmin zmax
            ITEM: ATOMS id mol type xu yu zu
            id segment beadtype x y z # original_bead_id original_molecule_number index_on_molecule
            # volume_radius radius              (extra information, always prefixed with '#')
            # center xc yc zc
            # shape sphere | cube | slab         (volume_radius: sphere radius / half side length)

        Ids and segment numbers start at 1 (every segment of a molecule is saved as a molecule
        of its own). beadtype is 2 for a surface bead (the preceding or the following bead of
        the original molecule is outside the sphere) and 1 for an internal bead (every
        neighbour along the molecule is inside; a chain end has just one neighbour).
        index_on_molecule is the position along the original molecule, the first bead being 1.
        The box bounds are the smallest and largest bead coordinates."""
        sel = self._vis                                 # sorted indices of the shown beads
        n = len(sel)
        if n == 0:
            print("No beads are shown: nothing to save.")
            return
        path = os.path.join(os.getcwd(), f"conf{self.conf_counter:05d}.dump")
        pos = self.positions[sel]
        lo, hi = pos.min(axis=0), pos.max(axis=0)

        inside, first = self._vis_mask, self.first_mask  # (a neighbour of a shown bead is inside
        previous_outside = np.zeros(len(inside), dtype=bool)     # exactly when it is shown)
        previous_outside[1:] = ~inside[:-1] & ~first[1:]         # a preceding bead exists and is outside
        next_outside = np.zeros(len(inside), dtype=bool)
        next_outside[:-1] = ~inside[1:] & ~first[1:]             # a following bead exists and is outside
        beadtype = np.where((previous_outside | next_outside)[sel], 2, 1)

        segment, _ = segment_ids(inside, first)
        segment_no = np.unique(segment[sel], return_inverse=True)[1].ravel() + 1
        index_on_molecule = np.arange(len(inside)) - np.flatnonzero(first)[self.mol_index] + 1

        table = np.column_stack([np.arange(1, n + 1), segment_no, beadtype, pos,
                                 self.original_ids[sel], self.original_molecules[sel],
                                 index_on_molecule[sel]])
        with open(path, "w") as f:
            f.write(f"# {n} {int(segment_no.max())}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for k in range(3):
                f.write(f"{lo[k]:.6f} {hi[k]:.6f}\n")
            f.write("ITEM: ATOMS id mol type xu yu zu\n")
            np.savetxt(f, table, fmt=["%d", "%d", "%d", "%.6f", "%.6f", "%.6f",
                                      "# %d", "%d", "%d"])
            f.write(f"# volume_radius {self.radius:.6f}\n")
            f.write(f"# center {self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}\n")
            f.write(f"# shape {self.shape}\n")
        print("Saved", path, f"({n} beads, {int(segment_no.max())} segments)")
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

    def _digit_slot(self, key, shift, char):
        """Map a key press to (slot 1..9, store?) or None. `key` is VTK's key name, `shift` the
        state of the Shift key and `char` the typed character ('' if none). The typed
        character is looked up in the table of the keyboard layout; where the platform
        reports only a key name, it is translated to a character first, and where it reports
        a plain digit together with the Shift modifier, that counts as Shift + the key."""
        if key.startswith("KP_") and len(key) == 4 and key[3] in "123456789":
            return int(key[3]), shift                       # numeric keypad: layout independent
        unshifted, shifted = KEYBOARD_LAYOUTS[self.keyboard_layout]
        ch = char or KEYSYM_TO_CHAR.get(key) or (key if len(key) == 1 else "")
        if ch and ch in shifted:
            return shifted.index(ch) + 1, True
        if ch and ch in unshifted:
            return unshifted.index(ch) + 1, shift
        return None

    def _on_key(self, obj, event):
        key = self.interactor.GetKeySym() or ""
        shift = bool(self.interactor.GetShiftKey())
        if len(key) == 1 and key.isalpha() and shift:
            key = key.upper()
        lower = key.lower()
        code = self.interactor.GetKeyCode()
        if isinstance(code, int):
            code = chr(code)
        if not code or not code.isprintable() or code.isspace():
            code = ""                                       # no typed character reported
        digit = self._digit_slot(key, shift, code)

        if key in ("question", "?", "h") or code in ("?", "h"):
            print("\n" + _controls_text() + "\n")
        elif key == "v":
            self._cycle_shape(+1)
        elif key == "V":
            self._cycle_shape(-1)
        elif lower == "p":
            self._save_image()
        elif lower == "s":
            self._save_configuration()
        elif key == "r":
            self._scale_volume(shrink=True)
        elif key == "R":
            self._scale_volume(shrink=False)
        elif key == "c":
            self._set_axes_visible(True)
        elif key == "C":
            self._set_axes_visible(False)
        elif key == "b":
            self._set_internal_visible(True)
        elif key == "B":
            self._set_internal_visible(False)
        elif key == "l":
            self._change_min_length(+1)
        elif key == "L":
            self._change_min_length(-1)
        elif key == "w":
            self._scale_spheres(grow=False)
        elif key == "W":
            self._scale_spheres(grow=True)
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
            self._set_depth_cueing(False)
        elif key == "d":
            self._set_depth_cueing(True)
        elif key == "E":
            self._set_paused(True)
        elif key == "e":
            self._set_paused(False)
        elif lower in ("up", "down", "left", "right"):
            self._pan_center(lower, depth=shift and lower in ("up", "down"))
        elif shift and key.islower() and len(key) > 1 and "_" not in key:
            print(f"Shift+{key} (character {code!r}) is not a slot key for the keyboard layout "
                  f"'{self.keyboard_layout}': use --keyboard-layout "
                  f"({', '.join(KEYBOARD_LAYOUTS)}) or add your layout to KEYBOARD_LAYOUTS")


# --------------------------------------------------------------------------
# Reading dump files
# --------------------------------------------------------------------------
def _open_text(path):
    """Text file object for a dump file; names ending in .gz are decompressed on the fly."""
    return gzip.open(path, "rt") if path.lower().endswith(".gz") else open(path)


def _open_binary(path):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")


def _volume_radius_from(line):
    """The radius of a "# volume_radius <r>" line, else None."""
    if line.startswith("#"):
        parts = line[1:].split()
        if len(parts) == 2 and parts[0] == "volume_radius":
            return float(parts[1])
    return None


def load_dump(path, center=(0.0, 0.0, 0.0), wrap=True):
    """Read a dump file written with the 's' key (any LAMMPS-style dump with the columns
    id, mol and x y z / xu yu zu works; only the first frame is read). Lines starting with
    '#' are comments, except "# volume_radius <r>" which gives the radius of the volume.

    Written for millions of beads: only the needed columns are parsed (numpy's C parser), and
    everything after that is vectorised.

    The bead type column is not used. Returns positions (n, 3), moleculeids (n,) uint32,
    end_mask (n,) bool, volume_radius, and the bead ids and molecule numbers of the file in the
    same order as the beads.
    Beads are ordered molecule by molecule (by id within a molecule); the first and last
    bead of every molecule are its chain ends. If the file has no volume radius, the
    distance of the farthest chain end from `center` is used.

    With `wrap=True` (default) the coordinates are wrapped into the bounding box given by
    the six numbers after "ITEM: BOX BOUNDS pp pp pp" (periodic boundary conditions): a
    bead outside [lo, hi] in a dimension is shifted by a multiple of the box length into it.
    Beads already inside the box (edges included) are left untouched. Note that the
    two beads of a bond may end up on opposite sides of the box."""
    center = np.asarray(center, dtype=float)
    volume_radius, columns, n_atoms, lines_read = None, None, None, 0
    box = None                                  # (3, 2): lo / hi of x, y, z
    with _open_text(path) as f:                 # header: a handful of lines
        while True:
            line = f.readline()
            if not line:
                break
            lines_read += 1
            line = line.strip()
            r = _volume_radius_from(line)
            if r is not None:
                volume_radius = r
            elif line.startswith("ITEM:"):
                item = line[len("ITEM:"):].strip()
                if item.startswith("ATOMS"):
                    columns = item.split()[1:]
                    break
                skip = 3 if item.startswith("BOX BOUNDS") else 1 if item.startswith(
                    ("NUMBER OF ATOMS", "TIMESTEP")) else 0
                values = []
                for _ in range(skip):
                    values.append(f.readline())
                    lines_read += 1
                if item.startswith("NUMBER OF ATOMS"):
                    n_atoms = int(values[0])
                elif item.startswith("BOX BOUNDS"):
                    # the 6 numbers: xlo xhi / ylo yhi / zlo zhi (a triclinic file has a
                    # third number per line, the tilt factor; it is ignored)
                    box = np.array([[float(v) for v in ln.split()[:2]] for ln in values])
    if columns is None or not n_atoms:
        raise ValueError(f"{path}: no 'ITEM: NUMBER OF ATOMS' / 'ITEM: ATOMS ...' sections found")

    with _open_binary(path) as f:               # the "# volume_radius" line is at the end
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4096))
        for line in f.read().decode("ascii", errors="ignore").splitlines():
            r = _volume_radius_from(line.strip())
            if r is not None:
                volume_radius = r

    def column(*names):
        for name in names:
            if name in columns:
                return columns.index(name)
        raise ValueError(f"{path}: column {'/'.join(names)} missing in 'ITEM: ATOMS {' '.join(columns)}'")

    wanted = [column("id"), column("mol"),
              column("xu", "x"), column("yu", "y"), column("zu", "z")]
    with _open_text(path) as f:
        table = np.loadtxt(f, skiprows=lines_read, max_rows=n_atoms, usecols=wanted, ndmin=2)
    if len(table) != n_atoms:
        raise ValueError(f"{path}: expected {n_atoms} beads, found {len(table)}")
    ids, mol, xyz = table[:, 0].astype(np.int64), table[:, 1].astype(np.int64), table[:, 2:5]
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: bead ids are not unique")
    if wrap:
        if box is None:
            print(f"Warning: {path} has no 'ITEM: BOX BOUNDS'; coordinates are not wrapped.")
        else:
            lo, length = box[:, 0], box[:, 1] - box[:, 0]
            if np.any(length <= 0.0):
                print(f"Warning: empty bounding box in {path}; coordinates are not wrapped.")
            else:
                outside = (xyz < lo) | (xyz > box[:, 1])
                xyz = np.where(outside, lo + np.mod(xyz - lo, length), xyz)
                if outside.any():
                    print(f"Wrapped {int(outside.any(axis=1).sum())} beads into the bounding box "
                          f"x {box[0]}, y {box[1]}, z {box[2]} (use --nowrap to disable)")

    # molecule by molecule, in order of first appearance (by id); by id within a molecule
    order = np.argsort(ids, kind="stable")
    uniq, first_pos, inverse = np.unique(mol[order], return_index=True, return_inverse=True)
    rank_of = np.empty(len(uniq), dtype=np.int64)
    rank_of[np.argsort(first_pos)] = np.arange(len(uniq))
    order = order[np.argsort(rank_of[inverse], kind="stable")]
    xyz, mol, ids = xyz[order], mol[order], ids[order]

    first, last = chain_ends(mol)
    if np.any(first & last):
        bad = mol[np.flatnonzero(first & last)[0]]
        raise ValueError(f"{path}: molecule {bad} has fewer than two beads")
    end_mask = first | last
    if volume_radius is None:
        volume_radius = float(np.linalg.norm(xyz[end_mask] - center, axis=1).max())
    return (np.ascontiguousarray(xyz), (mol - mol.min()).astype(np.uint32), end_mask,
            volume_radius, ids, mol)


def focus_radius(points, moleculeids, center, min_length, target=AUTOFOCUS_BEADS):
    """Smallest volume radius around `center` for which at least `target` beads are shown
    (inside the volume, in segments with at least `min_length` beads), found by
    bisection; None if even the largest possible volume shows fewer. Returns (radius, shown)."""
    first, _ = chain_ends(moleculeids)
    d = np.linalg.norm(points - center, axis=1)

    def shown(radius):
        inside = d <= radius * (1.0 + WINDOW_TOLERANCE) + 1e-9
        return int(np.count_nonzero(shown_mask(inside, first, min_length)))

    lo, hi = 0.0, float(d.max())
    if shown(hi) < target:
        return None
    for _ in range(60):                 # the count never decreases when the radius grows
        if hi - lo <= 1e-6 * hi:
            break
        mid = 0.5 * (lo + hi)
        if shown(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi, shown(hi)


def _controls_text():
    """The mouse / keyboard section of the module docstring, for --help."""
    doc = __doc__
    return ("Mouse and keyboard controls in the viewer window:\n\n"
            + doc[doc.index("Mouse\n"):doc.index("Command line")].rstrip())


# --------------------------------------------------------------------------
# Stand-in for the simulation. Replace with your LAMMPS wrapper (see below).
# --------------------------------------------------------------------------
class DummySimulation:
    """Toy dynamics, no randomness (needs scipy for the neighbour search):

    1. Chain contraction. Each internal bead relaxes toward the midpoint of its two
       chain neighbours (gradient descent on the sum of squared bond lengths). With
       the chain ends held this converges to a straight, evenly spaced chain, i.e.
       the shortest possible molecule.
    2. Excluded volume between molecules. Beads of DIFFERENT molecules are pushed
       apart until their centres are at least `2 * bead_radius` apart (position-based
       constraint, so it holds regardless of how strongly the chains contract).
       Beads of the SAME molecule may overlap freely.

    The toy model assumes the whole system lives inside the sphere: beads that are not
    attached to the surface are pulled inside the volume (radius - bead radius), chain ends
    attached to the surface are moved only tangentially (re-projected onto the sphere).
    Chain ends that the viewer is moving (travelling to a stored position) are immovable for
    the simulation, so they push other beads out of the way."""

    def __init__(self, positions, moleculeids, surface_mask, center, radius, bead_radius,
                 chain_rate=0.5, substeps=5):
        self.x = np.array(positions, dtype=float).reshape(-1, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.surface_mask = np.array(surface_mask, dtype=bool)
        first, last = chain_ends(self.moleculeids)
        self.end_mask = first | last
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.bead_radius = float(bead_radius)
        self.min_distance = 2.0 * self.bead_radius  # centres closer than this = overlap
        self.chain_rate = chain_rate    # fraction of the way toward the neighbour midpoint
        self.substeps = substeps        # dynamics substeps per viewer update
        self.held = np.zeros(len(self.x), dtype=bool)   # beads currently controlled by the viewer

    # --- called by the viewer -------------------------------------------------
    def set_driven_positions(self, indices, positions):
        """The viewer moves several chain ends (recall of a stored configuration)."""
        self.x[indices] = positions
        self.held[indices] = True

    def release_driven(self, indices):
        self.held[np.asarray(indices, dtype=np.intp)] = False

    def set_radius(self, radius, bead_radius):
        """The volume radius and the bead radius changed; surface-attached chain ends are
        projected onto the new surface."""
        self.radius = float(radius)
        self.bead_radius = float(bead_radius)
        self.min_distance = 2.0 * self.bead_radius
        self._apply_constraints()

    def set_center(self, center, surface_mask):
        """The user shifted the center of the volume; `surface_mask` says which chain ends
        are attached to the surface of the volume around the new center."""
        self.center = np.asarray(center, dtype=float)
        self.surface_mask = np.array(surface_mask, dtype=bool)

    def _update_structure(self):
        first, last = chain_ends(self.moleculeids)
        self.end_mask = first | last

    # --- dynamics ---------------------------------------------------------------
    def contour_length(self):
        """Total length of all chains (sum of bond lengths)."""
        same = self.moleculeids[:-1] == self.moleculeids[1:]
        return float(np.linalg.norm(np.diff(self.x, axis=0), axis=1)[same].sum())

    def _contract_chains(self):
        idx = np.flatnonzero(~self.end_mask)      # internal beads always have 2 neighbours
        mid = 0.5 * (self.x[idx - 1] + self.x[idx + 1])
        self.x[idx] += self.chain_rate * (mid - self.x[idx])

    def _apply_constraints(self):
        v = self.x - self.center
        r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        on_surface = self.surface_mask[:, None]
        limit = self.radius - self.bead_radius  # keep the other beads inside the shell
        scale = np.where(on_surface, self.radius / r, np.minimum(1.0, limit / r))
        self.x = self.center + v * scale

    def _separate_molecules(self, iterations=20, tol=1e-6):
        from scipy.spatial import cKDTree       # neighbour search: no n x n matrices
        n = len(self.x)
        if n < 2:
            return
        fixed = self.held
        d_min = self.min_distance

        for _ in range(iterations):
            pairs = cKDTree(self.x).query_pairs(d_min - tol, output_type="ndarray")
            if len(pairs) == 0:
                break
            i, j = pairs[:, 0], pairs[:, 1]     # i < j
            other = self.moleculeids[i] != self.moleculeids[j]
            i, j = i[other], j[other]
            if len(i) == 0:
                break
            diff = self.x[i] - self.x[j]
            dist = np.linalg.norm(diff, axis=1)
            direction = diff / np.maximum(dist, 1e-12)[:, None]
            tie = dist < 1e-9                   # coincident centres: pick opposite axes
            direction[tie] = 0.0
            direction[tie, 0] = 1.0
            overlap = d_min - dist
            # each bead of a pair takes half of the overlap; a held bead takes none,
            # so its partner takes all of it
            push_i = (overlap * np.where(fixed[j], 1.0, 0.5))[:, None] * direction
            push_j = -(overlap * np.where(fixed[i], 1.0, 0.5))[:, None] * direction
            push = np.zeros_like(self.x)
            for k in range(3):
                push[:, k] = (np.bincount(i, weights=push_i[:, k], minlength=n)
                              + np.bincount(j, weights=push_j[:, k], minlength=n))
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
        """Smallest distance between beads of different molecules (for checking); inf if
        there is no such pair closer than 2 * min_distance."""
        from scipy.spatial import cKDTree
        pairs = cKDTree(self.x).query_pairs(2.0 * self.min_distance, output_type="ndarray")
        if len(pairs) == 0:
            return np.inf
        i, j = pairs[:, 0], pairs[:, 1]
        other = self.moleculeids[i] != self.moleculeids[j]
        if not other.any():
            return np.inf
        return float(np.linalg.norm(self.x[i[other]] - self.x[j[other]], axis=1).min())


# Example of the same interface on top of the LAMMPS Python API.
# Keep a list `self.ids` (viewer index -> LAMMPS atom id) instead of assuming index == id - 1.
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
#       def set_driven_positions(self, indices, positions):
#           ...  # move these atoms (and zero their velocity, or drive a fix) to `positions`
#       def release_driven(self, indices):
#           ...
#       def set_radius(self, radius, bead_radius):
#           # lmp.command(f"variable R equal {radius}") if the wall region uses ${R}, set the
#           # new bead size (pair cutoff), and project the surface atoms onto the new sphere
#           ...
#       def set_center(self, center, surface_mask):
#           ...  # the volume moved: update the wall region / the atoms held on the surface
#       def step(self):
#           self.lmp.command("run 10 pre no post no")
#           x = np.array(self.lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)  # ordered by atom id
#           return x[np.array(self.ids) - 1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive viewer for bead-spring molecules: shows the beads of a dump "
                    "file that lie within a spherical volume. Dump files compressed with gzip "
                    "(file names ending in .gz) are read directly.",
        epilog=_controls_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", metavar="DUMP_FILE", required=True,
                        help="dump file with the molecules (e.g. written with the 's' key); "
                             "a name ending in .gz is read as a gzip-compressed file")
    parser.add_argument("--center", metavar=("XC", "YC", "ZC"), nargs=3, type=float,
                        default=[0.0, 0.0, 0.0],
                        help="center of the shown spherical volume (default: 0 0 0)")
    parser.add_argument("--volume-radius", type=float, default=None, metavar="R",
                        help="radius of the shown volume (default: chosen so that "
                             f"{AUTOFOCUS_BEADS} beads are shown, see --noautofocus; an explicit "
                             "value switches that off)")
    parser.add_argument("--scale", type=float, default=SPHERE_SCALE, metavar="NUMBER",
                        help="display scale of the beads: the drawn bead radius is this factor "
                             f"times the bead radius (default: {SPHERE_SCALE:g})")
    parser.add_argument("--min-length", type=int, default=MIN_LENGTH, metavar="N",
                        help="show only segments of molecules with at least N beads inside the volume "
                             f"(default: {MIN_LENGTH})")
    parser.add_argument("--depth-cueing", action="store_true",
                        help="start with depth cueing on (fade distant beads toward the "
                             "background; 'd' / 'D' switch it on / off)")
    parser.add_argument("--noautofocus", action="store_true",
                        help=f"do not choose the initial volume radius such that {AUTOFOCUS_BEADS} "
                             "beads are shown; use the radius stored in the dump file instead "
                             f"(files with fewer than {AUTOFOCUS_BEADS} beads are never changed)")
    parser.add_argument("--keyboard-layout", choices=sorted(KEYBOARD_LAYOUTS), default=KEYBOARD_LAYOUT,
                        help="layout of your keyboard, which decides what the number row types "
                             f"for the slot keys 1..9 and Shift+1..9 (default: {KEYBOARD_LAYOUT})")
    parser.add_argument("--nowrap", action="store_true",
                        help="do not wrap the coordinates into the bounding box given in the "
                             "dump file header (default: wrap, periodic boundary conditions)")
    args = parser.parse_args(argv)

    # ---- geometry ----
    center = np.array(args.center, dtype=float)     # (xc, yc, zc)
    sphere_radius = 1.0                             # radius of each bead
    # positions: (n, 3); moleculeids: (n,) uint32; end_mask: True for chain ends
    points, moleculeids, end_mask, file_radius, original_ids, original_molecules = load_dump(
        args.load, center, wrap=not args.nowrap)
    volume_radius = file_radius if args.volume_radius is None else args.volume_radius
    if not args.noautofocus and args.volume_radius is None and len(points) >= AUTOFOCUS_BEADS:
        focus = focus_radius(points, moleculeids, center, args.min_length)
        if focus is not None:
            volume_radius, n_focus = focus
            print(f"Auto focus: volume radius {volume_radius:g} (the dump file says {file_radius:g}) "
                  f"shows {n_focus} beads; use --noautofocus to keep the radius of the file")
    # chain ends lying on the surface of the volume are attached to it (stored / recalled)
    distance = np.linalg.norm(points - center, axis=1)
    surface_mask = end_mask & (np.abs(distance - volume_radius) <= SURFACE_TOLERANCE * volume_radius)
    print(f"Loaded {len(points)} beads in {len(np.unique(moleculeids))} molecules from {args.load}")
    print(f"Volume: radius {volume_radius:g} around ({center[0]:g}, {center[1]:g}, {center[2]:g}); "
          f"{int(surface_mask.sum())} chain ends are attached to the surface")
    print("Dynamics are off; press e to start them, E to pause them again.")
    del distance

    # ---- scene ----
    plotter = pv.Plotter(window_size=(1008, 800))
    plotter.set_background("white")
    plotter.add_axes()

    template = pv.Sphere(radius=sphere_radius, theta_resolution=16, phi_resolution=16)

    # ---- simulation + viewer wiring ----
    sim = DummySimulation(points, moleculeids, surface_mask, center, volume_radius,
                          bead_radius=sphere_radius)
    # sim = LammpsSimulation("in.sim")

    dragger = SphereDragger(
        plotter, template, points, moleculeids, surface_mask, center, volume_radius,
        sphere_radius=sphere_radius, paused=True, sphere_scale=args.scale,
        min_length=args.min_length, depth_cueing=args.depth_cueing,
        original_ids=original_ids, original_molecules=original_molecules,
        keyboard_layout=args.keyboard_layout,
        on_radius_change=sim.set_radius,
        on_center_change=sim.set_center,
        on_ends_driven=sim.set_driven_positions,
        on_ends_released=sim.release_driven,
    )
    del points
    print(f"Showing {len(dragger._vis)} beads in segments with at least "
          f"{dragger.min_length} beads inside the volume")

    set_initial_view(plotter)       # the sphere starts out larger than "fit to window"

    plotter.add_timer_event(max_steps=10**9, duration=30,
                            callback=lambda step: dragger.tick(sim.step))   # ~30 ms

    plotter.show()
    dragger.close()
    return dragger.get_positions(), dragger.moleculeids


if __name__ == "__main__":
    main()
