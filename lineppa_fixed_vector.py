"""
Interactive 3D viewer for polymers in a spherical volume, modeled as piecewise
linear curves WITHOUT internal beads.

Every molecule has two beads, a start bead and an end bead, both confined to the
surface of the sphere. Initially the molecule is the straight line (a chord through
the volume) between them. Beads are spheres colored by molecule; the line of a
molecule has the same color.

Intersections: when two lines of different molecules intersect, a junction is inserted
into both molecules at the intersection point, so each of them gets a bend there. A
junction is drawn as a small sphere with the average color of the two molecules. The
four line segments connected to a junction act as springs, with spring constant
spring_constant / contour_length of the segment (long segments are soft, short ones stiff);
their tension pulls the junction towards its four neighbours (chain ends or other junctions).
Forces: at every junction two colored arrows show, for each of the two molecules, the sum of
the two spring forces its two segments exert on the junction (in the color of that molecule),
and a black arrow shows the total force of all four springs, the force that moves the
junction. A spring pulls with k * (neighbour - junction), k = spring_constant / contour_length.
Arrow length = FORCE_ARROW_SCALE * force (NET_ARROW_SCALE * force for the black arrow).
Junction vector: when a junction is formed, the vector between the two closest points of the
two segments (from the point on the first molecule to the point on the second) is stored with
the junction. It is drawn as a short orange arrow, parallel to the vector (pointing the same
way) and centered on the junction (length JUNCTION_VECTOR_SCALE * |vector|, at least MIN_JUNCTION_VECTOR_LENGTH).
Release criterion: let A and B be the two molecules at a junction (A = the first, B = the
second), v the stored junction vector (from the closest point on A to the closest point on B),
FA the net force from the springs of A on the junction and FB the net force from those of B.
A junction is deleted (from both molecules) as soon as FA . v < -breakage_threshold or
FB . v > breakage_threshold (the projections are the force components along v, in force units).
While two segments of different molecules are closer than 2 (and not joined at a junction),
their two closest points are connected by a thin black dotted line. Crossing lines
are created by dragging the chain ends around.

Contour length: every line segment has a positive contour_length (initially the straight
length of its molecule). Its spring constant is spring_constant / contour_length. The
thickness of a segment shows the tension of its spring, spring_constant * d / contour_length
(d = distance between the end points of the segment), at most MAX_LINE_RADIUS thick. When a junction splits
a segment, its contour length is divided in proportion to where the junction sits on it;
when a junction disappears, the two segments it joined add up.

Slip: contour length can slip across a junction between the two neighbouring segments of the
same molecule. The slip dynamics lower the spring energy of each molecule, sum K d_i^2 / (2 L_i)
(d_i = distance between the end points of segment i, L_i = its contour length), while the
total contour length of the molecule stays constant: length flows into the segment with the
higher tension K d_i / L_i, until all segments of a molecule have L_i proportional to d_i
(equal tension). Detection and dynamics live in the
simulation (see DummySimulation); the viewer only draws what `get_network()` reports.

Mouse
- Left-drag on empty space  -> rotate the view
- Right-drag / mouse wheel  -> zoom
- Middle-drag (or Shift+left-drag) -> pan
- Left-drag on a chain-end sphere -> move it along the surface of the volume
- Hold Shift while dragging a chain end and move the cursor out past the circular
  edge of the projected sphere -> the bead crosses to the other side (front <-> back)
  of the volume; move the cursor back inside to keep dragging it there

Keyboard
- While a chain end is grabbed (mouse button held):
      d           -> delete the whole molecule the grabbed end belongs to
- Right / Left arrow -> add / remove contour length to the line segment connected to the
                  grabbed chain end (a longer contour makes the spring softer: lower tension)
- Hold the left button on a junction sphere (it turns black) and press
      x           -> remove the intersection: the junction disappears from both molecules at
                     once (each loses its bend there; the contour lengths of the two segments
                     it joined add up). The two molecules do not form a new junction until
                     they have separated, otherwise it would reappear immediately.
- a  -> create a new molecule: a straight line between two free spots on the surface
- r  -> shrink the spherical volume by 1 (down to a minimum radius of 5)
- R  -> grow the spherical volume by 1
- Shift + 1..9 -> store the current positions of all chain ends in slot 1..9
- 1..9         -> slowly move all chain ends to the positions stored in that slot, each
                  along a great circle of the sphere
- m  -> start recording a movie (movie00000.avi, movie00001.avi, ...); M -> stop recording
- p  -> save a PNG of the current view (img00000.png, img00001.png, ...)
- s  -> save all bead positions (conf00000.dump, conf00001.dump, ...), LAMMPS-dump-like:
        # <number of beads> <number of molecules>
        ITEM: NUMBER OF ATOMS / <number of beads>
        ITEM: BOX BOUNDS pp pp pp / xmin xmax / ymin ymax / zmin zmax  (extent of the beads)
        ITEM: ATOMS id mol type xu yu zu / one line per bead
        (ids and molecule numbers start at 1; type 1 = chain end, type 2 = junction with mol 0)
All files are written to the current folder; numbering continues after existing files.

Data flow
    simulation (LAMMPS)  --positions-->  viewer      (timer, every ~30 ms)
    viewer (user input)  --callbacks-->  simulation  (see SphereDragger.__init__)

Bead ordering: molecule by molecule, each as [start, end]; the junctions are not beads
and are reported separately by the simulation. Deleting a molecule shifts
the indices of all later beads (and removes its junctions); the callback reports the indices explained in its
comment, and the simulation must return positions in the new ordering from then on.
New molecules are appended at the end, so they don't shift existing indices.

Install:  pip install pyvista numpy
Movies:   pip install imageio imageio-ffmpeg
"""
import colorsys
import os
import re
import time

import numpy as np
import pyvista as pv
import vtk

KEYBOARD_LAYOUT = "nordic"

ACTIVE_COLOR = (0.0, 0.0, 0.0)  # highlight color while a sphere is being dragged

CONTOUR_STEP = 2.0              # contour length added/removed per press of Right/Left arrow
TENSION_THICKNESS = 0.01        # tube radius of a segment per unit of spring tension
FORCE_ARROW_SCALE = 0.1         # length of a force arrow per unit of force
FORCE_ARROW_WIDTH = 2.5         # width of the arrows with the force of one molecule at a junction
NET_ARROW_WIDTH = 1.5           # width of the black arrow with the total force on a junction
NET_ARROW_SCALE = 5.0 * FORCE_ARROW_SCALE   # the black total-force arrow is drawn 5 times longer
JUNCTION_VECTOR_SCALE = 20.0    # length of the orange arrow per unit length of the stored vector
MIN_JUNCTION_VECTOR_LENGTH = 5.0    # ... but never shorter than this, so it stays visible
JUNCTION_VECTOR_WIDTH = 1.5     # width of the orange arrow
JUNCTION_VECTOR_COLOR = (1.0, 0.55, 0.0)    # orange
DOT_SPACING = 0.2               # distance between the dots of the line between nearby segments
MAX_LINE_RADIUS = 1.0           # thickest tube drawn (a very short segment has a huge tension)

MIN_VOLUME_RADIUS = 5.0
VOLUME_RADIUS_STEP = 1.0

# Shift + digit produces a layout-specific symbol; which key name VTK reports for it
# depends on the keyboard layout (X11/Qt). Pick the layout matching your keyboard:
# keys for Shift+1 ... Shift+9. (Where VTK reports the plain digit together with the
# Shift modifier, e.g. on Windows, no table is needed.) Unknown shifted keys print a
# hint with the key name, so you can add your own entry.

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


def color_for_id(moleculeid):
    """Deterministic, well-separated RGB color (0..1) for an unsigned integer id."""
    hue = (int(moleculeid) * 0.618033988749895) % 1.0  # golden-ratio hue spacing
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


class SphereDragger:
    """Lets the user drag chain ends along a spherical surface, add/delete molecules,
    resize the volume, store/recall chain-end configurations, and save images / movies /
    bead positions; lets a simulation push new positions for all beads. Keeps the
    molecule lines and the volume shell in sync."""

    IMAGE_PATTERN = re.compile(r"^img(\d{5})\.png$")
    CONF_PATTERN = re.compile(r"^conf(\d{5})\.dump$")
    MOVIE_PATTERN = re.compile(r"^movie(\d{5})\.avi$")
    # characters whose built-in VTK meaning (reset camera, surface mode, pick, ...) is suppressed
    OWN_CHARS = frozenset("aAdDpPsSmMrRxX123456789")

    def __init__(self, plotter, template, actors, positions, moleculeids,
                 center, radius, sphere_radius, recall_speed=0.5, movie_fps=30, rng=None,
                 junction_radius=0.5, get_network=None, get_proximity=None, get_forces=None,
                 get_vectors=None,
                 on_drag=None, on_release=None,
                 on_add_molecule=None, on_delete_molecule=None,
                 on_radius_change=None, on_ends_driven=None, on_ends_released=None,
                 on_contour_change=None, on_remove_junction=None):
        self.plotter = plotter
        self.renderer = plotter.renderer
        self.template = template        # sphere mesh used for newly created beads
        self.actors = list(actors)
        self.positions = np.array(positions, dtype=float).reshape(-1, 3)   # (n, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)             # radius of the spherical volume
        self.sphere_radius = float(sphere_radius)   # radius of one bead
        self.junction_radius = float(junction_radius)   # radius of the junction spheres
        self.recall_speed = float(recall_speed)     # rad/s along great circles
        self.movie_fps = movie_fps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.next_moleculeid = int(self.moleculeids.max()) + 1 if len(self.moleculeids) else 0
        self.image_counter = self._first_free_number(self.IMAGE_PATTERN)
        self.conf_counter = self._first_free_number(self.CONF_PATTERN)
        self.movie_counter = self._first_free_number(self.MOVIE_PATTERN)

        # Pulled from the simulation every tick: (junction positions (J, 3),
        # [(moleculeid, moleculeid)] per junction, [(moleculeid, [node, ...], [contour_length, ...], [tension, ...])]).
        # Node k < n is bead k, node n + r is junction r. None = no junctions.
        self.get_network = get_network
        # [(molecule_a, molecule_b, point_a, point_b)]: closest points of nearby segments
        self.get_proximity = get_proximity
        self.get_vectors = get_vectors  # () -> (J, 3) closest-point vector stored with each junction
        self.get_forces = get_forces    # () -> (J, 2, 3) spring force of each molecule on each junction
        arrow = vtk.vtkArrowSource()    # unit arrow along +x
        arrow.SetTipResolution(16)
        arrow.SetShaftResolution(12)
        self.arrow_mapper = vtk.vtkPolyDataMapper()
        self.arrow_mapper.SetInputConnection(arrow.GetOutputPort())
        self.vector_arrows = []         # pool: orange arrow with the stored vector of each junction
        self.arrow_actors = []          # pool: three arrows per junction (molecule a, b, total)

        # Callbacks into the simulation:
        self.on_drag = on_drag          # on_drag(index, xyz)                    while dragging
        self.on_release = on_release    # on_release(index, xyz)                 when let go
        self.on_add_molecule = on_add_molecule        # (first_index, positions (2, 3), moleculeid)
        self.on_delete_molecule = on_delete_molecule  # (first_index, count)  old indices
        self.on_radius_change = on_radius_change      # (new_volume_radius)
        self.on_ends_driven = on_ends_driven          # (indices, positions) chain ends moved by a recall
        self.on_ends_released = on_ends_released      # (indices)            recall finished/cancelled
        self.on_remove_junction = on_remove_junction  # (row)  remove junction `row` from both molecules
        self.on_contour_change = on_contour_change    # (index, delta)  add/remove contour length at chain end

        self.interactor = plotter.iren.interactor
        self.picker = vtk.vtkPropPicker()
        self.dragged = None             # index of the chain end being dragged
        self.selected_junction = None   # index of the junction being held (left button down)
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
        # Segments are drawn as tubes whose radius follows the contour length. Every segment
        # has its own two points (so the radius is constant along it); the tube filter turns
        # the lines into tubes and the mapper colors them by the per-point "rgb" array.
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
        self.junction_template = pv.Sphere(radius=self.junction_radius,
                                           theta_resolution=16, phi_resolution=16)
        self.junction_actors = []
        # dotted black line between the closest points of nearby segments: a cloud of dots
        self.proximity_mapper = vtk.vtkPolyDataMapper()
        self.proximity_actor = vtk.vtkActor()
        self.proximity_actor.SetMapper(self.proximity_mapper)
        self.proximity_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.proximity_actor.GetProperty().SetPointSize(3)
        self.proximity_actor.GetProperty().SetRenderPointsAsSpheres(True)
        self.proximity_actor.SetVisibility(False)
        plotter.add_actor(self.proximity_actor, reset_camera=False, pickable=False)
        self.junction_positions = np.zeros((0, 3))
        self.junction_molecules = []
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
                    "r", "R", "x", "X", *"123456789"):
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
        # CharEvent that follows a key press ('p' = pick, 's' = surface mode, 'r' = reset
        # camera, '3' = stereo, ...). A higher-priority observer neutralises the
        # characters we use before the style sees them.
        self.interactor.AddObserver("CharEvent", self._on_char, 10.0)

    # ------------------------------------------------------------ timer tick
    def tick(self, step_simulation):
        """Call every ~30 ms: advance a running recall animation, take a simulation
        step (`step_simulation()` returns the (n, 3) positions), draw, record a frame.

        Re-entrancy guard: grabbing a movie frame renders the window again, and VTK may
        deliver the next timer event while that is still going on. A tick that arrives
        while one is still running is simply skipped."""
        if self._in_tick:
            return
        self._in_tick = True
        try:
            self.animate()
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
        self._pull_network()
        self._pull_proximity()
        self._pull_forces()
        self._pull_vectors()
        self.plotter.render()

    def get_positions(self):
        return self.positions.copy()

    # ------------------------------------------------------ molecule lines, junctions
    def _all_points(self):
        return np.vstack([self.positions, self.junction_positions])

    def _segment_radii(self):
        """Tube radius of every segment (order of self._seg_nodes) from its spring tension."""
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
        """Needed when the number of beads, junctions or lines changed."""
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
        """No simulation-provided network: every molecule is a straight line, no junctions."""
        lines = {}
        for i, m in enumerate(self.moleculeids.tolist()):
            lines.setdefault(m, []).append(i)
        return np.zeros((0, 3)), [], [(m, idx, [self.radius] * (len(idx) - 1),
                                       [30.0] * (len(idx) - 1))
                                      for m, idx in lines.items()]

    def _pull_network(self, force=False):
        """Fetch junctions and lines from the simulation and bring the drawing up to date."""
        jpos, jmols, lines = (self.get_network() if self.get_network is not None
                              else self._default_network())
        self.junction_positions = np.asarray(jpos, dtype=float).reshape(-1, 3)
        jmols = [(int(a), int(b)) for a, b in jmols]
        self.lines = [(int(m), [int(k) for k in idx], list(cs), [float(t) for t in ts])
                      for m, idx, cs, ts in lines]
        self._seg_tension = [t for _m, _idx, _cs, ts in self.lines for t in ts]
        signature = tuple((m, tuple(idx)) for m, idx, _cs, _ts in self.lines)
        changed = force or signature != self._line_signature or self.bond_mesh is None
        self._line_signature = signature

        # a junction sphere is never thinner than the segments meeting at it
        n = len(self.positions)
        radii = np.full(len(jmols), self.junction_radius)
        seg_radius = self._segment_radii()
        k = 0
        for _m, idx, _cs, _ts in self.lines:
            for a, b in zip(idx[:-1], idx[1:]):
                for node in (a, b):
                    if node >= n and node - n < len(radii):
                        radii[node - n] = max(radii[node - n], 1.1 * seg_radius[k])
                k += 1

        if jmols != self.junction_molecules or len(self.junction_actors) != len(jmols):
            self._sync_junction_actors(jmols)
        for actor, p, r in zip(self.junction_actors, self.junction_positions, radii):
            actor.SetPosition(*p)
            scale = r / self.junction_radius
            actor.SetScale(scale, scale, scale)
        if changed:
            self._rebuild_bonds()
        else:
            self._refresh_bonds()

    def _set_arrow(self, actor, origin, force, width, color, scale=FORCE_ARROW_SCALE):
        """Point `actor` (a unit arrow along +x) from `origin` along `force`."""
        magnitude = float(np.linalg.norm(force))
        if magnitude < 1e-9:
            actor.SetVisibility(False)
            return
        u = np.asarray(force, dtype=float) / magnitude
        helper = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        p = np.cross(u, helper)
        p /= np.linalg.norm(p)
        q = np.cross(u, p)
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):      # columns: arrow axis (scaled to its length), two width axes, origin
            matrix.SetElement(i, 0, u[i] * magnitude * scale)
            matrix.SetElement(i, 1, p[i] * width)
            matrix.SetElement(i, 2, q[i] * width)
            matrix.SetElement(i, 3, origin[i])
        actor.SetUserMatrix(matrix)
        actor.GetProperty().SetColor(*color)
        actor.SetVisibility(True)

    def _pull_vectors(self):
        """Draw the vector stored with each junction as a short orange arrow, parallel to the
        vector (pointing the same way) and centered on the junction."""
        vectors = (np.asarray(self.get_vectors(), dtype=float).reshape(-1, 3)
                   if self.get_vectors is not None else np.zeros((0, 3)))
        count = min(len(vectors), len(self.junction_positions))
        while len(self.vector_arrows) < count:
            actor = vtk.vtkActor()
            actor.SetMapper(self.arrow_mapper)
            actor.SetVisibility(False)
            self.plotter.add_actor(actor, reset_camera=False, pickable=False)
            self.vector_arrows.append(actor)
        for j in range(count):
            magnitude = float(np.linalg.norm(vectors[j]))
            if magnitude < 1e-12:
                self.vector_arrows[j].SetVisibility(False)   # direction undefined
                continue
            u = vectors[j] / magnitude
            length = max(JUNCTION_VECTOR_SCALE * magnitude, MIN_JUNCTION_VECTOR_LENGTH)
            start = self.junction_positions[j] - 0.5 * length * u    # middle of the arrow at the junction
            self._set_arrow(self.vector_arrows[j], start, u * length, JUNCTION_VECTOR_WIDTH,
                            JUNCTION_VECTOR_COLOR, scale=1.0)
        for actor in self.vector_arrows[count:]:
            actor.SetVisibility(False)

    def _pull_forces(self):
        """At each junction: the spring forces of the two molecules (in their colors) and the
        total force (black) as arrows."""
        forces = (np.asarray(self.get_forces(), dtype=float).reshape(-1, 2, 3)
                  if self.get_forces is not None else np.zeros((0, 2, 3)))
        count = min(len(forces), len(self.junction_positions), len(self.junction_molecules))
        while len(self.arrow_actors) < 3 * count:
            actor = vtk.vtkActor()
            actor.SetMapper(self.arrow_mapper)
            actor.SetVisibility(False)
            self.plotter.add_actor(actor, reset_camera=False, pickable=False)
            self.arrow_actors.append(actor)
        for j in range(count):
            origin = self.junction_positions[j]
            mol_a, mol_b = self.junction_molecules[j]
            self._set_arrow(self.arrow_actors[3 * j], origin, forces[j, 0],
                            FORCE_ARROW_WIDTH, color_for_id(mol_a))
            self._set_arrow(self.arrow_actors[3 * j + 1], origin, forces[j, 1],
                            FORCE_ARROW_WIDTH, color_for_id(mol_b))
            self._set_arrow(self.arrow_actors[3 * j + 2], origin, forces[j].sum(axis=0),
                            NET_ARROW_WIDTH, (0.0, 0.0, 0.0), NET_ARROW_SCALE)
        for actor in self.arrow_actors[3 * count:]:
            actor.SetVisibility(False)

    def _pull_proximity(self):
        """Connect the closest points of nearby, not yet joined segments by a thin black
        dotted line (dots every DOT_SPACING, including both end points)."""
        pairs = self.get_proximity() if self.get_proximity is not None else []
        clouds = []
        for _mol_a, _mol_b, point_a, point_b in pairs:
            n = max(int(np.linalg.norm(point_b - point_a) / DOT_SPACING), 1)
            t = (np.arange(n + 1) / n)[:, None]
            clouds.append(point_a + t * (point_b - point_a))
        if not clouds:
            self.proximity_actor.SetVisibility(False)
            return
        self.proximity_mapper.SetInputData(pv.PolyData(np.vstack(clouds)))
        self.proximity_actor.SetVisibility(True)

    def _sync_junction_actors(self, jmols):
        """(Re)create the junction spheres, each in the average color of its two molecules."""
        self.selected_junction = None  # the actors are recreated
        for actor in self.junction_actors:
            self.plotter.remove_actor(actor, reset_camera=False)
        self.junction_actors = []
        for a, b in jmols:
            color = 0.5 * (np.array(color_for_id(a)) + np.array(color_for_id(b)))
            actor = self.plotter.add_mesh(self.junction_template, color=tuple(color),
                                          smooth_shading=True, reset_camera=False)
            actor.SetPickable(True)    # a junction can be held and then removed with x
            self.junction_actors.append(actor)
        self.junction_molecules = jmols

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
        self._pull_network(force=True)                   # its junctions are gone too
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
            xmin xmax
            ymin ymax
            zmin zmax
            ITEM: ATOMS id mol type xu yu zu
            id molecule beadtype x y z          (one line per bead)

        Ids and molecule numbers start at 1 (molecules are renumbered 1..M in file order);
        beadtype is 1 for a chain end; junctions follow after all chain ends with
        beadtype 2 and molecule 0 (a junction belongs to two molecules); the box bounds are
        the smallest and largest coordinates."""
        path = os.path.join(os.getcwd(), f"conf{self.conf_counter:05d}.dump")
        # molecule numbers 1..M, in order of appearance (moleculeids may have gaps after deletions)
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
                if k <= n_ends:     # chain end: type 1, belongs to one molecule
                    mol, typ = order[int(self.moleculeids[k - 1])], 1
                else:               # junction: type 2, belongs to two molecules -> mol 0
                    mol, typ = 0, 2
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
        for k, actor in enumerate(self.junction_actors):
            if picked is actor:                           # hold a junction: x removes it
                self.selected_junction = k
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

    def _junction_color(self, k):
        a, b = self.junction_molecules[k]
        return tuple(0.5 * (np.array(color_for_id(a)) + np.array(color_for_id(b))))

    def _remove_junction(self):
        """Remove the held junction from both molecules."""
        row, self.selected_junction = self.selected_junction, None
        if self.on_remove_junction is not None:
            self.on_remove_junction(row)
        self._pull_network(force=True)
        self.plotter.render()

    def _on_release(self, obj, event):
        if self.selected_junction is not None:
            k, self.selected_junction = self.selected_junction, None
            if k < len(self.junction_actors):
                self.junction_actors[k].GetProperty().SetColor(*self._junction_color(k))
            self.plotter.render()
            return
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

        if lower == "p":
            self._save_image()
        elif lower == "s":
            self._save_configuration()
        elif lower == "a":
            self._add_molecule()
        elif key == "r":
            self._change_radius(-VOLUME_RADIUS_STEP)
        elif key == "R":
            self._change_radius(+VOLUME_RADIUS_STEP)
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
        elif self.selected_junction is not None and lower == "x":
            self._remove_junction()
        elif self.dragged is not None and lower == "d":
            self._delete_molecule()
        elif self.dragged is not None and key in ("Right", "Left"):
            if self.on_contour_change is not None:
                self.on_contour_change(self.dragged,
                                       CONTOUR_STEP if key == "Right" else -CONTOUR_STEP)
                self._pull_network()
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


def make_molecules(n_molecules, center, radius, rng):
    """Build M molecules, each a straight line between a start bead and an end bead,
    both on the surface (no internal beads).

    Returns positions (2M, 3) and moleculeids (2M,) uint32.
    Beads are ordered molecule by molecule as [start, end]."""
    anchors = make_surface_points(2 * n_molecules, center, radius)
    anchors = anchors[rng.permutation(2 * n_molecules)]
    starts, ends = anchors[:n_molecules], anchors[n_molecules:]

    positions = np.empty((2 * n_molecules, 3))
    positions[0::2] = starts
    positions[1::2] = ends
    moleculeids = np.repeat(np.arange(n_molecules), 2).astype(np.uint32)
    return positions, moleculeids


# --------------------------------------------------------------------------
# Stand-in for the simulation. Replace with your LAMMPS wrapper (see below).
# --------------------------------------------------------------------------
class DummySimulation:
    """Toy dynamics, no randomness.

    Molecules: a molecule is a piecewise linear curve  start end -> junction -> ... -> end
    end. Initially it has no junctions, i.e. it is the straight line between its two ends.

    Chain ends: stay on the spherical surface, and are pushed apart (excluded volume)
    until the centres of ends of DIFFERENT molecules are at least `2 * bead_radius` apart.
    Ends that the viewer is moving (grabbed by the user, or travelling to a stored
    position) are immovable for the simulation, so they push other beads out of the way.

    Junctions: when two line segments of two different molecules intersect, a junction
    node is inserted into BOTH molecules at the intersection point (each molecule gets a
    bend there). Intersection is detected when
      * the closest points of the two segments (both strictly inside the segments) are
        closer than `join_tolerance`, or
      * the two segments passed through each other since the previous step (the sign of
        their signed separation along the common normal flipped) and are now closer
        than `crossing_tolerance` - this catches fast crossings that the tolerance test
        alone would miss.
    Each junction has four springs, one along each of the four segments connected to it
    (previous and next node of both molecules). A spring has zero rest length and the
    spring constant  k = spring_constant / contour_length  of its segment, so its tension is
    F = k * (neighbour - junction) and the junction is pulled towards its four neighbours.
    Overdamped dynamics:  dx = (1 - exp(-spring_rate * sum(k))) * sum(F) / sum(k)  per substep,
    i.e. a fraction of the way to the k-weighted mean of the neighbours (never overshoots,
    however stiff the springs get). Junctions always stay inside the ball.

    Slip: across every junction contour length may slip between the two neighbouring segments
    of the same molecule. The spring energy of a molecule is  E = sum K d_i^2 / (2 L_i)  (d_i =
    segment length, L_i = contour length). For two neighbouring segments with fixed total
    L_1 + L_2 the minimum is at L_1 : L_2 = d_1 : d_2 (equal tension K d / L). Each substep a
    fraction (1 - exp(-slip_rate), at most 0.45) of the way to that minimum is slipped, so
    the energy decreases while the total contour length of the molecule is exactly conserved
    and no segment gets shorter than `min_contour_length` through slip.

    Breakage: a junction is deleted when FA . v < -breakage_threshold or FB . v > breakage_threshold,
    with v the stored junction vector (closest point on molecule A -> closest point on molecule B)
    and FA, FB the net spring forces of A and B on the junction (see `junction_forces`). The
    projections are the components of the forces along v / |v|, so the threshold is a force. As
    after a removal by the user, the two molecules cannot form a new junction until they have
    stopped intersecting.

    Proximity: pairs of segments of different molecules that are closer than
    `proximity_distance` and not joined at a junction are reported by `proximity()`
    (the two closest points of the segments), for drawing."""

    def __init__(self, positions, moleculeids, center, radius, bead_radius, substeps=5,
                 spring_rate=0.02, join_tolerance=0.3, crossing_tolerance=2.0,
                 min_contour_length=0.5, spring_constant=30.0, proximity_distance=2.0,
                 slip_rate=0.02, breakage_threshold=5.0):
        self.x = np.array(positions, dtype=float).reshape(-1, 3)   # chain ends (n, 3)
        self.moleculeids = np.array(moleculeids, dtype=np.uint32)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.bead_radius = float(bead_radius)
        self.min_distance = 2.0 * self.bead_radius  # centres closer than this = overlap
        self.substeps = substeps        # dynamics substeps per viewer update
        self.spring_rate = float(spring_rate)       # mobility: speed of the junctions
        self.spring_constant = float(spring_constant)   # segment spring constant = this / contour length
        self.proximity_distance = float(proximity_distance)
        self.breakage_threshold = float(breakage_threshold)   # force along v that breaks a junction
        self.slip_rate = float(slip_rate)           # speed of contour length slipping across junctions
        self.near = []                  # [(molecule_a, molecule_b, point_a, point_b)]
        self.join_tolerance = float(join_tolerance)
        self.crossing_tolerance = float(crossing_tolerance)
        self.held = set()               # beads currently controlled by the viewer

        # junctions: parallel containers (row r <-> jids[r]); a junction belongs to two molecules
        self.jids = []
        self.jx = np.zeros((0, 3))
        self.jmols = []                 # [(moleculeid_a, moleculeid_b)]
        self._suppressed = set()        # {frozenset({mol, mol})}: removed by the user, may not rejoin yet
        self.jvec = np.zeros((0, 3))    # vector between the two closest points when the junction formed
        self.chains = {int(m): [] for m in np.unique(self.moleculeids)}  # molecule -> [jid, ...] in order
        # contour length of every segment: molecule -> [L, ...], one more entry than chains[mol];
        # initially the straight length of the molecule
        self.contours = {}
        for m in self.chains:
            a, b = self.x[self.moleculeids == m][:2]
            self.contours[m] = [float(np.linalg.norm(b - a))]
        self.min_contour_length = float(min_contour_length)
        self.next_jid = 0
        self._prev_sign = {}            # (segment, segment) -> side of the previous step

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
        self.chains[int(moleculeid)] = []
        self.contours[int(moleculeid)] = [float(np.linalg.norm(positions[1] - positions[0]))]

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

    def remove_junction(self, row):
        """The user removed junction number `row`: it disappears from BOTH molecules (each
        loses the bend, contour lengths of the joined segments add up). The two molecules may
        not form a junction again until they have stopped intersecting (else it reappears)."""
        jid, mols = self.jids[row], self.jmols[row]
        self._remove_junction(jid)
        self._suppressed.add(frozenset(mols))
        self._prev_sign = {}

    def _release_junctions(self):
        """Delete every junction for which FA . v < -breakage_threshold or
        FB . v > breakage_threshold (see the class docstring)."""
        if not self.jids:
            return
        forces = self.junction_forces()
        doomed = []
        for r in range(len(self.jids)):
            length = float(np.linalg.norm(self.jvec[r]))
            if length < 1e-12:
                continue                                 # direction undefined
            u = self.jvec[r] / length
            if (forces[r, 0] @ u < -self.breakage_threshold
                    or forces[r, 1] @ u > self.breakage_threshold):
                doomed.append(self.jids[r])
        for jid in doomed:
            self.remove_junction(self.jids.index(jid))

    def delete_molecule(self, first_index, count):
        """The user deleted the molecule occupying beads [first_index, first_index + count).
        Its junctions disappear; the other molecule at each of them loses that bend."""
        mol = int(self.moleculeids[first_index])
        for jid in list(self.chains.get(mol, [])):
            self._remove_junction(jid)
        self.chains.pop(mol, None)
        self.contours.pop(mol, None)
        self._suppressed = {pair for pair in self._suppressed if mol not in pair}
        self._prev_sign = {}            # node numbering changed

        block = slice(first_index, first_index + count)
        self.x = np.delete(self.x, block, axis=0)
        self.moleculeids = np.delete(self.moleculeids, block)
        self.held = {h - count if h >= first_index + count else h
                     for h in self.held if not (first_index <= h < first_index + count)}

    def network(self):
        """Everything the viewer needs to draw the molecules and junctions:
        (junction positions (J, 3), junction molecules [(a, b)] * J, lines).
        `lines` is [(moleculeid, [node, ...], [contour_length, ...], [tension, ...])], the
        nodes ordered start -> end and one contour length and spring tension per segment
        between consecutive nodes;
        node k < n is chain end k (viewer bead index), node n + r is junction row r."""
        nodes = np.vstack([self.x, self.jx])
        lines = []
        for m, idx in self._lines().items():
            d = np.linalg.norm(nodes[idx[1:]] - nodes[idx[:-1]], axis=1)
            tension = self.spring_constant * d / np.array(self.contours[m])   # k * d, k = K / L
            lines.append((m, idx, list(self.contours[m]), tension.tolist()))
        return self.jx.copy(), list(self.jmols), lines

    def junction_vectors(self):
        """(J, 3) array: for each junction the vector between the two closest points of the two
        segments just before they were merged into the junction (stored when it formed)."""
        return self.jvec.copy()

    def junction_forces(self):
        """(J, 2, 3) array: for each junction, the sum of the two spring forces that the two
        segments of molecule a, and of molecule b, exert on it, in the order of `jmols`.
        A spring pulls the junction towards its neighbour with k * (neighbour - junction),
        k = spring_constant / contour_length of the segment. The total force on the junction
        is the sum over axis 1."""
        n = len(self.x)
        nodes = np.vstack([self.x, self.jx])
        forces = np.zeros((len(self.jids), 2, 3))
        for m, idx in self._lines().items():
            cl = self.contours[m]
            for k in range(1, len(idx) - 1):
                r = idx[k] - n
                f = (self.spring_constant / cl[k - 1] * (nodes[idx[k - 1]] - nodes[idx[k]])
                     + self.spring_constant / cl[k] * (nodes[idx[k + 1]] - nodes[idx[k]]))
                forces[r, self.jmols[r].index(m)] = f
        return forces

    def proximity(self):
        """[(moleculeid_a, moleculeid_b, point_on_a, point_on_b)] for every pair of segments
        of different molecules that are closer than `proximity_distance` and not joined."""
        return list(self.near)

    # --- topology -----------------------------------------------------------------
    def _lines(self):
        """{moleculeid: [node, ...]} from start to end, junctions included."""
        n = len(self.x)
        rows = {}
        for i, m in enumerate(self.moleculeids.tolist()):
            rows.setdefault(m, []).append(i)
        jrow = {jid: r for r, jid in enumerate(self.jids)}
        return {m: [r[0]] + [n + jrow[j] for j in self.chains.get(m, [])] + [r[-1]]
                for m, r in rows.items()}

    def _neighbors(self, lines):
        """Springs attached to each junction: (J, 4) neighbour nodes and (J, 4) spring
        constants  spring_constant / contour_length  of the segments leading to them."""
        n = len(self.x)
        nb = [[] for _ in self.jids]
        kk = [[] for _ in self.jids]
        for m, idx in lines.items():
            cl = self.contours[m]
            for k in range(1, len(idx) - 1):
                nb[idx[k] - n] += [idx[k - 1], idx[k + 1]]
                kk[idx[k] - n] += [self.spring_constant / cl[k - 1],
                                   self.spring_constant / cl[k]]
        return np.array(nb, dtype=int).reshape(-1, 4), np.array(kk, dtype=float).reshape(-1, 4)

    def _add_junction(self, seg_a, seg_b, position, lines, fractions, vector):
        """Insert a junction into both molecules, between the end nodes of their segments.
        `fractions` = where the junction sits on each segment (0 = first node, 1 = second);
        the contour length of the segment is divided in that proportion. `vector` is the vector
        between the two closest points (point on seg_a -> point on seg_b), stored for display."""
        jid = self.next_jid
        self.next_jid += 1
        for (m, a, _b), f in zip((seg_a, seg_b), fractions):
            k = lines[m].index(a)                           # segment k runs from node a ...
            self.chains[m].insert(k, jid)                   # ... and the junction goes right after a
            length = self.contours[m][k]
            f = float(f)
            self.contours[m][k:k + 1] = [length * f, length * (1.0 - f)]
        self.jids.append(jid)
        self.jx = np.vstack([self.jx, position])
        self.jvec = np.vstack([self.jvec, vector])
        self.jmols.append((seg_a[0], seg_b[0]))

    def _remove_junction(self, jid):
        r = self.jids.index(jid)
        for m in self.jmols[r]:
            if jid in self.chains.get(m, []):
                k = self.chains[m].index(jid)               # joins segments k and k + 1
                self.contours[m][k:k + 2] = [self.contours[m][k] + self.contours[m][k + 1]]
                self.chains[m].remove(jid)
        del self.jids[r]
        del self.jmols[r]
        self.jx = np.delete(self.jx, r, axis=0)
        self.jvec = np.delete(self.jvec, r, axis=0)

    # --- intersection detection -----------------------------------------------------
    def _segment_pairs(self):
        """All pairs of segments that belong to different molecules, as
        (lines, nodes, segs, A, B, I, J): segment k is segs[k] = (molecule, node a, node b)
        with node numbers A[k], B[k]; pair q is the segments I[q] and J[q]. None if no pair."""
        lines = self._lines()
        nodes = np.vstack([self.x, self.jx])
        segs = [(m, a, b) for m, idx in lines.items() for a, b in zip(idx[:-1], idx[1:])]
        if len(segs) < 2:
            return None
        mol = np.array([sg[0] for sg in segs])
        A = np.array([sg[1] for sg in segs])
        B = np.array([sg[2] for sg in segs])
        I, J = np.triu_indices(len(segs), 1)
        keep = mol[I] != mol[J]
        I, J = I[keep], J[keep]
        if len(I) == 0:
            return None
        return lines, nodes, segs, A, B, I, J

    @staticmethod
    def _closest_points(P, d, Q, e_vec):
        """Closest points between the segments P + s d and Q + t e_vec (s, t in [0, 1]),
        for arrays of segments. Returns the point on each."""
        r = P - Q
        a = np.maximum((d * d).sum(1), 1e-12)
        e = np.maximum((e_vec * e_vec).sum(1), 1e-12)
        b, c, f = (d * e_vec).sum(1), (d * r).sum(1), (e_vec * r).sum(1)
        den = a * e - b * b
        ok = den > 1e-9 * a * e
        s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, den, 1.0), 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0, np.clip(-c / a, 0.0, 1.0),
                     np.where(t > 1, np.clip((b - c) / a, 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
        return P + s[:, None] * d, Q + t[:, None] * e_vec

    def _update_proximity(self):
        """Find segments of different molecules that are close but not joined."""
        self.near = []
        pairs = self._segment_pairs()
        if pairs is None:
            return
        _lines, nodes, segs, A, B, I, J = pairs
        P0 = nodes[A]
        D = nodes[B] - P0
        p, q = self._closest_points(P0[I], D[I], P0[J], D[J])
        dist = np.linalg.norm(p - q, axis=1)
        shares = (A[I] == A[J]) | (A[I] == B[J]) | (B[I] == A[J]) | (B[I] == B[J])
        for k in np.flatnonzero(~shares & (dist < self.proximity_distance)):
            self.near.append((segs[I[k]][0], segs[J[k]][0], p[k].copy(), q[k].copy()))

    def _detect_crossings(self):
        """Find intersecting segments of different molecules and add junctions there."""
        pairs = self._segment_pairs()
        if pairs is None:
            self._prev_sign = {}
            return
        lines, nodes, segs, A, B, I, J = pairs

        P0 = nodes[A]
        D = nodes[B] - P0
        di, dj, r = D[I], D[J], P0[I] - P0[J]
        a, e = (di * di).sum(1), (dj * dj).sum(1)
        b, c, f = (di * dj).sum(1), (di * r).sum(1), (dj * r).sum(1)
        den = a * e - b * b
        ok = den > 1e-9 * a * e                      # not (nearly) parallel
        den = np.where(ok, den, 1.0)
        s = (b * f - c * e) / den                    # closest point on segment i: P0 + s*d
        t = (a * f - b * c) / den                    # closest point on segment j
        p = P0[I] + s[:, None] * di
        q = P0[J] + t[:, None] * dj
        dist = np.linalg.norm(p - q, axis=1)
        sign = np.sign((np.cross(di, dj) * (P0[J] - P0[I])).sum(1))   # side of j relative to i

        margin = 0.01
        shares = ((A[I] == A[J]) | (A[I] == B[J]) | (B[I] == A[J]) | (B[I] == B[J]))
        inside = (ok & ~shares & (s > margin) & (s < 1 - margin)
                  & (t > margin) & (t < 1 - margin))

        old = self._prev_sign
        ok_idx = np.flatnonzero(ok)
        self._prev_sign = {(segs[i], segs[j]): sg for i, j, sg in
                           zip(I[ok_idx].tolist(), J[ok_idx].tolist(), sign[ok_idx].tolist())}

        candidates = []
        for k in np.flatnonzero(inside & (dist < self.crossing_tolerance)):
            prev = old.get((segs[I[k]], segs[J[k]]))
            flipped = prev is not None and prev * sign[k] < 0
            if dist[k] < self.join_tolerance or flipped:
                candidates.append(k)
        candidates.sort(key=lambda k: dist[k])

        used = set()    # one new junction per molecule and step (node numbers shift)
        still_crossing = set()
        for k in candidates:
            si, sj = segs[I[k]], segs[J[k]]
            pair = frozenset((si[0], sj[0]))
            if pair in self._suppressed:    # removed by the user and still intersecting
                still_crossing.add(pair)
                continue
            if si[0] in used or sj[0] in used:
                continue
            self._add_junction(si, sj, 0.5 * (p[k] + q[k]), lines, (s[k], t[k]),
                               q[k] - p[k])
            used.update((si[0], sj[0]))
        self._suppressed &= still_crossing  # separated pairs may form a junction again

    # --- dynamics ---------------------------------------------------------------
    def total_length(self):
        """Total contour length of all molecules."""
        nodes = np.vstack([self.x, self.jx])
        return float(sum(np.linalg.norm(nodes[idx[1:]] - nodes[idx[:-1]], axis=1).sum()
                         for idx in self._lines().values()))

    def _slip(self, lines):
        """One slip step: across every junction, move contour length between the two
        neighbouring segments of a molecule towards the energy minimum L_1 : L_2 = d_1 : d_2.
        All transfers are computed from the old lengths and applied together; every transfer is
        antisymmetric, so the total contour length of each molecule is conserved."""
        f = min(1.0 - np.exp(-self.slip_rate), 0.45)
        nodes = np.vstack([self.x, self.jx])
        for m, idx in lines.items():
            if len(idx) < 3:
                continue                                 # no junction, nothing can slip
            d = np.linalg.norm(nodes[idx[1:]] - nodes[idx[:-1]], axis=1)
            old = self.contours[m]
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

    def _apply_constraints(self):
        """Chain ends stay on the spherical surface, junctions inside the ball."""
        v = self.x - self.center
        r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        self.x = self.center + self.radius * v / r
        if len(self.jx):
            v = self.jx - self.center
            r = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            self.jx = self.center + v * np.minimum(1.0, 0.999 * self.radius / r)

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
        """Advance the 'simulation' and return an (n, 3) array of chain-end positions
        (junctions are fetched with `network()`)."""
        lines = self._lines()
        if self.jids:
            nbr, kk = self._neighbors(lines)
            ksum = kk.sum(axis=1)
            fraction = (1.0 - np.exp(-self.spring_rate * ksum))[:, None]
        else:
            nbr = None
        for _ in range(self.substeps):
            self._apply_constraints()
            self._separate_molecules()
            if nbr is not None:
                # four zero-length springs per junction: tension k * (neighbour - junction);
                # move a fraction of the way to the k-weighted mean of the neighbours
                nodes = np.vstack([self.x, self.jx])
                pull = (kk[..., None] * (nodes[nbr] - self.jx[:, None, :])).sum(axis=1)
                self.jx = self.jx + fraction * pull / ksum[:, None]
                self._slip(lines)
                nbr, kk = self._neighbors(lines)        # spring constants follow the contour lengths
                ksum = kk.sum(axis=1)
                fraction = (1.0 - np.exp(-self.spring_rate * ksum))[:, None]
        self._release_junctions()
        self._detect_crossings()
        self._update_proximity()
        return self.x

    def min_intermolecular_distance(self):
        """Smallest distance between chain ends of different molecules (for checking)."""
        diff = self.x[:, None, :] - self.x[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        other = self.moleculeids[:, None] != self.moleculeids[None, :]
        return float(dist[other].min()) if other.any() else np.inf


# Example of the same interface on top of the LAMMPS Python API.
# Creating/deleting molecules changes the atom count and shifts viewer indices, so keep a
# list `self.ids` (viewer index -> LAMMPS atom id) instead of assuming index == id - 1.
# Each molecule is two atoms joined by one bond, e.g.
#   bond_style harmonic / or a rigid link; atoms confined to the sphere surface
#   (fix wall/region on a spherical shell, or re-projection after every run)
#
#   from lammps import lammps
#   class LammpsSimulation:
#       def __init__(self, input_file):
#           self.lmp = lammps()
#           self.lmp.file(input_file)
#           self.ids = list(range(1, self.lmp.get_natoms() + 1))  # molecule-by-molecule order
#       def set_surface_position(self, index, xyz): ...  # move atom self.ids[index], zero velocity
#       def release(self, index, xyz): ...               # let the atom move again
#       def set_driven_positions(self, indices, positions): ...
#       def release_driven(self, indices): ...
#       def set_radius(self, radius): ...                # update wall region, project atoms
#       def add_molecule(self, first_index, positions, moleculeid):
#           # create_atoms for both rows of `positions`, molecule id `moleculeid`,
#           # one bond between them, then self.ids.extend(new atom ids)
#           ...
#       def delete_molecule(self, first_index, count):
#           # group rm molecule ..., delete_atoms group rm bond yes molecule yes,
#           # then del self.ids[first_index:first_index + count]
#           ...
#       def change_contour_length(self, index, delta): ...   # Right / Left arrow while grabbing an end
#       def remove_junction(self, row): ...  # user removed a junction: remove it from both molecules
#       def junction_vectors(self): ...  # (J, 3) closest-point vector stored when a junction formed
#       def junction_forces(self): ...  # (J, 2, 3): spring force of molecule a / b on each junction
#       def proximity(self): ...  # [(mol_a, mol_b, point_a, point_b)] for nearby unjoined segments
#       def network(self): ...  # (junction positions, molecule pairs, lines), see DummySimulation.network
#       def step(self):
#           self.lmp.command("run 10 pre no post no")
#           x = np.array(self.lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)  # ordered by atom id
#           return x[np.array(self.ids) - 1]


def main():
    rng = np.random.default_rng(42)

    # ---- geometry ----
    center = np.zeros(3)
    volume_radius = 15.0    # radius of the spherical volume (r / R change it at run time)
    sphere_radius = 1.0     # radius of each bead
    n_molecules = 3         # initial number of molecules; press 'a' to add more

    # positions: (2 * n_molecules, 3) ordered [start, end, start, end, ...]; moleculeids: (n,) uint32
    points, moleculeids = make_molecules(n_molecules, center, volume_radius, rng)

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
                          bead_radius=sphere_radius)   # spring_rate, join_tolerance, ... tune here
    # sim = LammpsSimulation("in.sim")

    def on_release(i, xyz):
        print(f"Chain end {i} (moleculeid={dragger.moleculeids[i]}) released at",
              np.round(xyz, 4))
        sim.release(i, xyz)

    dragger = SphereDragger(
        plotter, template, actors, points, moleculeids, center, volume_radius,
        sphere_radius=sphere_radius, rng=rng,
        junction_radius=0.5 * sphere_radius,
        get_network=sim.network,
        get_proximity=sim.proximity,
        get_forces=sim.junction_forces,
        get_vectors=sim.junction_vectors,
        on_remove_junction=sim.remove_junction,
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
