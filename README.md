# EntanglementSimulator

Three scripts to experiment with Entanglements.

## Dependencies

```
pip install pyvista numpy
pip install imageio imageio-ffmpeg   (movies)
```

## Before you run

The scripts implement code to identify the keyboard such that shortcut keys work. This is tested on linux only.
Otherwise look for KEYBOARD_LAYOUT and LAYOUT_TABLES in case it fails.

Pressing 'h' or '?' in any script will display the full list of shortcuts. Running the scripts
with --help will print shortcuts as well as command-line arguments.

# Philosophy

Primitive path meshes are computationally efficient ways of representing polymer melts and networks.
There are different levels of descriptions of a mesh and its response to deformation:

* Bead based descriptions: Like standard KG simulations. The repulsive interactions between beads
ensures that chains does not pass through each other. Beads on the same chain do not interact, such
that they act like one big thick rod of particles. Computationally simple, but also inefficient.

* Cylinder based descriptions: Replace the straight rod like sections of beads with a single cylinder
with radial WCA interactions. When two cylinders touch, kinks (aka WCA beads) are dynamically introduced to
represent the entanglement. Neighboring beads on the same chain have an attractive FENE like potential
within a contour length of 1.5. This means they tend to aggregate at entanglements as bead segments
ensuring a potential barrier preserving entanglements. Kinks/beads are dynamically
removed when 1) it does not interact with beads on other chains, 2) if it is only the outermost beads
in a segment of intramolecular beads, 3) the resulting straight line does not cause entanglement creation.
Computationally we get rid of all the dummy beads in the straight sections, but dynamically retain a
KG model at the entanglement points. The time step is comparable to KG as the rods and beads still interact via WCA.

* Finite-width vertex based descriptions: We retain the idea of a lines as finite sized cylinders, but
now a line has no interaction. When two lines touch (at a finite-distance) they are connected by a
vertex at the closest point. The vertex stores a vector of how they approached each other. The vector
has a constant length and acts as a rigid rod always turning to oppose the strand generated forces. For nearly straight
conformations this vector dynamics is unstable and needs stabilization tricks. The vertex is kept while
strands are in compression, but is brokwn when it is in tension. Thus compared to the descriptions above, the
entanglement dynamics is now approximated at the level of strand pairs. We explicitly need to deal
with (or approximate) collisions between different entanglements and cross-links. Computationally, this
vastly more complex than the models above, however, the level of coarse-graining also suggests that the
model would be faster to simulate.

* Zero-width vertex based descriptions. The lines are now zero width. Like the finite-width model collisions
(during a timestep) can be detected and entanglements dynamically introduced and removed. The stability
issue is resolved. As above we explicitly need to deal with all types of entanglement collisions, however
this should probably be the fastest model to simulate.

If we want to reconstitute the original bead based PPA mesh, then this is duable with the cylinder based
description, but only approximate with the vertex based descriptions. 


## Bead based mesh simulator

![Screenshot of beadppa.py.](beadppa.png)

Implementation of beadppa logic. Polymers are sequences of beads, the chain ends are placed on the surface of a spherical volume. The end beads can be moved, while the PPA contraction dynamics is continually running, thus you can see how an entanglement dynamically responds to its environment.

You can interactively grab and drag chain ends on the surface. Use shift and mouse-drag to the horizon of the sphere to move chain ends between the front and back surface of the sphere. The PPA dynamics minimize contour length by placing a bead at the average position of its two neighbors, while overlapping beads bead on different chains are shifted to avoid overlap. Some times this produces a jittering motion.

Use a add new chains to make arbitrary complicated entanglements. Grab the end of a bead and use left or right arrow to add or remove beads to the chain. You can also grab and press 'd' to delete a polymer. Some times chains will pass through each other, to avoid that add more beads along the contour. The beads are added between the end and the first internal bead. When you grow or shrink the volume (keys r/R), beads are inserted or removed, such that the entanglement is essentially unaffected. Several of the short-cut keys for kgview.py also works for beadppa.py. Run beadppa.py --help for complete information.

## lineppa.py  Finite size vertex mesh model

![Screenshot of lineppa.py.](lineppa.png)

An attempt at implementing lineppa logic, the script is based on beadppa and many of the shortcuts still apply. Polymers are now straight lines. As in beadppa the ends can be moved on the spherical surface, and chains can be added and removed. The left and right arrow now insert additional contour length at the corresponding chain end.

When two lines get within a minimum distance (MIN_DISTANCE) of each other an entanglement vertex is created. The idea is to match the point when two beads touch in a beadppa. A thin dotted line illustrates the two points on the lines spatially closest to each other just before the vertex is created. When an entanglement is created, the vector between the two closest points on the line segments are stored in the vertex.

Each line segment is modelled as a spring with an entropic spring constant k/L, where k is a system constant, and L is the contour length of that segment. Entanglements moves spatially according to the net force generated by the four entangled strands. Contour length can slip across entanglements, while the total contour length of a polymer is preserved.

The width of a line segment illustrates its tension, when you decrease the contour length of a line segment, you increase its tension thus transiently pulling the entanglement towards the end. After a while the contour length has equilibrated along the chain. A chain with low total contour length is "stiff" and thus impacts the position of its entanglements more than a chain with a large total contour length.

Currently, the colored vectors at vertices illustrate the net force produced by the strands. The orange vector is the vector
of closest approach of what would be the beads in a beadppa.


** todo ** : No support for cross-links or handling collisions between entanglements.

## cylinderppa.py  Finite size cylinder mesh model

![Screenshot of lineppa.py.](cylppa.png)


I have not yet looked into this. The vertex model is complicated because you can create composite vertices of entanglement and cross-link vertices, because lines have zero width, and implementing each of these (up to some level of complexity) is nasty. On the other hand the graph structure of the resulting mesh is very natural and coarse-grained description of the physics. 

In the cylinder model we take the diameter of beads serious, and reflect this in the diameter of the lines. Thus entanglement vertices are never created, but kinks are created along the cylinders where they need to have a kink due to another cylinder. In this case, cross-link and entanglement coupling is simple, since they are fat objects that can never overlap, and each kink moves with the net force acting on it but with the constraint that no point on two line segments gets within MIN_DISTANCE from each other. This requires modelling repulsive line-line interactions and introducing and removing kinks accordingly. 



## Rod and spring simulator

![Screenshot of rod_springs_sim.py](rod_springs_sim.png)

2D interactive simulator of a stiff rod connected to two springs. The springs generate a net force and torque on the rod, besides the spring interactions the dynamics adds translational and rotational friction. The sign of the torque in the dynamics can be flipped. With +1 sign, the rod aligns with the forces (tensile stress), while with -1 the rod aligns in the opposite direction (compressive stress).

Compared to the beadppa and lineppa, the rod represents the distance between the two beads being entangled, or the vector of closest approch between two lines, where the entanglement vertex is inserted. An entanglement is stable as long as the stress is compresive, while it should be removed when the stress is tensile.




## kgview.py

![Screenshot of kgview.py.](kgview.png)

kgview.py visualizes the detailed structure of KG dump files produced by LAMMPS, essentially it is a KG microscope.

Initially beads inside a spherical volume centered on 0,0,0 is shown with a certain radius. You can do 3D rotations and zoom of the conformation. The size of the volume shown can be controlled by r/R, the size of the beads with w/W. Display of beads is toggled with b/B while bonds are always shown. The arrow keys move the center position thus shifting which beads are shown. Shift arrow-up/down in and out of the conformation. Use kgview.py --help to see all the options.

You can store views (rotation, center position) with shift+ 1-9, and restore a view with 1-9. This creates smooth transitions between the stored views. Use m/M to start/stop recording movies. p to save a png image. s to save a dump file of what is inside the volume.

** Note ** : The script becomes very slow if you render all beads in the melt. In that case press 'r' to reduce the radius and number of beads shown until the code becomes responsive. Similar if you start the PPA dynamics by accident (e/E), which is very experimental. Finally, recording movies while manually rotating the view does not work very well.

