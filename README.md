# EntanglementSimulator

Three scripts to experiment with Entanglements.

##Dependencies

```
pip install pyvista numpy
pip install imageio imageio-ffmpeg   (movies)
```

##Before you run

Note that to capture keys, the keyboard layout should be set in the code.
Default value is KEYBOARD_LAYOUT = "nordic", other options are us, uk, de.

## kgview.py

![Screenshot of kgview.py.](kgview.png)

kgview.py visualizes KG dump files produces with LAMMPS, essentially it is a KG microscope. 

Initially beads inside a spherical volume centered on 0,0,0 is shown with a radius that is determined to start with approximately 10K beads. You can do 3D rotations and zoom of the conformation. The size of the volume shown can be controlled by r/R, the size of the beads with w/W. Display of beads is toggled with b/B while bonds are always shown. The arrow keys move the center position shown. Shift arrow-up/down in and out of the conformation. Use kgview.py --help to see all the options.

You can store views (rotation, position) with shift+ 1-9, and restore a view with 1-9.  Use m/M to start/stop recording movies.  and p to save a png image. s to save a dump file of what is inside the volume.

Warning: The script becomes very slow if you render all beads in the melt, or if you by accident start the PPA dynamics which is very experimental.

##beadppa.py

![Screenshot of beadppa.py.](beadppa.png)

Implementation of beadppa logic. Polymers are straight lines between end beads placed on the surface of a spherical volume. You can interactively grab and drag chain ends on the surface. Use shift and mouse-drag to the horizon of the sphere, to move chain ends between the front and back surface of the sphere. The dynamics minimize contour length by placing a bead at the average position of its two neighbors. The PPA distance constrain is implemented, such that bead on different chains are not allowed to overlap. Thus you can continually see how the entanglement responds when the chain ends are moved around.

Use a/k to add or kill chains to make arbitrary complicated entanglements. Grab the end of a bead and use left or right arrow to add or remove beads to the chain. The beads are added between the end and the first internal bead. Several of the short-cut keys for kgview.py also works for beadppa.py. Run beadppa.py --help for complete information.

##lineppa.py  Vertex model

![Screenshot of lineppa.py.](lineppa.png)

An attempt at implementing lineppa logic, the script is based on beadppa and many of the short cuts still apply. Polymers are now straight lines. As in beadppa the ends can be moved on the spherical surface, chains can be added and removed. The left and right arrow insert additional contour length at the corresponding chain end.

Each line segment is modelled as a spring with a spring constant k/L, where k is a system constant, and L is the contour length of that segment. When two lines collide an entanglement is introduced. Entanglements moves according to the net force applied to them. Contour length can slip across entanglements, while the total contour length of a polymer is preserved. The width of a line segment illustrates its tension.

The colored vectors at vertices illustrate the net force produced by the strands. The orange vector is the vector of closest approach of what would be the beads in a beadppa.

