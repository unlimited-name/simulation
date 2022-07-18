from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
from linecache import clearcache
from chroma.geometry import Geometry,Solid
from chroma.make import linear_extrude
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum, r7081hqe_photocathode

from optics import teflon_surface, detector_surface, black_surface
from optics import argon, cf4, quartz, vacuum, sapphire
import chroma.stl as stl
import chroma.make as make
import numpy as np

"""
Construction of a geometry. To use it in a simulation, call detector_construction().

Tactics:
separate the construction into steps: the meshes of geometry objects used are either imported from
Solidworks model, or constructed externally using python packages (or any other reliable software)
chroma integrates a pack called 'PyMesh', which is also used in its GDML interface.
PyMesh is a good choice but will add (unnecessary) complexity to chroma installation. One may skip 
all the PyMesh stuff and still can run the simulations.
Notice the chroma GDML interface is not properly using Pymesh for geometry import, so one should 
avoid using GDML as a interface between Geant4 and chroma. 

here we import the needed meshes in .stl format, and make fine adjustments to them. 
for meshes used multiple times and needs adjustion after import, distinct functions are written 
but they can simply be ignored. 

The properties of Materials and Surfaces are moved to optics.py....
"""

def rotation_matrix(thetax,thetay,thetaz):
    # the 3d rotation matrix in spherical coord., for simplicity in adding solids
    # the algorithm chroma uses: inner product(vertice, rot) + pos
    # this rotation matrix is designed for x-y-z axis order's rotation, following np.dot(vector, rot) format of use
    rx = np.array([[1,0,0], [0,np.cos(thetax),-1*np.sin(thetax)], [0,np.sin(thetax),np.cos(thetax)]])
    ry = np.array([[np.cos(thetay),0,np.sin(thetay)], [0,1,0], [-1*np.sin(thetay),0,np.cos(thetay)]])
    rz = np.array([[np.cos(thetaz),-1*np.sin(thetaz),0], [np.sin(thetaz),np.cos(thetaz),0], [0,0,1]])
    m = np.dot(np.dot(rx, ry), rz)
    return m


def SiPM():
    # create a simple square with length a at x-y plane
    # used as black sipm
    a = 0.0141
    a = a/2
    #vertice = np.array([[a,a,0],[-a,a,0],[-a,-a,0],[a,-a,0]])
    #triangle = np.array([[1,2,3],[3,4,1]])
    #mesh = Mesh(vertice, triangle)
    mesh = linear_extrude([-a,a,a,-a],[-a,-a,a,a], height=0.0001, center=(0,0,0))
    solid = Solid(mesh, cf4,cf4, black_surface)
    return solid

def detector_construction():
    g = Geometry(vacuum)
    pos0 = np.array([0,0,0])
    # this is the position of inner jar center relative to [0,0,0]
    # in my construction, I actually considered all the position relative to inner jar center
    pos1 = np.array([0,0,0.2225])
    # the distance between outer jar center and inner jar.

    # World
    World_mesh = make.cube(2)
    World_solid = Solid(World_mesh, cf4, cf4, black_surface)
    g.add_solid(World_solid)
    # treat the PV as the world, a 2*2*2 cube centered at (0,0,0)

    # sapphire viewpoint
    """ currently only 1 viewpoint is set, needs improvement
    """
    Sapphire_mesh = stl.mesh_from_stl('sapphire.stl')
    Sapphire_solid = Solid(Sapphire_mesh, sapphire, cf4, detector_surface)
    rot_sapphire = rotation_matrix(22.5*np.pi/180, 0, 0)
    h_cone = 0.060325
    pos_measure = np.array([(0.03750+0.02913+0.04603)/2+0.060325, 0, (0.34097+0.38421)/2])
    # The measured position of head cone center(bottom), to zero point
    # ==========================================check!=================================================================

    pos_sapphire = pos0 + pos_measure + np.array([h_cone*np.sin(22.5*np.pi/180),0,h_cone*np.cos(22.5*np.pi/180)])
    g.add_solid(Sapphire_solid, rot_sapphire, pos_sapphire)

    # head cones
    Head_mesh = stl.mesh_from_stl('head_cone.stl')
    Head_solid = Solid(Head_mesh, cf4, cf4, black_surface) 
    # normal vector / optical axis: 28.16 degree to z axis
    # measured data above. Use 22.5 the design data instead.
    rot_head1 = rotation_matrix(22.5*np.pi/180, 0, 0)
    pos_head1 = pos_measure + pos0
    g.add_solid(Head_solid, rot_head1, pos_head1)
    rot_head2 = rotation_matrix(22.5*np.pi/180, 0, np.pi/3)
    pos_head2 = np.dot(pos_measure, rot_head2) + pos0 
    g.add_solid(Head_solid, rot_head2, pos_head2)
    rot_head3 = rotation_matrix(22.5*np.pi/180, 0, 2*np.pi/3)
    pos_head3 = np.dot(pos_measure, rot_head3) + pos0 
    g.add_solid(Head_solid, rot_head3, pos_head3)

    # head reflectors
    """
    Href_mesh = stl.mesh_from_stl('head_ref.stl')
    Href_solid = Solid(Href_mesh, PTFE, CF4, shiny_surface)
    pos_href1 = pos_head1
    pos_href2 = np.dot(pos_measure, rotation_matrix(0,0,np.pi/3))
    pos_href3 = np.dot(pos_measure, rotation_matrix(0,0,2*np.pi/3))
    g.add_solid(Href_solid, rot_head1, pos_href1)
    g.add_solid(Href_solid, rot_head2, pos_href2)
    g.add_solid(Href_solid, rot_head3, pos_href3)
    """

    # dome reflectors
    # 8 pieces, thus treated in a for loop
    Dref_mesh = stl.mesh_from_stl('dome_ref.stl')
    Dref_solid = Solid(Dref_mesh, cf4, cf4, teflon_surface)
    dr_dref = np.sqrt(0.01776 ** 2 + (0.07264+0.060325) ** 2) # difference in x-y plane
    dz_dref = 0.28018 # measured position in z
    t_plate = (-90 + 14.82) *np.pi/180 # slope of plate, 14.82 degree
    for i in range(8):
        pos_dref = np.array([dr_dref*np.cos(i*np.pi/4), dr_dref*np.sin(i*np.pi/4), dz_dref]) + pos0
        rot_dref = rotation_matrix(t_plate, 0, i*np.pi/4) 
        # every plate is originally in x-y plane, then rotated along 
        g.add_solid(Dref_solid, rot_dref, pos_dref)
        g.add_solid(SiPM(), rot_dref, pos_dref)
        # adding a sipm at each plate's center

    # Outer Jar
    # includes 2 layers: inside and outside.
    Ojout_mesh = stl.mesh_from_stl('oj_out.stl')
    Ojin_mesh = stl.mesh_from_stl('oj_in.stl')
    Ojout_solid = Solid(Ojout_mesh, quartz, cf4)
    pos_ojout = pos0 - np.array([0,0,0.0001])
    # make a bit room for distinction. moved outer part a bit lower
    rot_oj = rotation_matrix(0,0,0)
    g.add_solid(Ojout_solid, rot_oj, pos_ojout)
    Ojin_solid = Solid(Ojin_mesh, argon, quartz)
    pos_ojin = pos0
    g.add_solid(Ojin_solid, rot_oj, pos_ojin)

    # Outer Jar reflector
    Oref_mesh = stl.mesh_from_stl('oj_ref.stl')
    Oref_solid = Solid(Oref_mesh, cf4, cf4, teflon_surface)
    pos_oref = pos0
    g.add_solid(Oref_solid, rotation_matrix(0,0,0), pos_oref)

    # Inner Jar
    Ijout_mesh = stl.mesh_from_stl('ij_out.stl')
    Ijin_mesh = stl.mesh_from_stl('ij_in.stl')
    Ijout_solid = Solid(Ijout_mesh, quartz, argon)
    pos_ijout = pos0 - np.array([0,0,0.0001])
    rot_ij = rotation_matrix(0,0,0)
    g.add_solid(Ijout_solid, rot_ij, pos_ijout)
    Ijin_solid = Solid(Ijin_mesh, cf4, quartz)
    pos_ijin = pos0
    g.add_solid(Ijin_solid, rot_ij, pos_ijin)

    # Inner jar reflector
    Iref_mesh = stl.mesh_from_stl('ij_ref.stl')
    Iref_solid = Solid(Iref_mesh, cf4, cf4, teflon_surface)
    pos_iref = pos0
    g.add_solid(Iref_solid, rotation_matrix(0,0,0), pos_iref)

    # adding sipms onto reflectors
    # oj reflector
    dz_ojsipm = 0.2225 /4
    oj_r = 0.12235
    for i in range(8):
        pos_ojsipm = np.array([oj_r, 0, 0]) + pos0
        rot_ojsipm = rotation_matrix(0, 0, np.pi/4*i)
        g.add_solid(SiPM(), rot_ojsipm, pos_ojsipm)
        pos_ojsipm = pos_ojsipm + np.array([0,0,dz_ojsipm])
        g.add_solid(SiPM(), rot_ojsipm, pos_ojsipm)
        pos_ojsipm = pos_ojsipm + np.array([0,0,dz_ojsipm])
        g.add_solid(SiPM(), rot_ojsipm, pos_ojsipm)
        pos_ojsipm = pos_ojsipm + np.array([0,0,dz_ojsipm])
        g.add_solid(SiPM(), rot_ojsipm, pos_ojsipm)
    # ij reflector

    return g




# Test geometry copied from chroma.doc
def build_pd(size, glass_thickness):
    """Returns a simple photodetector Solid. The photodetector is a cube of
    size `size` constructed out of a glass envelope with a photosensitive
    face on the inside of the glass envelope facing up."""
    # outside of the glass envelope
    outside_mesh = make.cube(size)
    # inside of the glass envelope
    inside_mesh = make.cube(size-glass_thickness*2)

    # outside solid with water on the outside, and glass on the inside
    outside_solid = Solid(outside_mesh,glass,water)    

    # now we need to determine the triangles which make up
    # the top face of the inside mesh, because we are going to place
    # the photosensitive surface on these triangles
    # do this by seeing which triangle centers are at the maximum z
    # coordinate
    z = inside_mesh.get_triangle_centers()[:,2]
    top = z == max(z)

    # see np.where() documentation
    # Here we make the photosensitive surface along the top face of the inside
    # mesh. The rest of the inside mesh is perfectly absorbing.
    inside_surface = np.where(top,r7081hqe_photocathode,black_surface)
    inside_color = np.where(top,0x00ff00,0x33ffffff)

    # construct the inside solid
    inside_solid = Solid(inside_mesh,vacuum,glass,surface=inside_surface,
                         color=inside_color)

    # you can add solids and meshes!
    return outside_solid + inside_solid

def iter_box(nx,ny,nz,spacing):
    """Yields (position, direction) tuple for a series of points along the
    boundary of a box. Will yield nx points along the x axis, ny along the y
    axis, and nz along the z axis. `spacing` is the spacing between the points.
    """
    dx, dy, dz = spacing*(np.array([nx,ny,nz])-1)

    # sides
    for z in np.linspace(-dz/2,dz/2,nz):
        for x in np.linspace(-dx/2,dx/2,nx):
            yield (x,-dy/2,z), (0,1,0)
        for y in np.linspace(-dy/2,dy/2,ny):
            yield (dx/2,y,z), (-1,0,0)
        for x in np.linspace(dx/2,-dx/2,nx):
            yield (x,dy/2,z), (0,-1,0)
        for y in np.linspace(dy/2,-dy/2,ny):
            yield (-dx/2,y,z), (1,0,0)

    # bottom and top
    for x in np.linspace(-dx/2,dx/2,nx)[1:-1]:
        for y in np.linspace(-dy/2,dy/2,ny)[1:-1]:
            # bottom
            yield (x,y,-dz/2), (0,0,1)
            # top
            yield (x,y,dz/2), (0,0,-1)

def test_geometry():
    size = 100
    glass_thickness = 10
    nx, ny, nz = 20, 20, 20
    spacing = size*2
    """Returns a cubic detector made of cubic photodetectors."""
    g = Geometry(water)
    for pos, dir in iter_box(nx,ny,nz,spacing):
        # convert to arrays
        pos, dir = np.array(pos), np.array(dir)

        # we need to figure out what rotation matrix to apply to each
        # photodetector so that the photosensitive surface will be facing
        # `dir`.
        if tuple(dir) == (0,0,1):
            rotation = None
        elif tuple(dir) == (0,0,-1):
            rotation = make_rotation_matrix(np.pi,(1,0,0))
        else:
            rotation = make_rotation_matrix(np.arccos(dir.dot((0,0,1))),
                                            np.cross(dir,(0,0,1)))
        # add the photodetector
        g.add_solid(build_pd(size,glass_thickness),rotation=rotation,
                    displacement=pos)

    world = Solid(make.box(spacing*nx,spacing*ny,spacing*nz),water,vacuum,
                  color=0x33ffffff)
    g.add_solid(world)

    return g