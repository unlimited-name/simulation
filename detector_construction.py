from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
from linecache import clearcache
from chroma.geometry import Geometry,Solid,Mesh,Material,Surface
from chroma.make import linear_extrude
import chroma.stl as stl
import chroma.loader as loader
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
"""

def rotation_matrix(thetax,thetay,thetaz):
    #the 3d rotation matrix in spherical coord., for simplicity in adding solids
    #the algorithm chroma uses: inner product(vertice, rot) + pos
    rx = np.array([[1,0,0], [0,np.cos(thetax),-1*np.sin(thetax)], [0,np.sin(thetax),np.cos(thetax)]])
    ry = np.array([[np.cos(thetay),0,np.sin(thetay)], [0,1,0], [-1*np.sin(thetay),0,np.cos(thetay)]])
    rz = np.array([[np.cos(thetaz),-1*np.sin(thetaz),0], [np.sin(thetaz),np.cos(thetaz),0], [0,0,1]])
    m = np.dot(np.dot(rx, ry), rz)
    return m

# Materials
"""
    list of optical properties:
        self.refractive_index = None
        self.absorption_length = None
        self.scattering_length = None
        self.scintillation_spectrum = None
        self.scintillation_light_yield = None
        self.scintillation_waveform = None
        self.scintillation_mod = None
        self.comp_reemission_prob = []
        self.comp_reemission_wvl_cdf = []
        self.comp_reemission_times = []
        self.comp_reemission_time_cdf = []
        self.comp_absorption_length = []
        self.density = 0.0 # g/cm^3
        self.composition = {} # by mass
"""
vacuum = Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)

CF4 = Material('CF4')
CF4.set('refractive_index', 1.0004823)

SSteel = Material('SSteel')
#SSteel.set('absorption_length',np.array([(850, 1e-6)]))

# these 2 are not actually used
PTFE = Material('PTFE')
sapphire = Material('sapphire')

quartz = Material('quartz')
quartz.set('refractive_index', 1.49)

LAr = Material('LAr')
LAr.set('refractive_index', 2.1)

# surfaces
black_surface = Surface('black_surface')
black_surface.set('absorb', 1)

shiny_surface = Surface('shiny_surface')
shiny_surface.set('reflect_diffuse', 1)

detector_surface = Surface('detector_surface')
detector_surface.detect = np.array([(850, 1)])
detector_surface.set('absorb',1)

"""
for more parameters and details concerning optics, refer to demo/optics
"""

def SiPM():
    # create a simple square with length a at x-y plane
    # used as black sipm
    a = 0.0141
    a = a/2
    #vertice = np.array([[a,a,0],[-a,a,0],[-a,-a,0],[a,-a,0]])
    #triangle = np.array([[1,2,3],[3,4,1]])
    #mesh = Mesh(vertice, triangle)
    mesh = linear_extrude([-a,a,a,-a],[-a,-a,a,a], height=0.0001, center=(0,0,0))
    solid = Solid(mesh, CF4,CF4, black_surface)
    return solid

def detector_construction():
    g = Geometry(vacuum)
    pos0 = np.array([0,0,-1])
    # this is the position of inner jar center relative to [0,0,0]
    # in my construction, I actually considered all the position relative to inner jar center
    pos1 = np.array([0,0,0.2225])
    # the distance between outer jar center and inner jar.

    # World
    World_mesh = make.cube(2)
    World_solid = Solid(World_mesh, CF4, vacuum)
    g.add_solid(World_solid)
    # treat the PV as the world, a 2*2*2 cube centered at (0,0,0)

    # sapphire viewpoint
    """ currently only 1 viewpoint is set, needs improvement
    """
    Sapphire_mesh = stl.mesh_from_stl('sapphire.stl')
    Sapphire_solid = Solid(Sapphire_mesh, sapphire, CF4, detector_surface)
    rot_sapphire = rotation_matrix(-22.5*np.pi/180, 0, 0)
    h_cone = 0.060325
    pos_measure = np.array([(0.03750+0.02913+0.04603)/2+0.060325, 0, (0.34097+0.38421)/2])
    # The measured position of head cone center(bottom), to zero point
    # ==========================================check!=================================================================

    pos_sapphire = pos0 + pos_measure + np.array([h_cone*np.sin(22.5*np.pi/180),0,h_cone*np.cos(22.5*np.pi/180)])
    g.add_solid(Sapphire_solid, rot_sapphire, pos_sapphire)

    # head cones
    Head_mesh = stl.mesh_from_stl('head_cone.stl')
    Head_solid = Solid(Head_mesh, CF4, CF4, black_surface) 
    # normal vector / optical axis: 28.16 degree to z axis
    # measured data above. Use 22.5 the design data instead.
    rot_head1 = rotation_matrix(-22.5*np.pi/180, 0, 0)
    pos_head1 = pos_measure + pos0
    g.add_solid(Head_solid, rot_head1, pos_head1)
    rot_head2 = rotation_matrix(-22.5*np.pi/180, 0, np.pi/3)
    pos_head2 = np.dot(pos_measure, rot_head2) + pos0 
    g.add_solid(Head_solid, rot_head2, pos_head2)
    rot_head3 = rotation_matrix(-22.5*np.pi/180, 0, 2*np.pi/3)
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
    Dref_solid = Solid(Dref_mesh, PTFE, CF4, shiny_surface)
    dr_dref = np.sqrt(0.01776 ** 2 + (0.07264+0.060325) ** 2) # difference in x-y plane
    dz_dref = 0.28018 # measured position in z
    t_plate = 90 - 14.82 # slope of plate, 14.82 degree
    for i in range(8):
        pos_dref = np.array([dr_dref*np.cos(i*np.pi/4), dr_dref*np.sin(i*np.pi/4), dz_dref]) + pos0
        rot_dref = rotation_matrix(t_plate*np.pi/180, 0, i*np.pi/4)
        g.add_solid(Dref_solid, rot_dref, pos_dref)
        g.add_solid(SiPM(), rot_dref, pos_dref)

    # Outer Jar
    Ojout_mesh = stl.mesh_from_stl('oj_out.stl')
    Ojin_mesh = stl.mesh_from_stl('oj_in.stl')
    Ojout_solid = Solid(Ojout_mesh, quartz, CF4)
    pos_ojout = pos0 - np.array([0,0,0.0001])
    rot_oj = rotation_matrix(0,0,0)
    g.add_solid(Ojout_solid, rot_oj, pos_ojout)
    Ojin_solid = Solid(Ojin_mesh, LAr, quartz)
    pos_ojin = pos0
    g.add_solid(Ojin_solid, rot_oj, pos_ojin)

    # Outer Jar reflector
    Oref_mesh = stl.mesh_from_stl('oj_ref.stl')
    Oref_solid = Solid(Oref_mesh, PTFE, CF4, shiny_surface)
    pos_oref = pos0
    g.add_solid(Oref_solid, rotation_matrix(0,0,0), pos_oref)

    # Inner Jar
    Ijout_mesh = stl.mesh_from_stl('ij_out.stl')
    Ijin_mesh = stl.mesh_from_stl('ij_in.stl')
    Ijout_solid = Solid(Ijout_mesh, quartz, LAr)
    pos_ijout = pos0 - np.array([0,0,0.0001])
    rot_ij = rotation_matrix(0,0,0)
    g.add_solid(Ijout_solid, rot_ij, pos_ijout)
    Ijin_solid = Solid(Ijin_mesh, CF4, quartz)
    pos_ijin = pos0
    g.add_solid(Ijin_solid, rot_ij, pos_ijin)

    # Inner jar reflector
    Iref_mesh = stl.mesh_from_stl('ij_ref.stl')
    Iref_solid = Solid(Iref_mesh, PTFE, CF4, shiny_surface)
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