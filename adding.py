from chroma import make
from chroma.geometry import Solid, Geometry
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum, r7081hqe_photocathode
from chroma.demo.optics import black_surface, shiny_surface
from chroma.sample import uniform_sphere
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.loader import load_bvh
from chroma.generator import vertex
import chroma.stl as stl
import numpy as np
import pandas as pd
import datetime
import sys

def rotation_matrix(thetax,thetay,thetaz):
    # the 3d rotation matrix in spherical coord., for simplicity in adding solids
    # the algorithm chroma uses: inner product(vertice, rot) + pos
    # this rotation matrix is designed for x-y-z axis order's rotation
    rx = np.array([[1,0,0], [0,np.cos(thetax),-1*np.sin(thetax)], [0,np.sin(thetax),np.cos(thetax)]])
    ry = np.array([[np.cos(thetay),0,np.sin(thetay)], [0,1,0], [-1*np.sin(thetay),0,np.cos(thetay)]])
    rz = np.array([[np.cos(thetaz),-1*np.sin(thetaz),0], [np.sin(thetaz),np.cos(thetaz),0], [0,0,1]])
    m = np.dot(np.dot(rx, ry), rz)
    return m

def photon_bomb(n,wavelength,pos):
    pos = np.tile(pos,(n,1))
    dir = uniform_sphere(n)
    pol = np.cross(dir,uniform_sphere(n))
    wavelengths = np.repeat(wavelength,n)
    return Photons(pos,dir,pol,wavelengths)

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

def build_detector(mode=0):
    """Returns a cubic detector made of cubic photodetectors.
    
    used for testing, mode = 0,1,2,3,4....
    mode = 0: add nothing
    mode = 1: inner jar only
    mode = 2: outer jar
    mode = 3: IJ and OJ reflectors
    mode = 4: top reflectors
    mode = 5: cone + detector"""
    size = 0.2
    glass_thickness = 0.01
    nx, ny, nz = 2, 2, 2
    spacing = size*2
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
    # a .4*.4*.4 world with water
    # adding layers with mode number

    pos0 = np.array([0,0,-0.1])
    mode = int(mode)
    if (mode>0):
        Ijout_mesh = stl.mesh_from_stl('ij_out.stl')
        Ijin_mesh = stl.mesh_from_stl('ij_in.stl')
        Ijout_solid = Solid(Ijout_mesh, glass, water)
        pos_ijout = pos0 - np.array([0,0,0.0001])
        rot_ij = rotation_matrix(0,0,0)
        g.add_solid(Ijout_solid, rot_ij, pos_ijout)
        Ijin_solid = Solid(Ijin_mesh, water, glass)
        pos_ijin = pos0
        g.add_solid(Ijin_solid, rot_ij, pos_ijin)

    if (mode>1):
        Ojout_mesh = stl.mesh_from_stl('oj_out.stl')
        Ojin_mesh = stl.mesh_from_stl('oj_in.stl')
        Ojout_solid = Solid(Ojout_mesh, glass, water)
        pos_ojout = pos0 - np.array([0,0,0.0001])
     # make a bit room for distinction. moved outer part a bit lower
        rot_oj = rotation_matrix(0,0,0)
        g.add_solid(Ojout_solid, rot_oj, pos_ojout)
        Ojin_solid = Solid(Ojin_mesh, water, glass)
        pos_ojin = pos0
        g.add_solid(Ojin_solid, rot_oj, pos_ojin)

    if (mode>2):
        Oref_mesh = stl.mesh_from_stl('oj_ref.stl')
        Oref_solid = Solid(Oref_mesh, water, water, shiny_surface)
        pos_oref = pos0
        g.add_solid(Oref_solid, rotation_matrix(0,0,0), pos_oref)
        Iref_mesh = stl.mesh_from_stl('ij_ref.stl')
        Iref_solid = Solid(Iref_mesh, water, water, shiny_surface)
        pos_iref = pos0
        g.add_solid(Iref_solid, rotation_matrix(0,0,0), pos_iref)

    if (mode>3):
        # dome reflectors
        # 8 pieces, thus treated in a for loop
        Dref_mesh = stl.mesh_from_stl('dome_ref.stl')
        Dref_solid = Solid(Dref_mesh, water, water, shiny_surface)
        dr_dref = np.sqrt(0.01776 ** 2 + (0.07264+0.060325) ** 2) # difference in x-y plane
        dz_dref = 0.28018 # measured position in z
        t_plate = - 90 + 14.82 # slope of plate, 14.82 degree
        for i in range(8):
            pos_dref = np.array([dr_dref*np.cos(i*np.pi/4), dr_dref*np.sin(i*np.pi/4), dz_dref]) + pos0
            rot_dref = rotation_matrix(t_plate*np.pi/180, 0, i*np.pi/4) 
            # every plate is originally in x-y plane, then rotated along 
            g.add_solid(Dref_solid, rot_dref, pos_dref)
            # g.add_solid(SiPM(), rot_dref, pos_dref)
            # adding a sipm at each plate's center
    
    if (mode>4):
        Sapphire_mesh = stl.mesh_from_stl('sapphire.stl')
        Sapphire_solid = Solid(Sapphire_mesh, glass, water)
        rot_sapphire = rotation_matrix(-22.5*np.pi/180, 0, 0)
        h_cone = 0.060325
        pos_measure = np.array([(0.03750+0.02913+0.04603)/2+0.060325, 0, (0.34097+0.38421)/2])
        pos_sapphire = pos0 + pos_measure + np.array([h_cone*np.sin(22.5*np.pi/180),0,h_cone*np.cos(22.5*np.pi/180)])
        g.add_solid(Sapphire_solid, rot_sapphire, pos_sapphire)
        # head cones
        Head_mesh = stl.mesh_from_stl('head_cone.stl')
        Head_solid = Solid(Head_mesh, water, water, black_surface) 
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

    return g

def simulate(mode = 0):
    g = build_detector(mode)
    g.flatten()
    g.bvh = load_bvh(g)
    sim = Simulation(g)

    namestr = 'mode'+str(mode)
    position_list = []
    length_of_batch = 1000
    for i in range(1):
        for j in range(1000):
            for ev in sim.simulate([photon_bomb(length_of_batch,400,(0,0,0))],
                           keep_photons_beg=False,keep_photons_end=True,
                           run_daq=False,max_steps=20):
                detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                detected_index = np.arange(length_of_batch)[detected]
                position_list.append(pd.DataFrame(ev.photons_end.pos[detected], index = detected_index))
    
    position_full = pd.concat(position_list)
    namestr = namestr + '_position.csv'
    position_full.to_csv(namestr)


if __name__ == '__main__':
    # testing: try to add layers of geometry at a time, and run a simulation of ~~ photons
    # ti = pd.to_datetime(datetime.datetime.now())
    ti = pd.to_datetime(datetime.datetime.now())
    if len(sys.argv)==1:
        mode = 6
    else:
        mode = sys.argv[1] + 1
    
    for i in range(mode):
        simulate(i)
    
    tf = datetime.datetime.now()
    dt = tf - ti

    print("The total time cost: ")
    print(dt)