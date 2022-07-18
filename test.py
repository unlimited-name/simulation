from chroma import make, view
from chroma.geometry import Solid, Geometry
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum, r7081hqe_photocathode
from chroma.demo.optics import black_surface
import numpy as np

from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.event import Photons
from chroma.loader import load_bvh


def random_vector():
    # generate a normalized 3d random vector, used for polarization
    phi = 2*np.pi * random.random()
    theta = np.pi * random.random()
    vec = np.array((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(phi)))
    return vec

def random_vector_restricted(theta_restrict, phi_restrict):
    phi = phi_restrict * random.random()
    theta = theta_restrict * random.random()
    vec = np.array((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(phi)))
    return vec

def rotation_matrix(thetax,thetay,thetaz):
    #the 3d rotation matrix in spherical coord., for simplicity in adding solids
    #the algorithm chroma uses: inner product(vertice, rot) + pos
    rx = np.array([[1,0,0], [0,np.cos(thetax),-1*np.sin(thetax)], [0,np.sin(thetax),np.cos(thetax)]])
    ry = np.array([[np.cos(thetay),0,np.sin(thetay)], [0,1,0], [-1*np.sin(thetay),0,np.cos(thetay)]])
    rz = np.array([[np.cos(thetaz),-1*np.sin(thetaz),0], [np.sin(thetaz),np.cos(thetaz),0], [0,0,1]])
    m = np.dot(np.dot(rx, ry), rz)
    return m

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

size = 1
glass_thickness = 0.1
nx, ny, nz = 2, 2, 2
spacing = size*2

def build_detector():
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


def point_source(wavelength, n, pos, dir = (0,0,1)):
    # n photons in total emitting from points (pos), with angles restricted
    # accepts multiple points input, they will be evenly rotated
    # dir is the direction of source - the default is the direction of Z axis here

    n = int(n)
    pos = np.array(pos)
    dir = np.array(dir)
    # restrict angles
    phi_restrict = 2*np.pi
    theta_restrict = 90 /180*np.pi

    pos_photon = np.array([pos[-2],pos[-1]])
    dir_photon = np.array([random_vector_restricted(theta_restrict, phi_restrict),random_vector_restricted(theta_restrict, phi_restrict)])
    pol_photon = np.array([np.cross(dir_photon[-2],random_vector()),np.cross(dir_photon[-1],random_vector())])

    while len(pos_photon)<n:
        for i in range(len(pos)):
            pos_photon = np.append(pos_photon, [pos[i]], axis=0)
            dirvec = random_vector_restricted(theta_restrict, phi_restrict)
            dir_photon = np.append(dir_photon, [dirvec+dir], axis=0)
            pol_photon = np.append(pol_photon, [np.cross(dir_photon[-1],random_vector())], axis=0)
            if (len(pos_photon)==n): 
                break

    wavelengths = np.repeat(wavelength,n)
    return Photons(pos_photon, dir_photon, pol_photon, wavelengths)

def LED_ring(wavelength, n, pos, dir):
    # n photons emitting photons evenly from 24 LEDs arranged in a ring
    # use pos,dir as the position of center and the normal vector of the ring.
    n = int(n)
    pos = np.array(pos) # a list of 3d vectors
    dir = np.array(dir) # (x,y,z) array
    ring = (0.06985+0.12065/2)/2
    # obtain the angles of normal vector
    if dir[2]==0:
        theta = np.pi /2
    else:
        theta = np.arctan(np.sqrt(dir[0]*dir[0]+dir[1]*dir[1])/dir[2])

    if dir[0]==0:
        phi = np.pi /2
    else: 
        phi = np.arctan(dir[1]/dir[0])
    
    # position matrices for 24 LEDs
    pos_led = []
    delta = np.pi/12
    for i in range(24):
        x = ring * np.cos(i*delta + phi)
        y = ring * np.sin(i*delta + phi)
        z = 0
        pos_led.append(np.dot(np.array([x,y,z]), rotation_matrix(theta, 0, phi)) + pos)

    pos_led = np.array(pos_led)
    dir_led = np.dot(dir, rotation_matrix(theta, 0, phi))
    return point_source(wavelength, n, pos_led, dir_led)

def triple_LED_ring(wavelength, n, pos, dir):
    # the function used for 3 LEDs, the input parameters are the same with LED1
    # but with LED1 rotated 3/pi each time
    n = int(n/24/3)
    ring = (0.06985+0.12065/2)/2
    # obtain the angles of normal vector
    phi = np.arctan(dir[1]/dir[0])
    theta = np.arctan(np.sqrt(dir[0]*dir[0]+dir[1]*dir[1])/dir[2])
    pos_led = []
    delta = np.pi/12
    for i in range(24):
        x = ring * np.cos(i*delta + phi)
        y = ring * np.sin(i*delta + phi)
        z = 0
        pos_led.append(np.dot(rotation_matrix(theta, 0, phi), np.array([x,y,z])) + pos)

    pos_led = np.array(pos_led)
    np.concatenate((pos_led, np.dot(rotation_matrix(0,0,np.pi/3), pos_led)),axis=0)
    np.concatenate((pos_led, np.dot(rotation_matrix(0,0,2*np.pi/3), pos_led)),axis=0)

    return point_source(wavelength, n, pos_led, dir)

def photon_bomb(n,wavelength,pos):
    pos = np.tile(pos,(n,1))
    dir = uniform_sphere(n)
    pol = np.cross(dir,uniform_sphere(n))
    wavelengths = np.repeat(wavelength,n)
    return Photons(pos,dir,pol,wavelengths)



if __name__ == '__main__':

    g = build_detector()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # write it to a root file
    from chroma.io.root import RootWriter
    f = RootWriter('point.root')

    # sim.simulate() always returns an iterator even if we just pass
    # a single photon bomb
    for ev in sim.simulate([point_source(400,1000,(0,0,0))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):
        # write the python event to a root file
        f.write_event(ev)

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)

    f.close()

    f2 = RootWriter('led.root')

    # sim.simulate() always returns an iterator even if we just pass
    # a single photon bomb
    for ev in sim.simulate([LED_ring(400,1000,(0,0,0),(0,0,1))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):
        # write the python event to a root file
        f.write_event(ev)

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)

    f2.close()
