from chroma import make, view
from chroma.geometry import Solid, Geometry
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum, r7081hqe_photocathode
from chroma.demo.optics import black_surface
import numpy as np

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

size = 100
glass_thickness = 10

nx, ny, nz = 20, 20, 20

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

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma.sample import uniform_sphere
    from chroma.event import Photons
    from chroma.loader import load_bvh

    g = build_detector()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # photon bomb from center
    def photon_bomb(n,wavelength,pos):
        pos = np.tile(pos,(n,1))
        dir = uniform_sphere(n)
        pol = np.cross(dir,uniform_sphere(n))
        wavelengths = np.repeat(wavelength,n)
        return Photons(pos,dir,pol,wavelengths)

    # write it to a root file
    from chroma.io.root import RootWriter
    f = RootWriter('test.root')

    # sim.simulate() always returns an iterator even if we just pass
    # a single photon bomb
    for ev in sim.simulate([photon_bomb(10,400,(0,0,0))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):
        # write the python event to a root file
        f.write_event(ev)

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)

    f.close()
