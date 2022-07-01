import pymesh
import numpy as np

def generate_circular_vertex(theta,r,steps):
    # generate vertices on a circle, centering (0,0,0)
    x = r * np.cos(np.linspace(0,theta/180*np.pi,steps))
    y = r * np.sin(np.linspace(0,theta/180*np.pi,steps))
    z = np.zeros((1,len(x)))[0]
    points = np.transpose([x,y,z])
    return points

def generate_linear_vertex(start,end,steps):
    # 3d linear space
    x = np.linspace(start[0],end[0],steps)
    y = np.linspace(start[1],end[1],steps)
    z = np.linspace(start[2],end[2],steps)
    points = np.transpose([x,y,z])
    return points

def displacement(vertices, dx=0, dy=0, dz=0):
    # x-y-z dispacement for a numpy array
    points = vertices
    cache = points[:] + np.array([dx,dy,dz])
    return cache

def distance(vertice, x,y,z):
    # get the distance (squared) to point x,y,z
    p = np.array([x,y,z])
    delta = vertice - p
    d = np.dot(delta,delta)
    return d

def sapphire_viewpoint():
    # pick out the sensitive surface
    # mesh object: the center (0,0,0) being the lower surfece of the column
    d = 0.03647
    h = 0.001
    mesh = pymesh.generate_cylinder([0,0,0], [0,0,h], d/2, d/2, num_segments=64)
    return mesh

#head_ref.stl
def head_reflector():
    
    L = 0.15239
    D = 0.12065
    d = 0.02476
    vertices = generate_circular_vertex(64,L,50)
    endpoint = vertices[len(vertices)-1]
    vertices = np.concatenate((vertices,generate_linear_vertex(endpoint,[0,0,0],100)))
    vertices = np.concatenate((vertices, np.dot(vertices, np.diag([1,-1,1])))) # mirror in x-z plane
    # the bigger sector shape, centered at (0,0,0)
    hole = displacement(generate_circular_vertex(360,D/2,100), d+D/2, 0, 0)
    # the smaller circular hole inside, centered at x = d+D/2
    vertices = np.concatenate((vertices, hole))
    tri = pymesh.triangle()
    tri.max_num_steiner_points = 100
    tri.points = vertices
    tri.verbosity = 1
    tri.run()
    mesh = tri.mesh
    # the initial pymesh triangle method. This mesh has failed in creating the hole

    vertices = mesh.vertices
    faces = mesh.faces
    hole_index = []
    for i in vertices:
        hole_index.append(distance(i,d+D/2,0,0) < ((D*D/4)+0.001) )
    hole_index = np.arange(len(vertices))[np.array(hole_index)]
    # exclude all the triangles in the hole
    index = [] # label these triangles
    for i in faces:
        a = i[0] in hole_index
        b = i[1] in hole_index
        c = i[2] in hole_index
        index.append(not ((a and b) or (b and c) or (c and a)))

    index_array = np.array(index)
    faces_new = faces[index_array]
    mesh_new = pymesh.meshio.form_mesh(vertices, faces_new, voxels=None)
    return mesh_new

# head_cone.stl
def head_cone():
    # mesh object: the center (0,0,0) being the lower plane of the tube
    D = 0.12065
    d = 0.0508
    h = 0.060325
    mesh = pymesh.generate_tube([0,0,0], [0,0,h], D/2, d/2, D/2, d/2, num_segments=64, with_quad=False)
    return mesh

# dome_ref.stl
def dome_reflector():
    # mesh object: the center (0,0,0) being the center of the object, lies in x-y plane 
    d = 0.09805/2
    D = 0.11294/2
    L = 0.09119
    h = np.sqrt(L*L-(D-d)*(D-d))
    vertices = np.array([[D,h/2,0],[-D,h/2,0],[d,-h/2,0],[-d,-h/2,0]])
    tri = pymesh.triangle()
    tri.points = vertices
    tri.verbosity = 1
    tri.run()
    mesh = tri.mesh
    return mesh

# oj_ref.stl
def oj_ref():
    # mesh object: center (0,0,0) being the bottom
    r = 0.12235
    h = 0.22225
    mesh = pymesh.generate_tube([0,0,0], [0,0,h], r, r, r, r, num_segments=64, with_quad=False)
    return mesh

# ij_ref.stl
def ij_ref():
    # mesh object: center (0,0,0) being the bottom 
    R = 0.092075
    d = 0.0762
    L = 0.05771
    h = np.sqrt(L*L - (R-d/2)*(R-d/2))
    mesh = pymesh.generate_cylinder([0,0,0], [0,0,h], R, d/2, num_segments=64)
    return mesh

def sipm():
    # no longer used. replaced by newly written function in detector construction
    a = 0.0141
    mesh = pymesh.generate_box_mesh([0, 0], [a, a], num_samples=1, keep_symmetry=False, subdiv_order=0, using_simplex=True)
    return mesh

def mesh_grid(grid):
    begin = grid[:-1].flatten()
    end = grid[1:].flatten()
    begin_roll = np.roll(grid[:-1],-1,1).flatten()
    end_roll = np.roll(grid[1:],-1,1).flatten()
    
    mesh = np.empty(shape=(2*len(begin),3), dtype=begin.dtype)
    mesh[:len(begin),0] = begin
    mesh[:len(begin),1] = end
    mesh[:len(begin),2] = end_roll
    mesh[len(begin):,0] = begin
    mesh[len(begin):,1] = end_roll
    mesh[len(begin):,2] = begin_roll
    return mesh

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

def normalize(x):
    "Returns unit vectors in the direction of `x`."
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (x/norm(x)[:,np.newaxis]).squeeze()

def rotate(x, phi, n):
    """
    Rotate an array of points `x` through an angle phi counter-clockwise
    around the axis `n` (when looking towards +infinity).
    """
    n = normalize(n)
    x = np.atleast_2d(x)
    phi = np.atleast_1d(phi)

    return (x*np.cos(phi)[:,np.newaxis] + n*np.dot(x,n)[:,np.newaxis]*(1-np.cos(phi)[:,np.newaxis]) + np.cross(x,n)*np.sin(phi)[:,np.newaxis]).squeeze()

def rotate_extrude(x, y, nsteps=64):
    """
    Return the solid mesh formed by extruding the profile defined by the x and
    y points `x` and `y` around the y axis.

    .. note::
        The path traced by the points `x` and `y` should go counter-clockwise,
        otherwise the mesh will be inside out.

    Example:
        >>> # create a bipyramid
        >>> m = rotate_extrude([0,1,0], [-1,0,1], nsteps=4)
    """
    if len(x) != len(y):
        raise Exception('`x` and `y` arrays must have the same length.')

    points = np.array([x,y,np.zeros(len(x))]).transpose()

    steps = np.linspace(0, 2*np.pi, nsteps, endpoint=False)
    vertices = np.vstack([rotate(points,angle,(0,-1,0)) for angle in steps])
    triangles = mesh_grid(np.arange(len(vertices)).reshape((len(steps),len(points))).transpose()[::-1])
    mesh = pymesh.meshio.form_mesh(vertices, triangles, voxels=None)

    return mesh

def draw_a_circle(center,angle,r,theta):
    steps = 64
    x = r * np.cos(np.linspace(angle, angle+theta, steps)) + center[0]
    y = r * np.sin(np.linspace(angle, angle+theta, steps)) + center[1]
    return [x,y]

def oj_out():
    # mesh object: center (0,0,0) being the bottom
    h = 0.22225
    r = 0.12
    r1 = 0.04
    t1 = 66.4/180*np.pi
    c1 = [r-r1,h]
    r2 = 0.24
    t2 = 47.2/180*np.pi/2
    c2 = [0,h-r2*np.cos(t2)+r1*np.sin(t1)]
    # 2 lines, 2 curves. All suppose sitting at z = 0 (thus, starting from (0,0))
    X = np.array([0,r,r])
    Y = np.array([0,0,h])
    circle = draw_a_circle(c1, 0, r1, t1)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))
    circle = draw_a_circle(c2, np.pi/2-t2, r2, t2)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))

    return rotate_extrude(X, Y, nsteps=64)

def oj_in():
    # mesh object: center (0,0,0) being the bottom
    h = 0.22225
    r = 0.115
    r1 = 0.035
    t1 = 66.4/180*np.pi
    c1 = [r-r1,h]
    r2 = 0.235
    t2 = 47.2/180*np.pi/2
    c2 = [0,h-r2*np.cos(t2)+r1*np.sin(t1)]
    # 2 lines, 2 curves. All suppose sitting at z = 0 (thus, starting from (0,0))
    X = np.array([0,r,r])
    Y = np.array([0,0,h])
    circle = draw_a_circle(c1, 0, r1, t1)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))
    circle = draw_a_circle(c2, np.pi/2-t2, r2, t2)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))

    return rotate_extrude(X, Y, nsteps=64)

def ij_out():
    # mesh object: center (0,0,0) being the bottom
    h = 0
    r = 0.105
    r1 = 0.03
    t1 = 65.4/180*np.pi
    c1 = [r-r1,h]
    r2 = 0.21
    t2 = 49.2/180*np.pi/2
    c2 = [0,h-r2*np.cos(t2)+r1*np.sin(t1)]
    # 2 lines, 2 curves. All suppose sitting at z = 0 (thus, starting from (0,0))
    X = np.array([0,r,r])
    Y = np.array([0,0,h])
    circle = draw_a_circle(c1, 0, r1, t1)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))
    circle = draw_a_circle(c2, np.pi/2-t2, r2, t2)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))

    return rotate_extrude(X, Y, nsteps=64)

def ij_in():
    # mesh object: center (0,0,0) being the bottom
    h = 0
    r = 0.1
    r1 = 0.025
    t1 = 65.4/180*np.pi
    c1 = [r-r1,h]
    r2 = 0.205
    t2 = 49.2/180*np.pi/2
    c2 = [0,h-r2*np.cos(t2)+r1*np.sin(t1)]
    # 2 lines, 2 curves. All suppose sitting at z = 0 (thus, starting from (0,0))
    X = np.array([0,r,r])
    Y = np.array([0,0,h])
    circle = draw_a_circle(c1, 0, r1, t1)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))
    circle = draw_a_circle(c2, np.pi/2-t2, r2, t2)
    X = np.concatenate((X,circle[0]))
    Y = np.concatenate((Y,circle[1]))

    return rotate_extrude(X, Y, nsteps=64)

if __name__ == '__main__':
    pymesh.meshio.save_mesh('sapphire.stl', sapphire_viewpoint())
    pymesh.meshio.save_mesh('head_cone.stl',head_cone())
    pymesh.meshio.save_mesh('head_ref.stl',head_reflector())
    pymesh.meshio.save_mesh('dome_ref.stl',dome_reflector())
    pymesh.meshio.save_mesh('oj_ref.stl',oj_ref())
    pymesh.meshio.save_mesh('ij_ref.stl',ij_ref())
    pymesh.meshio.save_mesh('oj_out.stl',oj_out())
    pymesh.meshio.save_mesh('ij_out.stl',ij_out())
    pymesh.meshio.save_mesh('oj_in.stl',oj_in())
    pymesh.meshio.save_mesh('ij_in.stl',ij_in())
    #pymesh.meshio.save_mesh('sipm.stl',sipm())