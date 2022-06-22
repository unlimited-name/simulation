import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import datetime

def rotation_matrix(thetax,thetay,thetaz):
    #the 3d rotation matrix in spherical coord., for simplicity in adding solids
    #the algorithm chroma uses: inner product(vertice, rot) + pos
    rx = np.array([[1,0,0], [0,np.cos(thetax),-1*np.sin(thetax)], [0,np.sin(thetax),np.cos(thetax)]])
    ry = np.array([[np.cos(thetay),0,np.sin(thetay)], [0,1,0], [-1*np.sin(thetay),0,np.cos(thetay)]])
    rz = np.array([[np.cos(thetaz),-1*np.sin(thetaz),0], [np.sin(thetaz),np.cos(thetaz),0], [0,0,1]])
    m = np.dot(np.dot(rx, ry), rz)
    return m

def recoordinate(Row):
    # generates a new 3d tuple with proper coordinates, for optical analysis
    # focal point at (0,0,z0)
    pos = Row[0]
    dir = Row[1]
    z0 = 0.0184 / np.tan(84.1/180*np.pi) # back "focal" point of lens
    origin = np.array([0.11545,0,0.21560]) + np.array([0,0,0.22225]) # the center of lens
    focal = np.array([0,0,z0])
    pos_rel = pos - origin #relative position
    iris = 0.00165 / 2.8 /2
    index_delete = []
    for i in range(len(pos_rel)):
        if np.dot(pos_rel,pos_rel) > (iris*iris):
            index_delete.append(i)
    rotation = rotation_matrix(22.5/180*np.pi,0,0) # rotate along x axis so the optical axis is z
    pos_rel = np.dot(pos_rel,rotation) - focal #relative to focal point
    dir_rel = np.dot(dir,rotation)
    pos_rel = np.delete(pos_rel, index_delete, 0) # filter out by the iris
    dir_rel = np.delete(dir_rel, index_delete, 0)
    return [pos_rel, dir_rel]

def repropagate_plane(Row):
    # given the position and direction of ONE photon, find the point on focal plane
    pos = Row[0]
    dir = Row[1]
    f = 0.0016
    l = -1*abs(f/dir[2])
    end = dir * l + pos
    return end

def repropagate_sphere(Row):
    # given inital position, find the spherical coordinates theta, phi
    pos = Row[0]
    dir = Row[1]
    pos_new = pos
    R = 0.300
    while np.dot(pos_new,pos_new) < R:
        pos_new = pos_new - dir * 0.001
    x,y,z = pos_new[:]
    theta = np.arctan(np.sqrt(x*x+y*y)/abs(z))
    phi = np.arctan(abs(y)/abs(x))
    return theta, phi

def projection(points):
    # given a tuple of points on focal plane, get the ones on image plane
    # only x, y coordinates are preserved
    m_proj = np.array([[-2,0,0],[0,-2,0],[0,0,1]])
    p = np.dot(points, m_proj)
    x = np.transpose(p)[0]
    y = np.transpose(p)[1]
    return np.transpose([x,y])

def equidistant_projection(theta, phi):
    # the function of Fisheye lens
    i0 = 400
    j0 = 600
    p = 0.000003
    f = 0.00165
    i = i0 + int(f*theta / p *np.cos(phi))
    j = j0 + int(f*theta / p *np.sin(phi))
    return np.array([i,j])

if __name__ == '__main__':
    # name dependent on the time of running
    namestr = str(datetime.datetime.now())
    filename = sys.argv[1]
    # load data from pos.csv and dir. csv
    buffer = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            point = equidistant_projection(repropagate_plane(recoordinate(row)))
            buffer.append(point)
    
    buffer = np.transpose(np.array(buffer))
    x = buffer[0]
    y = buffer[1]
    plt.scatter(x,y, alpha=0.1)
    plt.savefig(namestr + '_scatter.png')

    plt.hist2d(x,y)
    plt.savefig(namestr + '_hist.png')