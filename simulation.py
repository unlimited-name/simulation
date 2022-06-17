import random
import numpy as np

from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.generator import vertex
import detector_construction
import csv
import datetime

def random_vector():
    # generate a normalized 3d random vector, used for polarization
    phi = 2*np.pi * random.random()
    theta = np.pi * random.random()
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

def point_source(wavelength, n, pos, dir):
    # n photons emitting from point (pos), with angles restricted
    
    # restrict angles
    n = int(n)
    pos_photon = np.array(pos)
    phi = 2*np.pi * random.random()
    theta = 60/180*np.pi

    dir_photon = []
    pol_photon = []
    for i in range(n):
        dirvec = np.array((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta)))
        dir_photon.append(dirvec + dir)
        pol_photon.append(np.cross(dir_photon,random_vector()))

    dir_photon = np.array(dir_photon)
    pol_photon = np.array(pol_photon)
    wavelengths = np.repeat(wavelength,n)
    return Photons(pos_photon, dir_photon, pol_photon, wavelengths)

def LED_ring(wavelength, n, pos, dir):
    # n photons emitting photons evenly from 24 LEDs arranged in a ring
    # use pos,dir as the position and normal vector of the ring.
    n = n//24
    ring = 0.24
    # obtain the angles of normal vector
    phi = np.arctan(dir[1]/dir[0])
    theta = np.arctan(np.sqrt(dir[0]*dir[0]+dir[1]*dir[1])/dir[2])

    # position matrices for 24 LEDs
    pos_light = []
    delta = np.pi/12
    for i in range(24):
        x = ring * np.cos(i*delta + phi)
        y = ring * np.sin(i*delta + phi)
        z = 0
        pos_light.append(np.dot(np.array([x,y,z]), rotation_matrix(theta, 0, phi)) + pos)

    return point_source(wavelength, n, pos_light, dir)


if __name__ == '__main__':
    ti = datetime.datetime.now()
    g = detector_construction.detector_construction()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # sim.simulate() always returns an iterator even if we just pass
    # a single photon bomb
    pos_detected = []
    dir_detected = []
    namestr = str(datetime.date.today())
    file = open(namestr + '_data.csv','w',newline='')
    csvwriter = csv.writer(file)
    
    for i in range(10):
        for j in range(1000):
    #csvwriter.writerow(['box_inside_l (mm)', 'pd_detect_l (mm)', 'detected counts'])
            for ev in sim.simulate([LED_ring(850, 1000, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
        #pos_detected.append(ev.photons_end.pos[detected])
                csvwriter.writerow(ev.photons_end.pos[detected])
        #dir_detected.append(ev.photons_end.dir[detected])
                csvwriter.writerow(ev.photons_end.dir[detected])
        # one line of position, along with on line of diretion

    file.close()
    tf = datetime.datetime.now()
    dt = tf - ti
    print("The total time cost: ")
    print(dt)
