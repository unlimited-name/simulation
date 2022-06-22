import random
import numpy as np

from chroma.sim import Simulation
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.generator import vertex
import detector_construction
import sys
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
    # n photons in total emitting from points (pos), with angles restricted
    # accepts multiple points input, they will be evenly rotated

    n = int(n)
    # restrict angles
    phi_restrict = 2*np.pi
    theta_restrict = 60/180*np.pi

    pos_photon = []
    dir_photon = []
    pol_photon = []

    while len(dir_photon)<n:

        for i in range(len(pos)):
            pos_photon.append(pos[i])
            phi = phi_restrict * random.random()
            theta = theta_restrict * random.random()
            dirvec = np.array((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta)))
            dir_photon.append(dirvec + dir)
            pol_photon.append(np.cross(dir_photon,random_vector()))

    pos_photon = np.array(pos_photon)
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
    pos_led = []
    delta = np.pi/12
    for i in range(24):
        x = ring * np.cos(i*delta + phi)
        y = ring * np.sin(i*delta + phi)
        z = 0
        pos_led.append(np.dot(np.array([x,y,z]), rotation_matrix(theta, 0, phi)) + pos)

    pos_led = np.array(pos_led)
    return point_source(wavelength, n, pos_led, dir)

def triple_LED_ring(wavelength, n, pos, dir):
    # the function used for 3 LEDs, the input parameters are the same with LED1
    # but with LED1 rotated 3/pi each time
    n = int(n/24/3)
    ring = 0.24
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

if __name__ == '__main__':
    mode = sys.argv[1]
    ti = datetime.datetime.now()
    g = detector_construction.detector_construction()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # sim.simulate() always returns an iterator even if we just pass
    pos_detected = []
    dir_detected = []
    namestr = str(datetime.date.today())
    file = open(namestr + '_data.csv','w',newline='')
    csvwriter = csv.writer(file)
    
    if mode=='led1':
        for i in range(1):
            for j in range(1000):
    #csvwriter.writerow(['box_inside_l (mm)', 'pd_detect_l (mm)', 'detected counts'])
                for ev in sim.simulate([LED_ring(850, 1000, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=False,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                    detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                    pos = ev.photons_end.pos[detected]
                    dir = ev.photons_end.dir[detected]
                    buffer = np.transpose(np.array([pos,dir]))
                    for item in buffer:
                        csvwriter.writerow(item[0])
                        csvwriter.writerow(item[1])

        # one line of position, along with on line of diretion
    elif mode=='led3':
        for i in range(10):
            for j in range(1000):
                for ev in sim.simulate([LED_ring(850, 1000, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                    detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                    csvwriter.writerow('')
                    csvwriter.writerow(ev.photons_end.pos[detected])
                    csvwriter.writerow('')
                    csvwriter.writerow(ev.photons_end.dir[detected])
    else:
        print('Please enter a mode: led1 or led3')
    
    tf = datetime.datetime.now()
    dt = tf - ti
    csvwriter.writerow('The total time cost: ')
    csvwriter.writerow(dt)
    file.close()

    print("The total time cost: ")
    print(dt)
