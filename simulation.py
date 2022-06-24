import random
import numpy as np

from chroma.sim import Simulation
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.generator import vertex
from chroma.sample import uniform_sphere
import detector_construction
import sys
import pandas as pd
import datetime

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

def point_source(wavelength, n, pos, dir):
    # n photons in total emitting from points (pos), with angles restricted
    # accepts multiple points input, they will be evenly rotated
    # dir is the direction of source - the direction of Z axis here

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
    # use pos,dir as the position and normal vector of the ring.
    n = int(n)
    pos = np.array(pos)
    dir = np.array(dir)
    ring = (0.06985+0.12065/2)/2
    # obtain the angles of normal vector
    if (dir[2])==0:
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
    return point_source(wavelength, n, pos_led, dir)

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
    if sys.argv[1]==0:
        mode = 'point'
    else:
        mode = sys.argv[1]
    ti = pd.to_datetime(datetime.datetime.now())
    g = detector_construction.detector_construction()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # sim.simulate() always returns an iterator even if we just pass
    namestr = str(datetime.date.today())
    #file = open(namestr + '_data.csv','w',newline='')
    #csvwriter = csv.writer(file)

    position_list = []
    direction_list = []
    if mode=='led1':
        for i in range(10):
            for j in range(1000):
    #csvwriter.writerow(['box_inside_l (mm)', 'pd_detect_l (mm)', 'detected counts'])
                for ev in sim.simulate([LED_ring(850, 1000, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=False,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                    detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                    detected_index = np.arange(1000)[detected]
                    position_list.append(pd.DataFrame(ev.photons_end.pos[detected], index = detected_index))
                    position_list.append(pd.DataFrame(ev.photons_end.dir[detected], index = detected_index))

    elif mode=='led3':
        for i in range(10):
            for j in range(1000):
                for ev in sim.simulate([LED_ring(850, 1000, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=False,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                    detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                    detected_index = np.arange(1000)[detected]
                    position_list.append(pd.DataFrame(ev.photons_end.pos[detected], index = detected_index))
                    position_list.append(pd.DataFrame(ev.photons_end.dir[detected], index = detected_index))

    elif mode=='point':
        for i in range(10):
            for j in range(1000):
                for ev in sim.simulate([photon_bomb(1000, 850, (0,0,0.1), (0,0,1))],
                           keep_photons_beg=False,keep_photons_end=True,
                           run_daq=False,max_steps=100):

                    detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
                    detected_index = np.arange(1000)[detected]
                    position_list.append(pd.DataFrame(ev.photons_end.pos[detected], index = detected_index))
                    position_list.append(pd.DataFrame(ev.photons_end.dir[detected], index = detected_index))
    else:
        print('Please enter a mode: point, led1 or led3')
    
    position_full = pd.concat(position_list)
    position_full.to_csv(namestr + '_position.csv')
    direction_full = pd.concat(direction_list)
    direction_full.to_csv(namestr + '_direction.csv')
    tf = datetime.datetime.now()
    dt = tf - ti


    print("The total time cost: ")
    print(dt)
