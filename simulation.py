if __name__ == '__main__':
    import numpy as np
    from chroma.sim import Simulation
    from chroma.sample import uniform_sphere
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import csv

    g = build_detector()
    g.flatten()
    g.bvh = load_bvh(g)

    sim = Simulation(g)

    # photon bomb from center
    def photon_bomb(n,wavelength,pos):
        pos = np.tile(pos,(n,0.25))
        dir = uniform_sphere(n)
        pol = np.cross(dir,uniform_sphere(n))
        wavelengths = np.repeat(wavelength,n)
        return Photons(pos,dir,pol,wavelengths)

    # sim.simulate() always returns an iterator even if we just pass
    # a single photon bomb
    pos_detected = []
    dir_detected = []

    csvwriter = csv.writer(csvfile) 
    #csvwriter.writerow(['box_inside_l (mm)', 'pd_detect_l (mm)', 'detected counts'])
    for ev in sim.simulate([photon_bomb(1000,850,(0,0,0.1))],
                           keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=100):
        f.write_event(ev)
        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
        #pos_detected.append(ev.photons_end.pos[detected])
        csvwriter.writerow(ev.photons_end.pos[detected])
        #dir_detected.append(ev.photons_end.dir[detected])
        csvwriter.writerow(ev.photons_end.dir[detected])

    f.close()
