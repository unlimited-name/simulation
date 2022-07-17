import numpy as np
from chroma.geometry import Material, Surface

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
CF4.set('absorption_length',1e6)
CF4.set('scattering_length',1e6)

ssteel = Material('SSteel')
ssteel.set('refractive_index', 1.0)
ssteel.set('absorption_length',1e6)
ssteel.set('scattering_length',1e6)

# these 2 are not actually used
PTFE = Material('PTFE')
PTFE.set('refractive_index', 1.0)
PTFE.set('absorption_length',1e6)
PTFE.set('scattering_length',1e6)

sapphire = Material('sapphire')
sapphire.set('refractive_index', 1.0)
sapphire.set('absorption_length',1e6)
sapphire.set('scattering_length',1e6)

quartz = Material('quartz')
quartz.set('refractive_index', 1.49)
quartz.set('absorption_length',1e6)
quartz.set('scattering_length',1e6)

LAr = Material('LAr')
LAr.set('refractive_index', 2.1)
LAr.set('absorption_length',1e6)
LAr.set('scattering_length',1e6)

# surfaces
black_surface = Surface('black_surface')
black_surface.set('absorb', 1)
# same as chroma/demo, used at SiPM and Steel

quartz_surface = Surface('quartz_surface')
quartz_surface.set('absorb', 1)

teflon_surface = Surface('teflon_surface')
teflon_surface.set('absorb', 1)

shiny_surface = Surface('shiny_surface')
shiny_surface.set('reflect_diffuse', 1)

detector_surface = Surface('detector_surface')
detector_surface.detect = np.array([(400,1),(850, 1),(1000,1)])
# chroma will do linear interpolation for the detection rate
detector_surface.set('absorb',1)
# let the detection and absorption both be 1
"""
for more parameters and details concerning optics, refer to chroma/demo/optics
"""