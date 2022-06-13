# Photon simulation via Python pack: chroma

## Explanation
To run the simualtion, please first run pymesh.py to build all the meshes used. They are stored in .stl format. 
The simulation script is written in simulation.py, the data are stored in .csv files. Run data_process as last step. 
## Chroma geometry scheme
Here is a brief scheme when I wrote the chroma geometry. The chroma geomtry are written in class Solid. 
They should be given a mesh, an inner material, and an outer material. In addition, surface material can be defined for absorption/reflection. 
+ Pressure Vessel (Vaccum-CF4)
    * Camera viewpoint (cylinder, sapphire)
    * Camera flange (cone, AISI 304)
    * LED rings (ring, copper)
        - more details to be fulfilled
    * Dome reflectors (sector, PTFE)
        - 3 plates next to the "head"
        - forms a cone-like shape, leaves some space at boundary
    * Cone reflectors (PTFE)
        - 8 plates with SiPM hole
+ Outer Jar (CF4-LAr)
    * Outer jar reflector (cylinder, PTFE)
        - with SiPM holes
    * Outer jar (JAR_SHAPE, quartz)
        - the above parts are quite close
+ Inner Jar (LAr-CF4)
    * Inner jar (JAR_SHAPE, quartz)
    * Inner jar reflector (circle plate + cone, PTFE)
    * Inner tower wiper (PE high density)

## To be continued...