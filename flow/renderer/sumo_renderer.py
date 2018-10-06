import numpy as np
import os
import scipy.misc
import traci

class SUMORenderer():

    def __init__(self, kernel, width=100, height=100):
        print("Using SUMORenderer...")
        self.kernel = kernel
        self.width = width
        self.height = height


    def render(self):
        print(self.kernel.gui.getIDList())
        traci.gui.screenshot("View #0", "frame.png")
        return scipy.misc.imread("frame.png", flatten=False, mode="RGB")


    def close(self):
        #os.remove("frame.png")
        pass
