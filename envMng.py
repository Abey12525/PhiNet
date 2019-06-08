import os
import sys
import numpy as np
# import Layer_Net as lnet
# import Neuron_network as nn
# import Neuro_Connections as nc
# import Pinp

class Manage:
    def __init__(self):
        try:
            num = np.load('./layers.npy')
        except:
            print("testing ")
        else:
            num = 10


if __name__ == '__main__':
    mng = Manage()
