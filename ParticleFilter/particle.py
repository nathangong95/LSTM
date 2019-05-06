import numpy as np


class Particle:
    def __init__(self, parent, pos, ht, ct):
        # parent: another particle object
        # pos: 14*1
        self.parent = parent
        self.pos = pos
        self.ht = ht
        self.ct = ct
