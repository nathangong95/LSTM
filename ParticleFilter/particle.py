import numpy as np


class Particle:
    def __init__(self, parent, pos):
        # parent: another particle object
        # pos: 14*1
        self.parent = parent
        self.pos = pos
