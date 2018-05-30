# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:35:58 2018

@author: Paolo Gabriel
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.lines as mlines

#  === Angle functions ===

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, deg=True):
    """ Returns the anti-clockwise angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """

    def inner_angle(v, w):
        cross = np.cross(v, w)
        cosang = np.dot(v, w)
        sinang = np.linalg.norm(cross)
        return np.arctan2(sinang, cosang)
    
    def determinant(v,w):
        return v[0]*w[1]-v[1]*w[0]
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    angle = inner_angle(v1_u, v2_u)
    det = determinant(v1_u, v2_u)
    
    if det < 0: angle = 2*np.pi - angle

    if deg: angle = angle * 180/np.pi
    
    return angle

def get_axillary_angle(base, rotator):
    """ Returns the shoulder angle in degrees between vectors 'base' (static) and 'rotator' after checking the direction of 'base'::

    """
    if base[0] > 0:
        return angle_between(rotator, base)
    else:
        return angle_between(base, rotator)
    
def get_elbow_angle(base, rotator):
    """ Returns the elbow angle in degrees between vectors 'base' (static) and 'rotator'::

    """
    return min(angle_between(rotator, base), angle_between(base, rotator))
# == / Angle functions / ==

# == Extra functions ==
def gen_rand_vecs(dims, number):
    vecs = np.random.normal(size=(number,dims))
    mags = np.linalg.norm(vecs, axis=-1)

    return vecs / mags[..., np.newaxis]

def plot_angles(ax, base, rotator):
    angle1 = angle_between((1,0), base) 
    angle2 = angle_between((1,0), rotator)
    
    l1 = mlines.Line2D([0,base[0]], [0, base[1]], color='k')
    l2 = mlines.Line2D([0,rotator[0]], [0, rotator[1]], color='b')
    
    return l1, l2, angle1, angle2

# NOTE: plot_arcs is bugged! 
#   Angles calculated are correct, but the way to plot arcs requires some logic 
#   regarding the relationship between v1 and v2
def plot_arcs(ax, v1, v2, offset1, offset2):
    theta1 = get_elbow_angle(v1, v2)
    theta2 = get_axillary_angle(v1, v2)
    
    if offset2 - offset1 > 180:
        e_offset = max(offset1, offset2)
    else:        
        e_offset = min(offset1, offset2)
    
    if offset2 - offset1 < theta2:
        a_offset = max(offset1, offset2)
    else:
        a_offset = min(offset1, offset2)

    arc1 = Arc([0,0], 0.5, 0.5, angle=0, theta1=e_offset, theta2=e_offset+theta1, color='g')
    arc2 = Arc([0,0], 0.75, 0.75, angle=0, theta1=a_offset, theta2=a_offset+theta2, color='r')

    
    return arc1, arc2, theta1, theta2

def run_demo():
    plt.close('all')
    # generate simulated data
    v1 = np.array((-1, 0)) # simulation of RS -> LS (shoulder to shoulder) vector
    v2s = ((1,0), (1,1), (0,1), (-1,1), (-1,0) , (-1,-1), (0, -1), (1,-1)) # simulation of fixed angles of RS->RE (shoulder to elbow)
    
    # plot simulated data
    for i, v2 in enumerate(v2s):
        fig, ax = plt.subplots(1)
        
        l1, l2, offset1, offset2 = plot_angles(ax, v1, v2)
        arc1, arc2, theta1, theta2 = plot_arcs(ax, v1, v2, offset1, offset2)
                
        ax.add_patch(arc1)
        ax.add_patch(arc2)
        ax.add_line(l1)
        ax.add_line(l2)
    
        ax.legend(('Base', 'Rotator', 'elbow angle = {0} deg'.format(theta1), 'shoulder angle = {0} deg'.format(theta2)))
    
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)    
        ax.set_aspect('equal')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()

#%%
a=1