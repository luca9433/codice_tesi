# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import math
import os
import itertools
    


class Cornerpoint:

    def __init__(self, id, X, Y, mult):
        self.id = id
        self.X = X
        self.Y = Y
        self.mult = mult
        
    @property
    def persistence(self):
        return self.Y-self.X
        
    
        
        
    
    def merging_level(self, other):
        if other.X<self.X and other.Y-other.X-self.Y+self.X>0 and other.X>2*self.X-self.Y and other.Y <= self.Y and self.X-other.X < math.inf:
# upside triangle
     		self.level = self.X-other.X
 		elif other.Y>self.Y and other.Y-other.X-self.Y+self.X >=0 and other.Y <2*self.Y-self.X and other.X >= self.X and other.Y-self.Y < math.inf:
# downside triangle
 			self.level = other.Y-self.Y
		elif other.Y>self.Y  and other.X<self.X and other.Y<other.X +2*(self.Y-self.X) and self.X-self.X + other.Y-self.Y < math.inf:
			self.level = self.X-other.X + other.Y-self.Y
		else:
            self.level=self.Y-self.X #death due to merging with the plateau
        other.level=self.level
        return self.level
    
def merge(cornerpoints): #dict where each key is the index of the cornerpoint 
                        #merging with a given one at level k 
                        #and the corresponding value is the level k itself
    levels=np.unique([c.level for c in cornerpoints])
    return {k: [c for c in cornerpoints if c.level==k] for k in levels}
        
        
        
def main(path_to_PDs_folders="C:\Users\Admin\Documents\python"):
    grouped_by_genre_PDs = [[np.load(os.path.join(PD_paths[i], f)) 
            for f in os.listdir(PD_paths[i])] 
           for i in range(len(PD_paths))]
    PD=np.load("""some numpy file containing an array""")
    cornerpoints=[Cornerpoint(p) for p in PD]
    [cs[0].merging_level(cs[1]) for cs in itertools.combinations(cornerpoints, 2)]
    [c.merge(cornerpoints) for c in cornerpoints] #First we want to compute all the merging levels
    levels=merge(cornerpoints) 
            
    #Now we want to sort the levels

