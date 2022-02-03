# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools    


class Cornerpoint:
    def __init__(self, id, x, y, level, mult=1):
        self.id = id
        self.x = x
        self.y = y
        self.level = level
        self.mult = mult
        self.merges_with = [self]
        
    @property
    def persistence(self):
        return self.y - self.x
    
    @property
    def plateau_merge(self):
        return self.persistence < self.level
    
    @property
    def merging_info(self):
        return self.merges_at, self.merges_with
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return self.level < other.level
    
    def __le__(self, other):
        return self.level <= other.level
    
    def __gt__(self, other):
        return self.level > other.level
    
    def __ge__(self, other):
        return self.level >= other.level

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y
    
    def upside_triangle(self, other):
        return ((other.x < self.x) and
                (other.persistence - self.persistence > 0) and 
                (other.x > 2*self.x - self.y) and (other.y <= self.y) and
                (self.x - other.x < self.level))
        
    def downside_triangle(self, other):
        return ((other.y > self.y) and
                (other.persistence - self.persistence >= 0) and
                (other.y < 2*self.y - self.x) and (other.x >= self.x) and
                (other.y - self.y < self.level))
        
    def case3(self, other):
        return ((other.y > self.y) and (other.x < self.x) and 
                (other.persistence < 2*self.persistence) and
                (other.persistence - self.persistence < self.level))
    
    def is_older(self, other):
        if self.persistence != other.persistence:
            return self.persistence > other.persistence
        elif self.x != other.x:
            return self.x > other.x
        else: 
            return True
    
    def merging_level(self, other):
        if self.upside_triangle(other):
            self.level = self.x - other.x
        elif self.downside_triangle(other):
            self.level = other.y - self.y
        elif self.case3(other):
            self.level = other.persistence - self.persistence
        elif self.plateau_merge:
            self.level = self.persistence
        if self.is_older(other):
            other.merges_with.append(self)
        else:
            self.merges_with.append(other)
        return  self.level
        
    
    def __repr__(self):
        return "Cornerpoint.\nx: {}\ty: {}\nlevel: {}\n".format(self.x, self.y, self.level)

def merge(cp1, cp2): 
    
    if  cp1.mult > cp2.mult:
         cp1.mult = cp1.mult + cp2.mult
         cp2.level = cp1.level
    elif cp1.mult < cp2.mult:
         cp2.mult = cp1.mult + cp2.mult
         cp1.level = cp2.level
    else:
        if cp1.persistence > cp2.persistence:
            cp1.mult = cp1.mult + cp2.mult
            cp2.level = cp1.level
        elif cp1.persistence < cp2.persistence:
            cp2.mult = cp1.mult + cp2.mult
            cp1.level = cp2.level
        else:
            if cp1.x > cp2.x:
                cp1.mult = cp1.mult + cp2.mult
                cp2.level = cp1.level
            elif cp1.x < cp2.x:
                cp2.mult = cp1.mult + cp2.mult
                cp1.level = cp2.level
                
    cp1.merges_at = cp2.level
    cp1.merges_with = cp2
    cp2.merges_at = cp1.level
    cp2.merges_with = cp1
    
    cp1.merging_info = cp1.merges_at, cp1.merges_with
    cp2.merging_info = cp2.merges_at, cp2.merges_with
    return cp1.merging_info, cp2.merging_info
       
def main(data_file="C:\\Users\\Admin\\Documents\\python\\dgm_example.npy"):       
    pers_dgm = np.load(data_file)
    cornerpoints=[Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in pers_dgm] 
    
    for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        if cp1.id != cp2.id:
            cp1.merging_level(cp2)
        
    cornerpoints = sorted(cornerpoints)
    cornerpoints[-1].level = np.inf
    for i in range(len(cornerpoints)):
        print((cornerpoints[i], cornerpoints[i].merges_with, cornerpoints[i] < cornerpoints[i].merges_with))
        if cornerpoints[i] < cornerpoints[i].merges_with:
            cornerpoints[i].merges_with.level = cornerpoints[i].level
            #elderly rule implementation
            
    
    
    

    
if __name__=="__main__":
    main()

