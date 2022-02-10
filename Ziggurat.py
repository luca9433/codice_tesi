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
        return  self.level, other.merges_with, self.merges_with
    
    def is_older2(self, other):
        if self.mult != other.mult:
            return self.mult > other.mult
        else:
            return self.is_older(other)        
        
        
    def __repr__(self):
        return "Cornerpoint.\nx: {}\ty: {}\nlevel: {}\n".format(self.x, self.y, self.level)

def merge(cp):
    support = -1
    ind_min = 0
    
    for l in range(len(cp.merges_with)):
        if support == -1:
            if cp.merges_with[l] > cp:
                support = cp.merges_with[l].level
                ind_min = l
        else:
            if cp.merges_with[l] > cp and cp.merges_with[l].level < support:
                cp.merges_with[l].level
                ind_min = l
                    
    return cp.merges_with[ind_min]

def merging_list(d, cp):
    """"
    Parameters
    ----------
    d: dict
    cp: Cornerpoint
    
    Returns
    -------
    list
    """
    merge_list = [cp.id]
    while cp.level != np.inf:
        cp = d[cp.id]
        merge_list += [cp.id]
    
    return merge_list

def merge2(cp1, cp2):
    if cp1.is_older2(cp2):
        cp2.merges_with.append(cp1)
        cp1.mult = cp1.mult + cp2.mult
        return False
    else:
        cp1.merges_with.append(cp2)
        cp2.mult = cp1.mult + cp2.mult
        return True
    
def select(cps, p_min):
    """"
    Parameters
    ----------
    cps: list
    p_min: float
    
    Returns
    -------
    list
    
    """
    cp_min = min({cp for cp in cps})
    selected_cps = []
    for i in range(len(cps)):
        if max({abs(cps[i].y - cp_min.y), abs(cps[i].x - cp_min.x)}) < p_min:
            selected_cps += [cps[i]]
            
    return selected_cps
        
    
def main(data_file="C:\\Users\\Admin\\Documents\\python\\dgm_example_4.npy"):       
    pers_dgm = np.load(data_file)
    cornerpoints = [Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in pers_dgm] 
    
    for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        if cp1.id != cp2.id:
            cp1.merging_level(cp2)
            
    cornerpoints = sorted(cornerpoints)
    cornerpoints[-1].level = np.inf
    print(cornerpoints)
    print([cp.level for cp in cornerpoints])
        
    dct = {cp.id: merge(cp) for cp in cornerpoints}
    for i in range(len(cornerpoints)):
        merging_sequence = merging_list(dct, cornerpoints[i])
        print(cornerpoints[i].id, merging_sequence)
        
    #p_min = min({cp.persistence for cp in cornerpoints})
    #print(select(cornerpoints, p_min))
    
    #for pairs (cp1, cp2) of consecutive cornerpointsfrom select(cornerpoints, 
                                                                    #p_min):
    #   merge(cp1, cp2)
        #if merge2(cp1, cp2) is True:
            #cp2 = "the next of cp2 in select(cornerpoints, #p_min)
            #cp1 = cp2
        #else:
            #cp2 = "the next of cp2 in selected_cps"
            
    
    
        
    
    

if __name__=="__main__":
    main()

