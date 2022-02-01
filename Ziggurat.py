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
        self.mult = mult 
        self.level = level
        
    @property
    def persistence(self):
        return self.y - self.x
    
    @property
    def plateau_merge(self):
        return self.persistence < self.level
    
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
                (other.persistence - self.persistence>0) and 
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
    
    def merging_level(self, other):
        if self.upside_triangle(other):
            self.level = self.x - other.x
        elif self.downside_triangle(other):
            self.level = other.y - self.y
        elif self.case3(other):
            self.level = other.persistence - self.persistence
        elif self.plateau_merge:
            self.level = self.persistence
        return  self.level
    
    def setBuddy(self, merging_buddy):
        self.merging_buddy = merging_buddy
        merging_buddy.merging_buddy = self
        
    
    def __repr__(self):
        return "Cornerpoint.\nx: {}\ty: {}\nlevel: {}\n".format(self.x, self.y, self.level)

#def merge(cornerpoints): #dict where each key is the index of the cornerpoint 
                        #merging with a given one at level k 
                        #and the corresponding value is the level k itself
    #levels=np.unique([c.level for c in cornerpoints])
    #return {k: [c.id for c in cornerpoints if c.level==k] for k in levels}
 
def main(data_file="C:\\Users\\Admin\\Documents\\python\\dgm_example.npy"):       
    pers_dgm = np.load(data_file)
    cornerpoints=[Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in pers_dgm] 
    
    #for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        #if cp1.id != cp2.id:
            #cp1.level = cp1.merging_level(cp2)
            #cp1.setBuddy(cp2)
            
    for i in range(len(cornerpoints)):
        for h in list(range(i)) + list(range(i+1,len(cornerpoints))):
            cornerpoints[i].level = cornerpoints[i].merging_level(cornerpoints[h])
        cornerpoints[i].setBuddy#...I would like to keep track of the cornerpoint    
                                #mergigng with the cornerpoint
                                #with id=i at this level.
            
    
    #buddies = [(cp, cp.merging_buddy) for cp in cornerpoints]
    cornerpoints = sorted(cornerpoints)
    cornerpoints[-1] = np.inf
    print(cornerpoints)
    #{k: [c.id for c in cornerpoints if c.level==k] for k in }
    
    #=[cs[0].merging_level(cs[1]) for cs in itertools.combinations(cornerpoints, 2)]
    #print(k)
    #print([c.merge(cornerpoints) for c in cornerpoints]) #First we want to compute all the merging levels
    #levels=merge(cornerpoints)
    #print(levels, "\n")
    

    
if __name__=="__main__":
    main()

