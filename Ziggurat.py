# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools    


class Cornerpoint:

    def __init__(self, id, x, y, level):
        self.id = id
        self.x = x
        self.y = y
        #self.mult = mult possibly to be added as an argument 
        self.level = level
        
    @property
    def persistence(self):
        return self.y - self.x
    
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
    
    def merging_level(self, other):
        if ((other.x<self.x) and 
            (other.y-other.x-self.y+self.x>0) and 
            (other.x>2*self.x-self.y) and 
            (other.y <= self.y) and 
            (self.x-other.x < self.level)):
# upside triangle
         		self.level = self.x-other.x
        elif ((other.y>self.y) and 
             (other.y-other.x-self.y+self.x >=0) and 
             (other.y <2*self.y-self.x) and 
             (other.x >= self.x) and 
             (other.y-self.y < self.level)):
# downside triangle
                self.level = other.y-self.y
        elif ((other.y>self.y)  and 
             (other.x<self.x) and 
             (other.y<other.x +2*(self.y-self.x)) and 
             (self.x-other.x + other.y-self.y < self.level)):
                self.level = self.x-other.x + other.y-self.y
        elif (self.persistence < self.level):
                self.level=self.persistence #death due to merging with the plateau
        #other.level=self.level
        return  self.level
    
#def merge(cornerpoints): #dict where each key is the index of the cornerpoint 
                        #merging with a given one at level k 
                        #and the corresponding value is the level k itself
    #levels=np.unique([c.level for c in cornerpoints])
    #return {k: [c.id for c in cornerpoints if c.level==k] for k in levels}
    
    def __repr__(self):
        return "Cornerpoint.\nx: {}\ty: {}\nlevel: {}\n".format(self.x, self.y, self.level)
    
def main(data_file="C:\\Users\\Admin\\Documents\\python\\gurrieri_dataset_npy.npy"):       
    PD=np.load(data_file)
    print(PD)
    cornerpoints=[Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in PD]
    
    for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        if cp1.id != cp2.id:
            cp1.level=cp1.merging_level(cp2)
        
        
    
    cornerpoints = sorted(cornerpoints)
    cornerpoints[-1]=np.inf
    print(cornerpoints)
    



#=[cs[0].merging_level(cs[1]) for cs in itertools.combinations(cornerpoints, 2)]
#print(k)
#print([c.merge(cornerpoints) for c in cornerpoints]) #First we want to compute all the merging levels
#levels=merge(cornerpoints)
#print(levels, "\n")
#print([(cs[0].id,cs[1].id, cs[0].merging_level(cs[1])) for cs in itertools.combinations(cornerpoints, 2)])
            
    #Now we want to sort the levels
    
if __name__=="__main__":
    main()

