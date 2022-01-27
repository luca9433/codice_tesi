# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools
import math
    


class Cornerpoint:

    def __init__(self, id, x, y, level):
        self.id = id
        self.x = x
        self.y = y
        #self.mult = mult possibly to be added as an argument 
        self.level = level
        
    @property
    def persistence(self):
        return self.Y-self.X
        
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
        elif (self.y-self.x < self.level):
                self.level=self.y-self.x #death due to merging with the plateau
        #other.level=self.level
        return  self.level
    
#def merge(cornerpoints): #dict where each key is the index of the cornerpoint 
                        #merging with a given one at level k 
                        #and the corresponding value is the level k itself
    #levels=np.unique([c.level for c in cornerpoints])
    #return {k: [c.id for c in cornerpoints if c.level==k] for k in levels}
        
def main(data_file="C:\\Users\\Admin\\Documents\\python\\gurrieri_dataset_npy.npy"):       
    PD=np.load(data_file)
    print(PD)
    k_max=0
    k=math.inf
    cornerpoints=[Cornerpoint(int(p[0]), p[1], p[2], k) for p in PD]
    #for (i,h) in itertools.combinations(cornerpoints, 2):
        #i.level=i.merging_level(h)
    for i in range(len(cornerpoints)):
        for h in list(range(i)) + list(range(i+1,len(cornerpoints))):
            cornerpoints[i].level=cornerpoints[i].merging_level(cornerpoints[h])
        if cornerpoints[i].level > k_max:
            k_max = cornerpoints[i].level
            j=i
    cornerpoints[j].level = math.inf
    for i in range(len(cornerpoints)):
        print(cornerpoints[i].level)



#=[cs[0].merging_level(cs[1]) for cs in itertools.combinations(cornerpoints, 2)]
#print(k)
#print([c.merge(cornerpoints) for c in cornerpoints]) #First we want to compute all the merging levels
#levels=merge(cornerpoints)
#print(levels, "\n")
#print([(cs[0].id,cs[1].id, cs[0].merging_level(cs[1])) for cs in itertools.combinations(cornerpoints, 2)])
            
    #Now we want to sort the levels
    
if __name__=="__main__":
    main()

