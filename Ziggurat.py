# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools
import math
    


class Cornerpoint:

    def __init__(self, id, X, Y, level):
        self.id = id
        self.X = X
        self.Y = Y
        #self.mult = mult possibly to be added as an argument 
        self.level = level
        
    @property
    def persistence(self):
        return self.Y-self.X
        
    def merging_level(self, other):
        if ((other.X<self.X) and 
            (other.Y-other.X-self.Y+self.X>0) and 
            (other.X>2*self.X-self.Y) and 
            (other.Y <= self.Y) and 
            (self.X-other.X < self.level)):
# upside triangle
         		self.level = self.X-other.X
        elif ((other.Y>self.Y) and 
             (other.Y-other.X-self.Y+self.X >=0) and 
             (other.Y <2*self.Y-self.X) and 
             (other.X >= self.X) and 
             (other.Y-self.Y < self.level)):
# downside triangle
                self.level = other.Y-self.Y
        elif ((other.Y>self.Y)  and 
             (other.X<self.X) and 
             (other.Y<other.X +2*(self.Y-self.X)) and 
             (self.X-other.X + other.Y-self.Y < self.level)):
                self.level = self.X-other.X + other.Y-self.Y
        else: #(self.Y-self.X < self.level)
                self.level=self.Y-self.X #death due to merging with the plateau
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

