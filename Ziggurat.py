# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools  
import pandas as pd  


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
        return "Cornerpoint.\nx: {}\ty: {}\nlevel: {}\nmult: {}\n".format(self.x, self.y, self.level, self.mult)

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
        cp1.mult = cp1.mult + cp2.mult
    else:
        cp2.mult = cp1.mult + cp2.mult
    return cp1.mult, cp2.mult
        
    
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
    cp_min = min(cps)
    selected_cps = [cp for cp in cps if max({abs(cp.x - cp_min.x),
                                             abs(cp.y - cp_min.y)}) < p_min]
    #cps = [cp for cp in cps if cp not in selected_cps]
    #cps = sorted(cps)
            
    return sorted(selected_cps)
        
    
def main(data_file="C:\\Users\\Admin\\Documents\\python\\dgm_example_5.npy"):       
    pers_dgm = np.load(data_file)
    cornerpoints = [Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in pers_dgm] 
    for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        if cp1.id != cp2.id:
            cp1.merging_level(cp2)
    
    cornerpoints = sorted(cornerpoints)
    cornerpoints[-1].level = np.inf
    print([(cp.x,cp.y) for cp in cornerpoints])
        
    dct = {cp.id: merge(cp) for cp in cornerpoints} 
    for i in range(len(cornerpoints)):
        merging_sequence = merging_list(dct, cornerpoints[i]) #merging according 
        print(cornerpoints[i].id, merging_sequence)           #to the original
                                                              #elderly rule
        
    p_min = min({cp.persistence for cp in cornerpoints})
    print(p_min)
    selections = []
    
    while(len(cornerpoints) != 0): #We want to apply the new elderly rule locally.
        selected_cornerpoints = select(cornerpoints, p_min) #With the function 
        selections += [selected_cornerpoints]               #select, we divide
        cornerpoints = [cp for cp in cornerpoints           #the diagram into
                        if cp not in selected_cornerpoints] #"cornerpoint clusters" 
        cornerpoints = sorted(cornerpoints)                 #ordered by level 
                                                            #with infinity-distance 
                                                            #from the point 
                                                            #with minimum level 
                                                            #of the cluster 
                                                            #less than the 
                                                            #minimum persistence 
                                                            #of a point of the
                                                            #entire diagram.
        
        
    #We obtain the list selections whose elements are the lists of cornerpoints of each "cluster"
    #ordered by level.
    #If necessary, we can represent each cluster in a dataframe, using the following code 
    dataframes = []
        
    for c in range(len(selections)):
        ids = [selections[c][i].id for i in range(len(selections[c]))]
        xs = [selections[c][i].x for i in range(len(selections[c]))]
        ys = [selections[c][i].y for i in range(len(selections[c]))]
        levels = [selections[c][i].level for i in range(len(selections[c]))]
        mults = [selections[c][i].mult for i in range(len(selections[c]))]
        d = {"id": ids, "x": xs, "y": ys, "level": levels, "mult": mults}
        dataframe = pd.DataFrame(d, index=range(1, len(selections[c])+1))
        dataframes += [dataframe]
        
    
    for c in range(len(selections)): #Local application
        for (cp1, cp2) in zip(selections[c], selections[c][1:]):
            merge2(cp1, cp2)
            if cp1.is_older2(cp2):
               cp2 = cp1
        
    cum_mults = [[(cp.id,cp.mult) for cp in selections[c]] 
                 for c in range(len(selections))]
    print(cum_mults)
        
                
                
                
    
 
        
    
        
        #min_cor = cornerpoints[0]
        #selected_cps = [cp for cp in cornerpoints if max({abs(cp.x - min_cor.x),
                                                          #abs(cp.y - min_cor.y)}) < p_min]
        #print(selected_cps)
        #frame["selection_"+str(c)] = selected_cps
        #c+=1
        #cornerpoints = [x for x in cornerpoints if x not in selected_cps]
        #cornerpoints = sorted(cornerpoints)
        #print(cornerpoints)
        
        
    
    
    
        
    
    
    #define a function which does the selection on the ramaining elements of the list 
    
    #for i in range(len(dataframe[c]))
    #   merge(dataf, cp2)
        #if merge2(cp1, cp2) is True:
            #cp2 = "the next of cp2 in select(cornerpoints, #p_min)
            #cp1 = cp2
        #else:
            #cp2 = "the next of cp2 in selected_cps"
            
    
    
        
    
    

if __name__=="__main__":
    main()

