# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:10:41 2022

@author: Admin
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import os
import pandas as pd




class Cornerpoint:
    
    def __init__(self, id, x, y, level, mult=1):
        self.id = id
        self.x = x
        self.y = y
        self.level = level
        self.mult = mult
        self.merges_with = [self] #for the original elderly rule
        self.merges_with2 = [self] #for the new elderly rule
        self.above_gap = False
        
    @property
    def persistence(self):
        return self.y - self.x
    
    @property
    def plateau_merge(self):
        return self.persistence < self.level
    
    @property
    def merging_info(self):
        return self.merges_at, self.merges_with
    
    #The following functions define an order for cornerpoints by level.
    
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
    
    def is_older(self, other): #classic elderly rule 
        if self.persistence != other.persistence:
            return self.persistence > other.persistence
        elif self.x != other.x:
            return self.x > other.x
        else: 
            return True
    
    def merging_level(self, other): #computes the level at which "self" merges with "other" 
        if self.upside_triangle(other): #(with the plateau if "other" is not in "self"'s the trapezoidal region) )
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

    
    def is_older2(self, other): #new elderly rule
        if self.mult != other.mult:
            return self.mult > other.mult
        else:
            return self.is_older(other)        
        
        
    def __repr__(self):
        return "Cornerpoint.\nid: {}\nx: {}\ty: {}\nlevel: {}\n".format(
                                                                          self.id,
                                                                          self.x, 
                                                                          self.y, 
                                                                          self.level 
                                                                          )

def merge(cp): #takes the minimum cornerpoint among those greater than cp in cp.merges_with
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

def merging_list(d, cp): #forms a list of consecutive mergings starting with cornerpoint cp
                            #and carrying on with the minimum identified by fucntion merge and so on...
    """Parameters            
    ----------
    d: dict
    cp: Cornerpoint
    
    Returns
    -------
    list
    """
    merge_list = [cp.id]
    while(len(cp.merges_with) > 1):
        cp = d[cp.id]
        merge_list += [cp.id]
    
    return merge_list

def merge2(cp1, cp2): #update attributes merge_with2 and mult according to the new elderly rule
    if cp1.is_older2(cp2):
        cp2.merges_with2.append(cp1)
        cp1.mult = cp1.mult + cp2.mult
    else:
        cp1.merges_with2.append(cp2)
        cp2.mult = cp1.mult + cp2.mult
        
    
def select(cps, p_min): #to use for selecting cornerpoints 
                        #from a list to form groups
                        #based on the minimum distance from  
                        #the minimum cornerpoint of each group 
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
            
    return sorted(selected_cps)

def merge_list(cps): #update mergings and multiplicities according to the new elderly rule

    cps_twin = cps.copy()   
    presumed_older = cps_twin[0]
    for i in range(1, len(cps)):
            merge2(presumed_older, cps_twin[i])
            #print(presumed_older.id, cps_twin[i].id)
            if presumed_older.is_older2(cps_twin[i]):
                cps_twin[i] = presumed_older
            presumed_older = cps_twin[i]
            
def widest_gap_cp(cps):
    """
    cps: list (of Cornerpoint objects)
    """
    cps = sorted(cps)[::-1]
    gaps_dct = {i: abs(cps[i].level - cps[i+1].level) 
                for i in range(1,len(cps)-1)}
    sorted_gaps_dct = {k: v for k, v in sorted(gaps_dct.items(),  
                                              key=lambda item: item[1])}
    return cps[list(sorted_gaps_dct)[-1]]

def Kurlin_select(cp, cps):
    cps = sorted(cps)[::-1]
    if cp >= widest_gap_cp(cps):
        cp.above_gap = True
    else:
        cp.above_gap = False
        


def plot_animated_rank(cps):             
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    colors = []
    plt.xlim(0, 15)
    plt.ylim(0, 20)
    plt.draw()
    
    for cp in cps[::-1]:
        x.append(cp.x)
        y.append(cp.y)
        colors.append(cm.Set1(cp.level))
        sc.set_offsets(np.c_[np.asarray(x),np.asarray(y)])
        sc.set_color(colors)
        fig.canvas.draw_idle()
        plt.pause(0.3)
        plt.show()
    
        
        
def test_plot_animated_rank():
    cps = [Cornerpoint(i, x, y, l) for i, (x,y,l) in enumerate(np.random.rand(10, 3))]
    plot_animated_rank(cps)
    
def build_GIF(cps): 
    with imageio.get_writer('mygifM.gif', mode='I') as writer:
        for filename in [str(d)+'.png' for d in range(len(cps))]:
            image = imageio.imread(filename)
            writer.append_data(image)
            
def plot_gap(cps, save_path=None):
    cps_a = [c for c in cps if c.above_gap]
    cps_b = [c for c in cps if not c.above_gap]
    f, ax = plt.subplots()
    ax.scatter([c.x for c in cps_a], [c.y for c in cps_a], label="above gap")
    ax.scatter([c.x for c in cps_b], [c.y for c in cps_b], label="below gap")
    plt.xlabel("x")
    plt.ylabel("y")
    f.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "plot_gap.png"))
        
def plot_gap_ZigguratPD(cps, save_path=None):
    cps_a = [c for c in cps if c.above_gap]
    cps_b = [c for c in cps if not c.above_gap]
    f, ax = plt.subplots()
    ax.scatter([0 for c in cps_a], [c.level for c in cps_a], label="above gap")
    ax.scatter([0 for c in cps_b], [c.level for c in cps_b], label="below gap")
    plt.ylabel("level")
    f.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "plot_gap_1D.png"))
        
def plot_diagram(cps, save_path=None):
    f, ax = plt.subplots()
    X = [cp.x for cp in cps]
    Y = [cp.y for cp in cps]
    plt.scatter(X, Y)
    plt.xlabel("x")
    plt.ylabel("y")
    if save_path is not None:
        f.savefig(os.path.join(save_path, "dgm_Masimo_repr.png"))
        
def create_dataframe(cps):
    ids = [cp.id for cp in cps]
    xs = [cp.x for cp in cps]
    ys = [cp.y for cp in cps]
    levels = [cp.level for cp in cps]
    d = {"id": ids, "x": xs, "y": ys, "level": levels}
    dataframe = pd.DataFrame(d, index=range(1, len(cps)+1))
    print(dataframe.to_latex(index=False))

    
    
def main(data_file="C:\\Users\\Admin\\Documents\\python\\dgm_Massimo.npy"):       
    pers_dgm = np.load(data_file)
    cornerpoints = [Cornerpoint(int(p[0]), p[1], p[2], np.inf) for p in pers_dgm] 
    for (cp1, cp2) in itertools.product(cornerpoints, repeat=2):
        if cp1.id != cp2.id:
            cp1.merging_level(cp2)

    #print([(cp.id, [cp.merges_with[i].id for i in range(len(cp.merges_with))])
             #for cp in cornerpoints])
    #print([(cp.id,cp.x,cp.y,cp.level) for cp in cornerpoints])
    #print(cornerpoints)

        
        
    #abs_vals = [abs(cornerpoints[i].level-cornerpoints[i+1].level)
                  #for i in range(len(cornerpoints)-1)]
    #print(abs_vals)
    #diffs = {cornerpoints[i].id: abs_vals[i] for i in range(len(cornerpoints)-1)}
    #print(diffs)
    #sort_diffs = {k: v for k, v in sorted(diffs.items(), key=lambda item: item[1])}
    #print(sort_diffs)
    
    plot_diagram(cornerpoints, save_path="C:\\Users\\Admin\\Documents\\python\\_codice_tesi_")
    
    create_dataframe(cornerpoints)
    cornerpoints = sorted(cornerpoints)
    create_dataframe(cornerpoints[::-1])
    #print([cp for cp in cornerpoints[::-1]])
    print(widest_gap_cp(cornerpoints))
    
    plot_animated_rank(cornerpoints)
    
    #Build GIF
    with imageio.get_writer('mygif_gap_1D.gif', mode='I') as writer:
        for filename in ["gap_1D"+str(d)+'.png' for d in range(len(cornerpoints))]:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    #test_plot_animated_rank    
        
    #X = np.asarray([cp.x for cp in cornerpoints[33:]])
    #Y = np.asarray([cp.y for cp in cornerpoints[33:]])
    #plt.scatter(X, Y)
    #plt.show()
        
    dct = {cp.id: merge(cp) for cp in cornerpoints} 
    for i in range(len(cornerpoints)):
        merging_sequence = merging_list(dct, cornerpoints[i])  
        print(cornerpoints[i].id, merging_sequence)        
     #merging according to the original elderly rule                                                            
        
    p_min = min([cp.persistence for cp in cornerpoints])
    selections = []
    
    while(len(cornerpoints) != 0): #We want to apply the new elderly rule locally.
        selected_cornerpoints = select(cornerpoints, p_min) #With the function 
        selections += [selected_cornerpoints]               #select, we divide
        cornerpoints = [cp for cp in cornerpoints           #the diagram into groups
                        if cp not in selected_cornerpoints] #"nearby cornerpoints" 
        cornerpoints = sorted(cornerpoints)                 #ordered by level 
                                                            #with infinity-distance 
                                                            #from the point 
                                                            #with the minimum level 
                                                            #of each group 
                                                            #less than the 
                                                            #minimum persistence 
                                                            #of a point in the
                                                            #entire diagram.
        
    #We obtain the list selections whose elements are the lists of cornerpoints 
    #of each "cluster" ordered by level.
        
    #print([cp.id for cp in selections[0]])
    #rint(selections[0])
    
    for c in range(len(selections)): #Local application of the new elderly rule
        merge_list(selections[c])
            
    #print([[selection[i].id for i in range(len(selection))] for selection in selections])    
    #cum_mults = [[(cp.id, cp.mult) 
                 #for cp in selections[c]] 
                 #for c in range(len(selections))]
    #print(cum_mults)
    
    #dct2 = [{cp.id: [cp.merges_with2[i].id for i in range(len(cp.merges_with2))] 
                     #for cp in selections[c]} 
                        #for c in range(len(selections))]
    #print(dct2)
                               

if __name__=="__main__":
    main()

