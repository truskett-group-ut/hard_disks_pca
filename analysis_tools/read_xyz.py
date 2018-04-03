from numpy import mean, array, unique, concatenate, power, std, abs
from numpy.random import shuffle, rand
from copy import deepcopy
import re

def ReadXYZ(filename, shuffle_data=True, randomize=False, remove_types=[]):
    #regex used to parse the data
    xyz_regex = r'(?:([a-zA-Z]+)\s+([0-9\.e\-\+]+)\s+([0-9\.e\-\+]+)\s+([0-9\.e\-\+]+)\s+)'
    L_regex = r'(?:L\s*=\s*([0-9\.e\-\+]+))'
    Ns = set([])
    Ds = set([])
    
    #read in the xyz file line by line and use regex to extract and build a new datastructure for the coordinates
    with open(filename, "r") as ins:
        frames = []
        coords, coords_count = [], 0
        types = []
        L = None
        for line in ins:
            search = re.search(xyz_regex, line)
            if search:
                coords.append(array([float(search.group(2)), float(search.group(3)), float(search.group(4))]))
                types.append(search.group(1))
                coords_count = coords_count + 1
            elif coords:
                coords = array(coords)
                types = array(types)
                D = 3 - int(abs(max(coords[:,2]) - min(coords[:,2]))/L < 1e-10)
                Ds.add(D)
                Ns.add(len(coords))
                
                #replace with random positions if randomize is selected 
                if randomize:
                    coords = L*rand(len(coords), len(coords[0]))
                frames.append({'coords': coords[:,0:D], 'types': types, 'L': L, 'D': D})
                
                #remove a component from the trajectory
                for removal_type in remove_types:
                    coords = coords[types == removal_type]
                    types = types[types == removal_type]
                
                coords, coords_count = [], 0
                types = []
                L = None
                
                
                search_L = re.search(L_regex, line)
                if search_L:
                    L = float(search_L.group(1))
            else:
                search_L = re.search(L_regex, line)
                if search_L:
                    L = float(search_L.group(1))
                    
        #append final frame
        if coords:
            coords = array(coords)
            types = array(types)
            D = 3 - int(abs(max(coords[:,2]) - min(coords[:,2]))/L < 1e-10)
            Ds.add(D)
            Ns.add(len(coords))

            #replace with random positions if randomize is selected 
            if randomize:
                coords = L*rand(len(coords), len(coords[0]))
            frames.append({'coords': coords[:,0:D], 'types': types, 'L': L, 'D': D})

            #remove a component from the trajectory
            for removal_type in remove_types:
                coords = coords[types == removal_type]
                types = types[types == removal_type]

            coords, coords_count = [], 0
            types = []
            L = None
        
    #check that all of the frames contain the same number of particles
    if len(Ns) != 1:
        raise Exception('For some reason the reader did not identify the same number of particles for all frames.')
        
    #check that all of the frames contain the same dimensionality
    if len(Ds) != 1:
        raise Exception('For some reason the reader did not identify the same dimensionality all frames.')
         
    #perform random shuffle of identical particles coordinates to help facilitate learning    
    if shuffle_data:
        shuffled_frames = []
        for frame in frames:
            #extract local copies for organizational convenience
            coords = frame['coords']
            types = frame['types']
            L = frame['L']
            D = frame['D']
            
            #prepare for shuffle
            coords_shuffled = None
            unique_types, start, count = unique(types, return_index=True, return_inverse=False, return_counts=True, axis=None)
            start__end = zip(start, start+count)
            #print types
            
            #check for errors
            if len(start__end) != len(unique_types):
                raise Exception('Bad data!!!')
            
            #do the shuffling
            for start, end in start__end:
                grouped = deepcopy(coords[start:end])
                shuffle(grouped)
                if coords_shuffled is not None:
                    coords_shuffled = concatenate((coords_shuffled, grouped), axis=0)
                else:
                    coords_shuffled = deepcopy(grouped)
            shuffled_frames.append({'coords': array(coords_shuffled), 'types': array(types), 'L': L, 'D': D})
        
        #set the data
        frames = shuffled_frames
        shuffled_frames = []
    
    return frames