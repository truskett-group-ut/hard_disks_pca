import gsd.hoomd
from numpy import mean, array, unique, concatenate
from numpy.random import shuffle, rand
from copy import deepcopy

def ReadGSD(filename, shuffle_data=True, randomize=False, remove_types=[]):
    frames = []
    traj = gsd.hoomd.open(name=filename, mode='rb')
    
    #static quantities
    box = traj[0].configuration.box
    D = traj[0].configuration.dimensions
    N = len(traj[0].particles.position)
    
    #loop over the configurations and stor in new format
    for snap in traj:
        #dynamic quantities
        diameters = snap.particles.diameter
        coords = snap.particles.position
        
        #check for a square box
        if (max(box[0:D]) - min(box[0:D]))/mean(box[0:D]) > 0.000000000001:
            raise Exception('Not a rectangle or square!!!')
            
        L = box[0]
        
        #get the particle types
        possible_types = snap.particles.types
        types = array([possible_types[type_id] for type_id in snap.particles.typeid])
        
        #replace with random positions if randomize is selected (for comparing to randomized PCA result and useful information content)
        if randomize:
            coords = L*rand(N, D) - L/2.0
            
        #remove a component from the trajectory
        for removal_type in remove_types:
            coords = coords[types == removal_type]
            diameters = diameters[types == removal_type]
            types = types[types == removal_type]
        
        #create our new data structure and shift to upper right quadrant
        frames.append({'coords': (coords[:,0:D]+L/2.0), 'diameters': diameters, 'types': types, 'L': L, 'D': D})
   
    #perform random shuffle of identical particles coordinates to help facilitate learning    
    if shuffle_data:
        shuffled_frames = []
        for frame in frames:
            #extract local copies for organizational convenience
            coords = frame['coords']
            diameters = frame['diameters']
            types = frame['types']
            L = frame['L']
            D = frame['D']
            
            #prepare for shuffle
            coords_shuffled = None
            unique_types, start, count = unique(types, return_index=True, return_inverse=False, return_counts=True, axis=None)
            start__end = zip(start, start+count)
            
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
            shuffled_frames.append({'coords': array(coords_shuffled), 'diameters': array(diameters), 'types': array(types), 'L': L, 'D': D})
        
        #set the data
        frames = shuffled_frames
        shuffled_frames = []    
        
    return frames