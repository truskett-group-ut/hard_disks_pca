import pyvoro
from numpy import sum, all, any, array

#create a pure defect cells data structure from a voro data structure
def DefectCells(voro):
    #data structure to store only the 5 or 7-fold defective cells
    defect_cells = {}
    cell_index = 0
    for cell in voro:
        num_faces = len(cell['faces'])
        if num_faces in [5,7]:
            defect_cells[cell_index] = {
                                        'num_faces': num_faces, 
                                        'nbr_cells': [ac['adjacent_cell'] for ac in cell['faces']], 
                                        'original': cell['original']
                                       }
        cell_index = cell_index + 1
    return defect_cells

#generate clusters of defects
def DefectClusters(defect_cells):
    #build up clusters of connected defects to identify isolated and bound disclinations etc.
    clusters = []
    while defect_cells:
        #initialize the cluster building datastructures
        cell_stack = []
        cluster = array([0, 0])

        #pop off some arbitrary cell to seed a defect cluster
        cell, data = defect_cells.popitem()
        num_faces, coord, nbr_cells = data['num_faces'], data['original'], data['nbr_cells']
        cluster = cluster + (num_faces == 5)*array([1, 0]) + (num_faces == 7)*array([0, 1])
        cell_stack = cell_stack + nbr_cells

        #iteratively pop off and add to the cell stack until emptied signifying a complete cluster
        while cell_stack:
            cell = cell_stack.pop()
            #see if this cell is also a defect otherwise jusk keep going
            if cell in defect_cells:
                data = defect_cells.pop(cell)
                num_faces, coord, nbr_cells = data['num_faces'], data['original'], data['nbr_cells']
                cluster = cluster + (num_faces == 5)*array([1, 0]) + (num_faces == 7)*array([0, 1])
                cell_stack = cell_stack + nbr_cells

        #add the completed cluster to the clusters
        clusters.append(cluster)
    clusters = array(clusters)
    return clusters

#generate average stats for the defects
def DefectStats(frames):
    #stats to return
    num_def, num_def_pts, num_5f_disc, num_7f_disc, num_disl, num_disl_pairs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    #loop over every frame and average stats regarding defects
    for frame in frames:
        voro = pyvoro.compute_2d_voronoi(
                                          frame['coords'], # point positions
                                          [[0.0, frame['L']], [0.0, frame['L']]], # limits
                                          4.0, # block size
                                          periodic=[True, True]
                                        )
        
        #generate the defect cells data structure
        defect_cells = DefectCells(voro)
        
        #generate the defect clusters
        defect_clusters = DefectClusters(defect_cells)
        
        #get numbers regarding the various types of defects
        if any(defect_clusters):
            num_def = num_def + float(len(defect_clusters))
            num_def_pts = num_def_pts + float(sum(defect_clusters))
            num_5f_disc = num_5f_disc + float(sum(all(defect_clusters == array([1,0]), axis=1)))
            num_7f_disc = num_7f_disc + float(sum(all(defect_clusters == array([0,1]), axis=1)))
            num_disl = num_disl + float(sum(all(defect_clusters == array([1,1]), axis=1)))
            num_disl_pairs = num_disl_pairs + float(sum(all(defect_clusters == array([2,2]), axis=1)))
    
    num_frames = float(len(frames))
    return (num_5f_disc/num_frames, 
            num_7f_disc/num_frames, 
            num_disl/num_frames,
            num_disl_pairs/num_frames, 
            num_def/num_frames, 
            num_def_pts/num_frames)  