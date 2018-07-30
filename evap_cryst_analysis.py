from numpy import savetxt
import pickle
from analysis_tools.read_traj import ReadTraj
from analysis_tools.feature_creation import FrameToFeaturesPosition, TrajectoryToFeaturesPosition
from analysis_tools.feature_creation import FrameToFeaturesComposition, TrajectoryToFeaturesComposition
from analysis_tools.radial_distribution_function import RDF, PositionalSuceptibility
from analysis_tools.reservoir_sampler import ReservoirSampler
from analysis_tools.defect_analysis import DefectStats
from analysis_tools.pop2d import POP2D
from numpy import array, arange
from numpy import array_split
from sklearn.decomposition import IncrementalPCA

#conditions
N_nn = 100
nn_inc = 1
N_batch = 50
batches_per_frame = 100
splits = 5
traj_type = 'gsd'
shuffle_data = True

#loop over the simulation indices
for sim index in range(1,16):
    
    #establish the trajectory to be analyzed
    filebase = 'p100_N4_{}'.format(sim_index)
    filename = '../../evap_cryst/{}/colloids.gsd'.format(filebase)
    print 'Working on {}'.format(filebase)
    
    #require two PCA phases (first to whiten and then second to analyze)
    incpca_ig = IncrementalPCA(n_components=None, whiten=True)  #whitening is ON
    incpca = IncrementalPCA(n_components=20, whiten=False)  #whitening is OFF
    OPs = []
    
    #loop over the various splits of the data
    for stage in ['whiten_pca', 'full_pca', 'calculate_ops']:

        #if whitening, the data needs to be ideal gas random
        if stage == 'whiten_pca':
            randomize = True
        elif stage in ['full_pca', 'calculate_ops']:
            randomize = False

        #read in and randomize if required
        frames = ReadTraj(filename, traj_type=traj_type, shuffle_data=shuffle_data, randomize=randomize, remove_types=[])

        #iteratively fit the pca model
        inc = len(frames)/splits + 1
        for i in range(splits):
            print 'Stage: {}'.format(stage)
            print 'Working on split {} of {}'.format(i+1, splits)
            features = TrajectoryToFeaturesPosition(frames[i*inc:i*inc+inc], 
                                                    N_nn=N_nn,  
                                                    nn_inc=nn_inc,
                                                    N_batch=N_batch, 
                                                    batches_per_frame=batches_per_frame)

            if stage == 'whiten_pca':
                incpca_ig.partial_fit(features)
            elif stage == 'full_pca':
                incpca.partial_fit(incpca_ig.transform(features))
            elif stage == 'calculate_ops':
                OPs.extend(incpca.transform(incpca_ig.transform(features)))

            clear_output()
    
    OPs = array(OPs)
    
    #save the OP analysis
    savetxt('./evap_cryst_analysis/{}.txt'.format(filebase), OPs)
 
    #save both models
    pickle_model = open('./evap_cryst_analysis/incpca_ig_{}.pkl'.format(filebase), 'wb')
    pickle.dump(incpca_ig, pickle_model)
    pickle_model.close()

    pickle_model = open('./evap_cryst_analysis/incpca_{}.pkl'.format(filebase), 'wb')
    pickle.dump(incpca, pickle_model)
    pickle_model.close()