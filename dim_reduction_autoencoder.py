from analysis_tools.read_traj import ReadTraj
from analysis_tools.feature_creation import FrameToFeaturesPosition, TrajectoryToFeaturesPosition
from analysis_tools.feature_creation import FrameToFeaturesComposition, TrajectoryToFeaturesComposition
from analysis_tools.radial_distribution_function import RDF, PositionalSuceptibility
from analysis_tools.reservoir_sampler import ReservoirSampler
from analysis_tools.defect_analysis import DefectStats
from analysis_tools.pop2d import POP2D
import gc
from IPython.display import clear_output
import h5py

def DownSampleFrames(frames, frame_samples):
    return frames[0::max(len(frames)/frame_samples, 1)][0:frame_samples]

#####SPECIFY DATA AND CONDITIONS#####
from numpy import arange, array

N_nn = 3200 #number of nearest neighbors for pca analysis
split = 1 #chunks the data up so it can be processed by the pca tool if really large
nn_inc = 10 #1 (full in paper) #reduces the number of nearest neighbors to include as features
remove_types = []
shuffle_data = True
N_batch = 1
batches_per_frame = 1 

#specify what data to read in and process
traj_type = 'gsd'
file_data = [(arange(0.550, 0.690001, 0.005), '../hoomd_disks/trajectories_4000p', 400), 
             (arange(0.695, 0.820001, 0.005),'../hoomd_disks/trajectories_4000p_longer', 400)]

#construct a contiguous list of densities
etas = []
[etas.extend(etas_) for etas_, _, _ in file_data]
etas = array(etas)

#####READ IN THE DATA#####
from numpy import array_split
from sklearn.decomposition import IncrementalPCA

corrected_features = []
incpca_ig = IncrementalPCA(n_components=None, whiten=True) 
force_randomize = False

#loop over data sets 
for phase in ['fit_whitener', 'correct_features']:
    raw_features = []
    
    for etas_, file_base, frame_samples in file_data:

        for eta in etas_:
            print 'COMPUTATION DETAILS'
            print 'file_base = {}'.format(file_base)
            print 'eta = {}'.format(eta)

            #read in data and randomize positions if performing ideal gas correction
            filename = "{}/trajectory_{:.4f}.{}".format(file_base, eta, traj_type)
            randomize = (phase == 'fit_whitener') or force_randomize
            frames = ReadTraj(filename, traj_type, shuffle_data, randomize, remove_types)

            #control the number of total frames to analyze
            len_frames_init = len(frames)
            frames = DownSampleFrames(frames, frame_samples)
            print 'using {} frames of {} total'.format(len(frames), len_frames_init)
            
            raw_features.extend(TrajectoryToFeaturesPosition(frames, 
                                                             N_nn=N_nn,  
                                                             nn_inc=nn_inc,
                                                             N_batch=N_batch, 
                                                             batches_per_frame=batches_per_frame))
    if phase == 'fit_whitener':
        print 'Fitting the whitener\n'
        incpca_ig.fit(raw_features)
    else:
        print 'Correcting features\n'
        corrected_features = incpca_ig.transform(raw_features)
            

#####AUTOENCODING#####
from analysis_tools.autoencoder import AutoEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from numpy import savetxt

dim = corrected_features.shape[1]
corrected_features_train, corrected_features_test = train_test_split(corrected_features, test_size=0.25, random_state=42)

#try various random starts
for i in range(20):
    print "---------------------\n"
    print "Starting on model {}\n".format(i)
    print "---------------------\n"
    model = AutoEncoder(dim=dim, encode_dim=1, 
                        initial_growth=0.25, shrink=0.65,
                        activation='selu', kernel_initializer='lecun_normal')
    print "Layers: {}\n".format(model.layer_dims)
    model.Compile(optimizer='adamax', loss='mean_squared_error')
    checkpointer = ModelCheckpoint(filepath='./model/weights_{}.hdf5'.format(i), verbose=1, save_best_only=True)
    history = model.Fit(corrected_features_train, corrected_features_test,
                    epochs=600, batch_size=500, shuffle=True, verbose=0, callbacks=[checkpointer])
    savetxt('./model/val_loss_{}.txt'.format(i), history.history['val_loss'])








