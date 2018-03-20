from read_xyz import ReadXYZ
from read_gsd import ReadGSD

def ReadTraj(filename, traj_type, shuffle_data, randomize, remove_types):
    if traj_type == 'xyz':
        return ReadXYZ(filename, shuffle_data, randomize, remove_types)
    elif traj_type == 'gsd':
        return ReadGSD(filename, shuffle_data, randomize, remove_types)
    else:
        raise Exception('{} is not a supported trajectory file format'.format(traj_type))
