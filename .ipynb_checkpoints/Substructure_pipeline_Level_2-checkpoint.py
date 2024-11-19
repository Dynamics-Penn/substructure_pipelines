#-----------------------------------
# Importing the packages
#-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import halo_analysis as rockstar
import os
import sys
import time
import h5py
import pickle
from collections import Counter
from tqdm import tqdm
from scipy.spatial import KDTree



#------------------------------------------------------------------------------#
#---------CLASSIFICATION CRITERIA----------------------------------------------#
#------------------------------------------------------------------------------#

#----------------------------------------------------------------
# Criterion on number of star particles at z = 0
#----------------------------------------------------------------
def nstar_cut(st, ncut=100, ncut_max=np.inf):
    '''
    a criterion imposed upon the number of star particles

    input
    st = array of size n, containing indices of star particles
    that belong to a given object

    output
    boolean True or False whether the number of star particles
    is in a desirable range.
    '''

    if len(st) < ncut:
        return False
    if len(st) > ncut_max:
        return False
    return True




#------------------------------------------------
# Distance cut
#------------------------------------------------
def distance_cut(pos_stars, dcut=120):
    '''
    a criterion imposed upon the size of the object. We want the size
    of the object to be greater than dcut in kpc.

    input
    pos_stars = (n,3) array. Positions of all the star particles that
                belong to the object.

    output
    boolean True or False
    '''
    N = len(pos_stars[:,0])
    for i in range(N):
        dis = np.sqrt(((pos_stars - pos_stars[i])**2).sum(axis=1))
        if(np.max(dis) > dcut):
            return True
    return False




#-------------------------------------------------
# Functions for making phase-space cuts
#-------------------------------------------------

# pick out N closest stars in phase-space
def pick_closest_phase_space(pos, vel, N = 20):
    '''
    pick out N closest stars in phase-space to a star particle at
    position pos_hal and velocity vel_hal

    input
    pos = (n,3) array. positons of all star particles in a stream
    pos_hal = (x,y,z) position of the star particle of interest
    vel = (n,3) array. velocities of all star particles in a stream
    vel_hal = (vx,vy,vz) velocity of the star particle of interst

    N = the number of closest neighbors. I usually set N=20 for
    objects with more than 300 star particles and N=7 if the object
    has fewer than 300 star particles.

    output
    indices of the closest N star particles.
    '''

    # compute distance for all stars in the stream
    d_star_hal = np.sqrt(((pos_hal - pos)**2).sum(axis=1))

    # velocity space
    v_star_hal = np.sqrt(((vel_hal - vel)**2).sum(axis=1))

    sigma_x = np.std(d_star_hal)
    sigma_v = np.std(v_star_hal)

    # compute the standardized distances
    d_tot = np.sqrt((d_star_hal**2)/sigma_x**2 + (v_star_hal**2)/sigma_v**2)

    return (d_tot.argsort())[:N]


def pick_closest_phase_space_KDTree(pos, vel, N=20):

    # Combine position and velocity arrays
    data = np.concatenate((pos, vel), axis=1)

    # Build KD-tree
    tree = KDTree(data)

    # Query nearest neighbors for each particle
    nearest_neighbors = []
    for i in range(len(data)):
        dists, indices = tree.query(data[i], k=N)
        # Exclude the first element (self) from the indices
        nearest_neighbors.append(indices[0:].tolist())

    return nearest_neighbors


def pick_closest_real_space(pos, pos_hal, N = 20):
    '''
    pick out N closest stars in real space to a star particle at
    position pos_hal

    input
    pos = (n,3) array. positi
    ons of all star particles in a stream
    pos_hal = (x,y,z) position of the star particle of interest

    N = the number of closest neighbors. I usually set N=20 for
    objects with more than 300 star particles and N=7 if the object
    has fewer than 300 star particles.

    output
    indices of the closest N star particles.
    '''

    # compute distance for all stars in the stream
    d_star_hal = np.sqrt(((pos_hal - pos)**2).sum(axis=1))

    return (d_star_hal.argsort())[:N]




# star index j in the stream
def find_vel_dis_real(j,pos, vel, N=20):
    close = pick_closest_real_space(pos, pos[j], N=N)

    dispersions = np.std(vel[close], axis=0)
    dis = np.linalg.norm(dispersions)

    return dis





def dispersion_cut_threshold(stellar_mass):
    '''
    If the median value of the local velocity dispersion is greater than this
    value, the object is classified as phase-mixed. Otherwise, it is classified
    as a coherent stream.

    input
    stellar_mass = stellar mass of the object in M_sun

    output
    the treshold value

    The values come from Panithanpaisal et al. (2021)
    '''
    cut = -5.275459682837681*np.log10(stellar_mass) + 53.550272108698486
    return cut




def find_vel_dis(i, pos, vel, N=20):
    '''
    finding local velocity dispersion around the i-th particle within
    the object.
    '''
    close = pick_closest_phase_space(pos, pos[i], vel, vel[i], N=N)

    dispersions = np.std(vel[close], axis=0)
    dis = np.linalg.norm(dispersions)

    return dis

def find_vel_dis_KDTree(vel, close):

    dispersions = np.std(vel[close], axis=0)
    dis = np.linalg.norm(dispersions)

    return dis




def dispersion_cut(pos, vel, stellar_mass):
    '''
    a criterion imposed upon the local velocity dispersion of the object.
    we want the median local velocity dispersion to be less than the
    stellar mass dependent threshold value to be classified as a stream.

    input
    pos = (n,3) array. positons of all star particles in a stream
    vel = (n,3) array. velocities of all star particles in a stream
    stellar mass = stellar mass of the object in M_sun

    output
    boolean True or False
    '''
    n = len(pos[:,0])
    N = 20
    if n < 300:
        N = 7

    #local_dispersion = np.zeros(n)
    #for i in range(n):
    #    local_dispersion[i] = find_vel_dis(i, pos, vel, N=N)
    close = pick_closest_phase_space_KDTree(pos, vel, N=N)
    #local_dispersion = Parallel(n_jobs=-1)(delayed(find_vel_dis_KDTree)(vel, close[i]) for i in range(n))
    local_dispersion = np.zeros(n)
    for i in range(n):
        local_dispersion[i] = find_vel_dis_KDTree(vel, close[i])

    # median value is
    med = np.median(local_dispersion)

    return (med < dispersion_cut_threshold(stellar_mass))




#------------------------------------------------------
# Function to perfrom the classification at z = 0
#------------------------------------------------------
def classify_substructures(unclassified, host_no, simdir):
    
    all_st = []
    present_stellar_mass = []
    stream = []
    pm = []
    dwarf_gal = []
    index_at_sample = []
    first_infall_snap = []
    last_infall_snap = []
    first_infall_snap_index = []
    last_infall_snap_index = []

    halo_z0 = rockstar.io.IO.read_catalogs('redshift', 0, simulation_directory=simdir, rockstar_directory=rockdir)
    part_z0 = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simdir, assign_formation_coordinates=True, assign_hosts=True)
    
    if(host_no == 0):
        rad = halo_z0['radius'][np.argmax(halo_z0['star.mass'])]
    else:
        rad = np.sort(halo_z0['radius'])[-2]
        
    obj_inds = []
    len_obj = len(unclassified['index_at_sample'])

    for i in tqdm(range(len_obj)):

        st = unclassified['st'][i]
        print(i, len(st))

        all_st.append(st)
        m_st = part_z0['star']['mass'][st].sum()
        present_stellar_mass.append(m_st)

        index_at_sample.append(unclassified['index_at_sample'][i])
        first_infall_snap.append(unclassified['first_infall_snap'][i])
        first_infall_snap_index.append(unclassified['first_infall_snap_index'][i])
        last_infall_snap.append(unclassified['last_infall_snap'][i])
        last_infall_snap_index.append(unclassified['last_infall_snap_index'][i])

        if(host_no == 0):
            pos = part_z0['star'].prop('host.distance')[st]
            vel = part_z0['star'].prop('host.velocity')[st]
            
        else:
            pos = part_z0['star'].prop('host2.distance')[st]
            vel = part_z0['star'].prop('host2.velocity')[st]

        if nstar_cut(st):

            if(host_no == 0):
                mean_pos = np.mean(part_z0['star'].prop('host.distance.total')[st])
            else:
                mean_pos = np.mean(part_z0['star'].prop('host2.distance.total')[st])

            if(mean_pos <= rad):
                
                if(distance_cut(pos)):

                    print(vel.shape)
                    if(dispersion_cut(pos, vel, m_st)):
                        
                        stream.append(True)
                        pm.append(False)
                        dwarf_gal.append(False)
                        obj_inds.append(i)

                    else:
                        
                        stream.append(False)
                        pm.append(True)
                        dwarf_gal.append(False)
                        obj_inds.append(i)

                else:

                    stream.append(False)
                    pm.append(False)
                    dwarf_gal.append(True)
                    obj_inds.append(i)

            else:
                
                stream.append(False)
                pm.append(False)
                dwarf_gal.append(False)
                obj_inds.append(i)

        else:
            
            stream.append(False)
            pm.append(False)
            dwarf_gal.append(False)
            obj_inds.append(i)

    classified = {}
    
    classified['st'] = np.array(all_st, dtype=object)
    classified['present_stellar_mass'] = np.array(present_stellar_mass, dtype=float)
    classified['index_at_sample'] = np.array(index_at_sample, dtype=int)
    classified['first_infall_snap'] = np.array(first_infall_snap, dtype=int)
    classified['first_infall_snap_index'] = np.array(first_infall_snap_index, dtype=int)
    classified['last_infall_snap'] = np.array(first_infall_snap, dtype=int)
    classified['last_infall_snap_index'] = np.array(first_infall_snap_index, dtype=int)

    classified['stream'] = np.array(stream, dtype=bool)
    classified['dwarf_gal'] = np.array(dwarf_gal, dtype=bool)
    classified['pm'] = np.array(pm, dtype=bool)
    classified['present_stellar_mass'] = np.array(present_stellar_mass, dtype=float)

    return classified.copy()


#-----------------------------------------
# File input/output handling done here
#-----------------------------------------
# location of the outputs to be saved
SAVE_LOC = 'your_path_goes_here'   # Save location of the catalogs (change accordingly)
rockdir = 'halo/rockstar_dm_highZ/'                     # Halo directory (change accordingly)




#---------------------------------------------------------------------------
# Running the classification pipeline to get the classified substructures
#---------------------------------------------------------------------------
def main(simname, host_no):
    
    # simname = Simulation name
    if(simname == 'm12_elvis_RomeoJuliet_res3500' or simname == 'm12_elvis_ThelmaLouise_res4000' or simname == 'm12_elvis_RomulusRemus_res4000'):
        simdir = f'/data11/fire2-pairs/{simname}/'       # Simulation directory
        if(host_no == 0):
            filename = f'{simname}_cdm_classified_lvl_1_host1.pkl'       # Filename of output to be saved
        else:
            filename = f'{simname}_cdm_classified_lvl_1_host2.pkl'       # Filename of output to be saved
    else:
        simdir = f'/data10/fire2/metaldiff/{simname}/'   # Simulation directory
        filename = f'{simname}_cdm_classified_lvl_1.pkl'       # Filename of output to be saved
        
    fsave = SAVE_LOC + filename                      # Location of output

    if(simname == 'm12_elvis_RomeoJuliet_res3500' or simname == 'm12_elvis_ThelmaLouise_res4000' or simname == 'm12_elvis_RomulusRemus_res4000'):
        if(host_no == 0): 
            unclassified_filename = f'{simname}_cdm_unclassified_lvl_2_host1.pkl'
        else:
            unclassified_filename = f'{simname}_cdm_unclassified_lvl_2_host2.pkl'
    else:
        unclassified_filename = f'{simname}_cdm_unclassified_lvl_2.pkl'     # Unclassified file
        
    unclassified_fsave = SAVE_LOC + unclassified_filename         # Directory of the unclassified file
    
    # Loading the unclassified catalog
    with open(unclassified_fsave, 'rb') as f:
        unclassified_catalog = pickle.load(f)
    
    classified_catalog = {}
    classified_catalog = classify_substructures(unclassified_catalog.copy(), host_no, simdir)
    
    
    #----------------------------------------
    # Saving the classified level 1 catalog
    #----------------------------------------
    with open(fsave, 'wb') as file:
        pickle.dump(classified_catalog, file)
        file.close()
    
    
    
    
if __name__ == "__main__":
    
    if(len(sys.argv) > 3):
        print("Enter in the following way: python Substructure_pipeline_Level_2.py <simname> <host_no>")
        sys.exit(1)        # Exit code

    # If the user forgets to provide arguments for start_snap, snap_interval, host_no, then their default values will be used
    elif(len(sys.argv) == 2):
        simname = sys.argv[1]
        host_no = 0
        
    else:
        simname = sys.argv[1]
        host_no = int(sys.argv[2])
    
    main(simname, host_no)
