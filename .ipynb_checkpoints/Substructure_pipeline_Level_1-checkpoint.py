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




#------------------------------------
# The halo tracking functions
#------------------------------------
def find_main(halo_tree, last_snap=600, host_no=0):
    '''
    return merger tree indices of the main halo (at z=0) across all snapshots
    
    halo_tree = halo merger tree
    host_no = 0 for isolated sims; 0 and 1 for paried sims
    '''
    if host_no == 0:
        # what is the tree index at the last snapshot
        main_tid_ls = np.where((halo_tree['snapshot'] == last_snap) & 
                               (halo_tree.prop('host.distance.total') == 0))[0][0]
        
    else:
        main_tid_ls = np.where((halo_tree['snapshot'] == last_snap) & 
                               (halo_tree.prop('host2.distance.total') == 0))[0][0]
    
    main_tids = np.flip(halo_tree.prop('progenitor.main.indices', main_tid_ls))
    
    return main_tids



def find_hal_ind_backward(halo_tree, halo_tid, last_snap=600):
    '''
    find merger tree indices of the progenitor of subhalo halo_tid
    in all previous snapshots
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    prog_halo_tids = np.flip(halo_tree.prop('progenitor.main.indices', halo_tid))
    
    return prog_halo_tids


    
def find_hal_ind_forward(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''
    find merger tree indices of the descendant of subhalo halo_tid in all
    subsequent snapshots until merging with the host
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    desc_halo_tids = np.flip(np.setdiff1d(halo_tree.prop('descendant.indices', halo_tid), 
                                          find_main(halo_tree, last_snap, host_no)))
    
    return desc_halo_tids



def find_hal_ind_all(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''find merger tree indicies of both the descendant and progenitor 
    of the subhalo halo_tid
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    ind_progs = find_hal_ind_backward(halo_tree, halo_tid, last_snap)
    ind_descs = find_hal_ind_forward(halo_tree, halo_tid, last_snap, host_no)
    
    return np.append(ind_progs, ind_descs[1:])



def find_hal_index_at_snap(halo_tree, index_start, end_snap):
    '''
    return halo merger tree index at snapshot end_snap
    
    halo_tree = halo merger tree
    index_start = merger tree index of the subhalo
    end_snap = snapshot where we want to find the merger tree index
               of the subhalo
    '''
    
    start_snap = halo_tree['snapshot'][index_start]
    ind = index_start
    
    if end_snap == start_snap:
        return index_start
    elif end_snap < start_snap:
        while start_snap != end_snap:
            ind = halo_tree['progenitor.main.index'][ind]
            start_snap = halo_tree['snapshot'][ind]
        return ind
    else:
        while start_snap != end_snap:
            ind = halo_tree['descendant.index'][ind]
            start_snap = halo_tree['snapshot'][ind]
        return ind



def find_infall_snapshots(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''
    find infall snapshots for the subhalo with merger tree index halo_tid
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo of interest
    
    return
    1) a list of infalling snapshots.
    empty list = does not fall into the host
    length 1 = one infall
    length greater than 1 = multiple infalls

    2) corresponding indices of the subhalo at infalling snaps
    '''

    main = find_main(halo_tree, last_snap=last_snap, host_no=host_no) # indices of the main halo
    main_r = halo_tree['radius'][main]                                # virial radii of the main halo
    # pad zeros to the front so that the array index corresponds to snapshot
    to_pad = last_snap + 1 - len(main)
    main_r = np.append(np.zeros(to_pad, dtype=int), main_r)
    
    # finding all indices of the subhalo
    halo_tids = find_hal_ind_all(halo_tree, halo_tid, last_snap=last_snap, host_no=host_no)
    
    # differences between central distance of the subhalo and virial radius of the host
    # at all snapshots where the subhalo exists
    if host_no == 0:
        diff = halo_tree.prop('host.distance.total')[halo_tids] - main_r[halo_tree['snapshot'][halo_tids]]
    else:
        diff = halo_tree.prop('host2.distance.total')[halo_tids] - main_r[halo_tree['snapshot'][halo_tids]]

    # looping over to find infall snapshots
    infall_snaps = []
    infall_snaps_i = []
    for i in range(1, len(diff), 1):
        if (diff[i] < 0) and (diff[i-1] > 0):
            infall_snaps.append(halo_tree['snapshot'][halo_tids[i]])
            infall_snaps_i.append(halo_tids[i])

    return infall_snaps, infall_snaps_i


#------------------------------------------
# Star particle tracking functions
#------------------------------------------
def find_present_stars_ind(simdir, snap, st_i, last_snap=600):
    '''
    find indices of stars at present day (aka snap = last_snap)
    
    sim = simulation name
    snap = snapshot of the star particles
    st_i = indices of star particles in snapshot snap
    last_snap = present day snapshot
    '''
    if snap == last_snap:
        return st_i
    try:
        # execute this when the simulations don't have star_gas_pointers_XYZ.hdf5
        #print('opening:', simdir + 'track/star_indices_{:03}.hdf5'.format(snap))
        with h5py.File(simdir + 'track/star_gas_pointers_{:03}.hdf5'.format(snap), 'r') as pt:
            bool_stream = np.isin(pt['z0.to.z.index'][:],st_i)
            #print('success')
        ind = np.where(bool_stream)[0] #gives the positions of the stars we are tracking in the list at snap 600

    except Exception as e:
        print(e)
        part = gizmo.io.Read.read_snapshots(['star'], 'index', snap, simdir, assign_pointers=True);
        pointers = part.Pointer.get_pointers(species_name_from='star', species_names_to='star', forward=True)
        ind = pointers[st_i]
        #part = read_part(sim, snap, pointer=True)
    return ind



# samplings has to be in increasing order or we can sort
def recover_stars_union_complete(simdir, halo_tree, halo_tid, samplings=None, last_snap=600, snapshot_interval=10, host_no=0):
    '''
    recovering stars of a subhalo by tracking it for its entire evolution (if samplings = None) and sample stars
    for every snapshot_interval snapshots. 

    sim = simulation name (i.e. 'm12i_res7100')
    halo_tree = merger tree with star particle information
    halo_tid = merger tree index of subhalo at any snapshot
    samplings = a list of snapshots that we want to sample from (i.e. [350, 360, 370, 380]). 
                If None, sample from the entire evolution with interval set by snapshot_interval
    last_snap = 600 for latte simulations, 60 for sidm simulations
    '''
    
    # if tracking has no problem, sample stars from peak stellar mass snapshots and every snapshot_interval snapshot
    if samplings is None:
        #print ("There are no samplings", )
        # find merger tree indices of the subhalo at all snapshots
        halo_tids =  find_hal_ind_all(halo_tree, halo_tid, last_snap=last_snap, host_no=host_no)

        # find the snapshot of the peak stellar mass
        peak_index = halo_tree['star.mass'][halo_tids].argsort()[::-1][0]
        ind = np.arange(0, len(halo_tids), snapshot_interval)

        # shifting these around such that the intervals contain peak stellar mass snapshot
        while peak_index not in ind:
            ind += 1
        if ind[-1] >= len(halo_tids):
            ind = ind[:-1]
        ind = halo_tids[ind]
    # if there is a problem, only sample from the fixed set of snapshots that we manually specify.
    else:
        #print ("We are sampling from a fixed set")
        ind = []
        for snap in samplings:
            ind.append(find_hal_index_at_snap(halo_tree, halo_tid, snap))

    
    to_hold = [] # indices of star particles to hold
    all_ind = []
    #print(f'Running j loop for {ind}')
    for j in ind:
        # catching bug for the sidm m12b
        if simdir == 'm12b7e3_sidm1': # this will not work be careful
            if halo_tree['snapshot'][j] in [109,110,111,112,113,114]:
                continue
        # check if the subhalo contain star particles at this snapshot
        if halo_tree['star.mass'][j] > 0:
            #print('Stars found')
            snap = halo_tree['snapshot'][j]
            st_i = halo_tree['star.indices'][j]
            
            # get star particle indices at present day
            ind_j = find_present_stars_ind(simdir, snap, st_i, last_snap=last_snap)
            # if to_hold is empty, populate it
            
            if len(to_hold) == 0:
                to_hold = ind_j
                if len(ind) <=1:
                    #print(f'Return recovered stars just from one snapshot')
                    return to_hold
                
            # otherwise append star particles that are both in to_hold and not in all_ind
            else:
                # mask for stars that are not in the final list
                boo1 = np.logical_not(np.isin(ind_j, all_ind))
                # only pick stars that are both part of the subhalo at both snapshot snap and snap-snapshot_interval
                # in other words, is part of the subhalo for at least snapshot_interval snapshots
                boo2 = np.isin(ind_j, to_hold)
                boo = np.where(boo1 & boo2)[0]
                all_ind.extend(ind_j[boo])
                to_hold = ind_j

    return all_ind



#-------------------------------------------------------
# Function to find the substructures at the present day
#-------------------------------------------------------
def find_streams(simdir, last_snap, start_snap, snap_interval=10,  host_no=0, halo_tree=None):
    '''
    find streams in FIRE simulations
    simdir = simulation name
    last_snap = last snapshot of that corresponding simulation
    rad = distance of the bound objects that will be considered stream candidates
    '''

    #pick snapshots to sample stream candidates.
    #Stream candidates are defined as bound objects within ~350kpc (defined by rad) of the host galaxy
    #between certain redshifts (snapshots)
    if last_snap == 600:
        start_snaps = ([(k) for k in range(start_snap,last_snap,snap_interval)])  # changed to sample from every 10
#     elif last_snap == 60: # for the sidm simulations with only 60 snapshots stored
#         start_snaps = [30, 35, 40, 45] # disregard this
    else:
        raise ValueError()

    if simdir is None:
        raise Exception("No simulation directory has been provided")

    if halo_tree is None:
        halo_tree = rockstar.io.IO.read_tree(simulation_directory=simdir, rockstar_directory = rockdir, species='star')

    # read in star particles at present day
    part_z0 = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simdir, assign_formation_coordinates=True)
    
    # find indices of the main halo across all snapshots
    main = find_main(halo_tree, last_snap, host_no)
    rad = halo_tree['radius'][main[-1]]
    
    if rad < 0:
        rad = int(halo_tree['radius'][main[-1]])
    print('--------------------------------------------------------------------------', flush=True)
    print('---------------  searching for halos within {} kpc  ----------------------'.format(rad), flush=True)
    print('--------------------------------------------------------------------------', flush=True)

    # what information to save
    all_st = []
    max_stellar_snap_index = []
    max_stellar_mass = []
    max_stellar_total_mass = []
    present_stellar_mass = []
    oldest_snap_index = []
    index_at_sample = []
    stream = []
    pm = []
    dwarf = []
    disk_star_removed = []
    first_infall_snap = []
    last_infall_snap = []
    first_infall_snap_index = []
    last_infall_snap_index = []
    
    if(host_no == 0):
    
        hindices_condition1 = halo_tree.prop('host.distance.total') > 0
        hindices_condition2 = halo_tree.prop('host.distance.total') < rad
        
    else:

        hindices_condition1 = halo_tree.prop('host2.distance.total') > 0
        hindices_condition2 = halo_tree.prop('host2.distance.total') < rad

    for start_snap in tqdm(start_snaps):
    #------------------------------------------- pick out only subhalos with stellar mass----------------------------
        #print('doing snapshot {}'.format(start_snap), flush=True)
        
        
        hindices = np.where((halo_tree['snapshot'] == start_snap) & 
                            (hindices_condition1) & (hindices_condition2))[0]


        h_with_stars = np.where((np.isnan(halo_tree['star.mass'][hindices]) == False)
                               & (halo_tree['star.mass'][hindices] > 0))[0] # with stellar mass
        h_with_stars = hindices[h_with_stars]

        # tracking each subhalo finding its index at peak stellar mass timescales

        # select star particles from infall snapshots instead
        for count, i in enumerate(h_with_stars):
            #print('doing {}/{}'.format(count, len(h_with_stars)))

            # add infall times
            infall_snaps, infall_snaps_i = find_infall_snapshots(halo_tree, i, last_snap=last_snap, host_no=host_no)
            if len(infall_snaps) == 0:
                continue
            elif halo_tree['star.mass'][infall_snaps_i[0]] < 0:
                continue
            elif infall_snaps_i[0] in first_infall_snap_index:
                continue
            else:
                first_infall_snap.append(infall_snaps[0])
                last_infall_snap.append(infall_snaps[-1])
                first_infall_snap_index.append(infall_snaps_i[0])
                last_infall_snap_index.append(infall_snaps_i[-1])

            # recover all stars
            ii_max = infall_snaps_i[0]
            st_recov = np.array(recover_stars_union_complete(simdir, halo_tree, ii_max, samplings=None, snapshot_interval=snap_interval, last_snap=last_snap, host_no=host_no))
            
            #print(f'# Stars found: {len(st_recov)}')
            all_st.append(st_recov)
            index_at_sample.append(i)
            
            # storing the unclassified catalog
            tree = {'st':np.array(all_st, dtype=object),
                    'index_at_sample':np.array(index_at_sample, dtype=int),
                    'first_infall_snap':np.array(first_infall_snap, dtype=int),
                    'last_infall_snap':np.array(last_infall_snap, dtype=int),
                    'first_infall_snap_index':np.array(first_infall_snap_index, dtype=int),
                    'last_infall_snap_index':np.array(last_infall_snap_index, dtype=int)
                    }

    return tree




#-----------------------------------------
# File input/output handling done here
#-----------------------------------------
# location of the outputs to be saved
SAVE_LOC = 'your_path_goes_here'   # Save location of the catalogs (change accordingly)
rockdir = 'halo/rockstar_dm_highZ/'                     # Halo directory (change accordingly)

#------------------------------------------------------------
# Running the pipeline to get the unclassified substructures
#------------------------------------------------------------

def main(simname, start_snap, snap_interval, host_no):
    
    # simname = Simulation name
    if(simname == 'm12_elvis_RomeoJuliet_res3500' or simname == 'm12_elvis_ThelmaLouise_res4000' or simname == 'm12_elvis_RomulusRemus_res4000'):
        simdir = f'/data11/fire2-pairs/{simname}/'       # Simulation directory
        if(host_no == 0):
            filename = f'{simname}_cdm_unclassified_host1.pkl'     # Filename of output to be saved
        else:
            filename = f'{simname}_cdm_unclassified_host2.pkl'     # Filename of output to be saved
    else:
        simdir = f'/data10/fire2/metaldiff/{simname}/'   # Simulation directory
        filename = f'{simname}_cdm_unclassified.pkl'     # Filename of output to be saved
    
    # host_no = 0 for all isolated sims and one host of paired sims; 1 for the other host
    
    fsave = SAVE_LOC + filename                      # Location of output
    
    last_snap = 600                                  # Final snapshot
    substructure_tree = find_streams(simdir, last_snap, start_snap, snap_interval, host_no)



    # A bit of post-processing
    # Cleaning the unclassified catalog a bit
    unclassified_catalog = {}
    obj_inds = []

    for i in range(len(substructure_tree['st'])):

        if(len(substructure_tree['st'][i]) > 0):
            obj_inds.append(i)

    obj_inds = np.array(obj_inds)

    for keys in substructure_tree.keys():

        if(substructure_tree[keys].size > 0):
            unclassified_catalog[keys] = substructure_tree[keys][obj_inds]

    #-------------------------------------
    # Saving the unclassified catalog
    #-------------------------------------
    with open(fsave, 'wb') as file:
        pickle.dump(unclassified_catalog, file)
        file.close()
    
if __name__ == "__main__":
    
    if(len(sys.argv) > 5 or len(sys.argv) < 2):
        print("Enter in the following way: python Substructure_pipeline_Level_1.py <simname> <start_snap> <snap_interval> <host_no>")
        sys.exit(1)        # Exit code
    
    # If the user forgets to provide arguments for start_snap, snap_interval, host_no, then their default values will be used
    elif(len(sys.argv) == 2):
        simname = sys.argv[1]
        start_snap = 195
        snap_interval = 10
        host_no = 0
        
    elif(len(sys.argv) == 3):
        simname = sys.argv[1]
        start_snap = int(sys.argv[2])
        snap_interval = 10
        host_no = 0
    
    elif(len(sys.argv) == 4):
        simname = sys.argv[1]
        start_snap = int(sys.argv[2])
        snap_interval = int(sys.argv[3])
        host_no = 0
        
    else:
        simname = sys.argv[1]
        start_snap = int(sys.argv[2])
        snap_interval = int(sys.argv[3])
        host_no = int(sys.argv[4])
        
    
    main(simname, start_snap, snap_interval, host_no)
    
    
 