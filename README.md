# substructure_pipelines
Contains the pipelines to track and classify disrupted substructures at level 1 and level 2.

Level 1 - 
(Substructure_pipeline_Level_1.py) - Tracks the subhalos to the present-day and makes a catalog of the substructures without any classification.

How to run the Python script? python Substructure_pipeline_Level_1.py <simname> <start_snapshot> <snapshot_interval> <host_no>

simname = Enter the simulation name, e.g., m12i_res7100, m12_elvis_RomeoJuliet_res3500, etc.
start_snapshot = The snapshot from where you want the pipeline to start tracking the subhalos.
snapshot_interval = The sampling interval between two snapshots.
host_no = 0 for all isolated sims and the most massive halo in paired sims; 1 for the second most massive halo in paired sims.




Level 2 - (Substructure_pipeline_Level_2.py) - Reads the unclassified catalog, uses the classification scheme from Panithanpaisal et al. (2021) and makes a (semi)classified catalog.

How to run the Python script? python Substructure_pipeline_Level_2.py <simname> <host_no>

simname = Enter the simulation name, e.g., m12i_res7100, m12_elvis_RomeoJuliet_res3500, etc.
host_no = 0 for all isolated sims and the most massive halo in paired sims; 1 for the second most massive halo in paired sims.




Level 3 - Up to the user. The level 2 code will generate a classfified catalog according to the classification algorithm. But some manual reclassification might be needed depending on the use case.

Up to level 1 and 2, one can just run the script with the necessary changes in the parameters and path to files. It will be lot more efficient to run the scripts using bash if you are running them for multiple simulations on the cluster. 
