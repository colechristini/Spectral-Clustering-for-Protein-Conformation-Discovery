import MDAnalysis as mda
from MDAnalysis.analysis import align, diffusionmap
import matplotlib as plt
import numpy as np

def main():
    topology = '/Users/colechristini/Desktop/CaTK/CATK_ionized_center.psf'
    trajectories = ['/Users/colechristini/Desktop/CaTK/out/CaTK_WT_APO_eq.dcd',
                    '/Users/colechristini/Desktop/CaTK/out/CaTK_WT_APO_eq_2.dcd',
                    '/Users/colechristini/Desktop/CaTK/out/CaTK_WT_APO_eq_3.dcd',
                    '/Users/colechristini/Desktop/CaTK/out/CaTK_WT_APO_eq_4.dcd']
    u = mda.Universe(topology, *trajectories)
    ref = u.copy()
    u.trajectory[-1]
    ref.trajectory[0]
    aligner = align.AlignTraj(u, ref, select='protein and (resnum 31:35 or resnum 45:50 or resnum 87:91 or resnum 95:105 or resnum 128:131 or resnum 148:152 or resnum 165:168)',
                              filename='aligned_to_first_frame.dcd').run()
    u = mda.Universe(topology, 'aligned_to_first_frame.dcd')
    matrix = diffusionmap.DistanceMatrix(u, select='backbone and (resnum 35:45 or resnum 103:128 or resnum 146:168)').run()
    np.savetxt('rmsd_matrix.txt', matrix.dist_matrix)
    np.save('rmsd_matrix.npy', matrix.dist_matrix)
            

    plt.imshow(matrix.dist_matrix, cmap='viridis')
    plt.xlabel('Frame')
    plt.ylabel('Frame')
    plt.colorbar(label=r'RMSD ($\AA$)')
    for i in [1000, 2000, 3000]:
        plt.axhline(i)
        plt.axvline(i)
    plt.show()

main()