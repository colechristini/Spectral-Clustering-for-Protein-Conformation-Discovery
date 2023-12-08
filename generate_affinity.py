import numpy as np

# Converts an RMSD distance matrix to an affinity matrix (where RMSD 0 is mapped to 1).
def main():
    matrix = np.load('rmsd_matrix.npy')
    matrix = 1-(matrix/np.max(matrix))
    matrix[matrix <= 0.1] = 0
    np.save('affinity_matrix.npy', matrix)

main()