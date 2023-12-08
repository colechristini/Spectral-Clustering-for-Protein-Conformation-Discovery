using NPZ, LinearAlgebra, Clustering

# Load the precomputed affinity matrix of the RMSD data
A = npzread("affinity_matrix.npy")

# Load the RMSD data
dist = npzread("rmsd_matrix.npy")

# Construct the inverse square root degree matrix to normalize A
D = zeros(Float64, size(A))
sums = sum(A, dims=2)
for i in 1:size(A, 1)
    D[i,i] = 1 / sqrt(sums[i])
end

# Multiply to get the normalized Laplacian
L = D * A * D

# Compute the eigenvectors and eigenvalues of L
eig = eigen(L)

# Chosen (semi)-arbitrarily based on manual visual
# analysis of the RMSD data. This is not the most principled
# way to do it, but this work is primarily proof of concept.
k = 5

# The eigenvectors are already sorted in increasing order
# We select the first k eigenvectors with the smallest eigenvalues
evecs = eig.vectors[:,1:k]

# K-means uses columns as datapoints, so we transpose because we want to
# cluster the rows of the matrix made up of the first k eigenvectors.
clustering_matrix = transpose(evecs)

# Cluster the rows, and extract the assignments. The cluster each row
# (now column) is assigned to corresponds to the cluster the frame with
# the same index is assigned to.
clusters = Clustering.kmeans(clustering_matrix, k)
cluster_assignments = assignments(clusters)

# Construct a dictionary with clusters as keys and the frames assigned to
# the cluster in a list as the value.
assignment_dict = Dict{Int, Vector{Int}}()
for i in eachindex(cluster_assignments)
    if !(haskey(assignment_dict, cluster_assignments[i]))
        assignment_dict[cluster_assignments[i]] = Vector{Int}()
    end
    append!(assignment_dict[cluster_assignments[i]], i)
end

# In order to find the 'most representative frame' of each cluster, we
# compute the medoid - the frame who's sum of RMSD to all other frames
# in the cluster is minimized.
medoid = zeros(Int,k)
for k in keys(assignment_dict)
    assigned_elements = assignment_dict[k]
    distances = Vector{Float64}()
    # Compute the sum of RMSDs for each element in the cluster
    for i in assigned_elements
        sum = 0.0
        for j in assigned_elements
            sum += dist[i,j]
        end
        append!(distances, sum)
    end
    # Determine the frame with minimal sum.
    medoid[k] = assigned_elements[argmin(distances)]
end

print(medoid)










