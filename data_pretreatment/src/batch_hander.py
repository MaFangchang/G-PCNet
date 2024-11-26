import numpy as np


def normalize_data(data, epsilon=1e-10):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ranges = max_vals - min_vals
    ranges_nonzero = np.where(ranges != 0, ranges, epsilon)

    normalized_data = (data - min_vals) / ranges_nonzero
    normalized_data -= 0.5

    return normalized_data


def point_labels_map(labels, sequence):
    repetitions = np.diff(np.insert(sequence, 0, 0))

    labels_map = np.repeat(labels, repetitions, axis=0)

    return labels_map


def point_batches_map(batches):
    repetitions = np.diff(np.insert(batches, 0, 0))

    batches_map = np.repeat(np.arange(len(batches)), repetitions, axis=0)

    return batches_map


def expand_matrix(mat, seq):
    n = sum(seq)
    expanded_mat = np.zeros((n, n))
    indices = np.cumsum(seq)

    for i, s in enumerate(seq):
        start_i = indices[i] - s
        end_i = indices[i]
        expanded_mat[start_i:end_i, start_i:end_i] = mat[i, i]

        for k, sk in enumerate(seq):
            if k != i:
                start_k = indices[k] - sk
                end_k = indices[k]
                expanded_mat[start_i:end_i, start_k:end_k] = mat[i, k]
                expanded_mat[start_k:end_k, start_i:end_i] = mat[k, i]

    return expanded_mat


def pad_and_combine_matrices(matrices, pad_value=0):
    """
    Fills all matrices in the list to the same size as the largest matrix and combines them into an array

    Parameters.
    matrices (list of np.ndarray): list of input matrices
    pad_value (int, optional): the value to pad, default is 0

    Returns: combined_matrices (np.ndarray)
    combined_matrices (np.ndarray): a three-dimensional array of the filled matrices.
    """

    max_size = max(matrix.shape[0] for matrix in matrices)
    num_matrices = len(matrices)

    combined_matrices = np.full((num_matrices, max_size, max_size), pad_value)

    for i, matrix in enumerate(matrices):
        current_size = matrix.shape[0]
        combined_matrices[i, :current_size, :current_size] = matrix

    return combined_matrices


def expand_and_pad_matrices(instance_labels_list, sequence, num_nodes):
    repetitions = np.diff(np.insert(sequence, 0, 0))

    # Group sequence by node_num
    grouped_sequence = []
    index = 0
    for num in num_nodes:
        grouped_sequence.append(repetitions[index:index + num])
        index += num

    expanded_instance = []
    instance = []
    for mat, seq in zip(instance_labels_list, grouped_sequence):
        expanded_mat = expand_matrix(mat, seq)
        expanded_instance.append(expanded_mat)

        instance.append(mat)

    instance = pad_and_combine_matrices(instance)

    return expanded_instance, instance


if __name__ == "__main__":
    # Input instances and sequences
    inst = [
        np.array([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]),
        np.array([[1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
    ]

    sequ = np.array([2, 5, 7, 10, 13, 16, 19, 23, 25, 27, 29, 32])
    node_n = np.array([4, 3, 5])

    # Expand and fill the matrix and get the filled matrix of the original matrix
    expanded_padded_matrices, padded_original_matrices = expand_and_pad_matrices(inst, sequ, node_n)

    print("Expanded and Padded Matrices:")
    print(expanded_padded_matrices)

    print("\nPadded Original Matrices:")
    print(padded_original_matrices)
