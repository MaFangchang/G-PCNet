import torch
from torch_scatter import scatter_mean


def offset_to_batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)


def node_to_batch(num_nodes):
    if num_nodes.dim() == 0:
        batches = torch.repeat_interleave(torch.tensor([0]).to(num_nodes.device), num_nodes)
    else:
        batches = torch.repeat_interleave(torch.arange(num_nodes.size(0)).to(num_nodes.device), num_nodes)

    return batches


def graph_to_point(original_embedding, offset):
    offset = torch.diff(offset, prepend=torch.tensor([0]).to(offset.device))
    return torch.repeat_interleave(original_embedding, offset, dim=0)


def face_to_point(original_embedding, sequence):
    """
    A function to align face embedding tensors with point cloud tensors.

    1.Calculate the repeat counts for each row.
    2.Since 'sequence' represents the row number to repeat to, we need to calculate the difference between
    adjacent elements.
    """

    repeat_counts = torch.empty_like(sequence)
    repeat_counts[0] = sequence[0]
    repeat_counts[1:] = sequence[1:] - sequence[:-1]

    embedding = torch.repeat_interleave(original_embedding, repeat_counts, dim=0)

    return embedding


def segmentation_voter(seg_pred, sequence):
    seg_pred_per_face = scatter_mean(seg_pred, sequence, dim=0)

    return seg_pred_per_face
