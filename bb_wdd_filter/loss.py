import torch


def calculate_cpc_loss(encodings, predictions, detach_accuracies=True):

    assert isinstance(encodings, list)
    assert isinstance(predictions, list)

    n_timesteps = len(encodings)
    batch_size = encodings[0].shape[0]

    nce_loss = 0.0
    accuracies = []

    for i in range(n_timesteps):
        encoding = encodings[i]
        prediction = predictions[i]

        projections = torch.mm(encoding, prediction)
        assert projections.shape[0] == batch_size
        assert projections.shape[1] == batch_size

        logs_projections = torch.nn.functional.log_softmax(projections, dim=1)

        # Count the number of times the highest element is on the diagonal.
        hits = logs_projections.argmax(dim=0) == torch.arange(
            batch_size, device=logs_projections.device
        )
        hits = hits.float().mean()
        if detach_accuracies:
            hits = hits.detach()

        accuracies.append(hits)

        # Now the InfoNCE loss.
        nce = torch.diag(logs_projections).mean()

        nce_loss += -1.0 * nce / n_timesteps

    losses = {f"acc_t{i}": acc for (i, acc) in enumerate(accuracies)}
    losses["nce_loss"] = nce_loss

    return losses
