import numpy as np
import torch
import torch.nn
import torch.utils

from .loss import calculate_cpc_loss


class SubsampleBlock(torch.nn.Module):
    def __init__(self, n_channels, n_mid_channels=64, n_out_channels=64):
        super().__init__()

        norm = torch.nn.utils.spectral_norm
        self.seq = torch.nn.Sequential(
            norm(
                torch.nn.Conv2d(
                    n_channels, n_mid_channels, kernel_size=3, padding=2, dilation=2
                )
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Mish(),
            torch.nn.BatchNorm2d(n_mid_channels // 2),
            norm(
                torch.nn.Conv2d(
                    n_mid_channels // 2,
                    n_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            torch.nn.Mish(),
            torch.nn.BatchNorm2d(n_out_channels),
        )

    def forward(self, x):
        return self.seq(x)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, n_channels=1, temporal_length=20):
        super().__init__()

        self.temporal_length = temporal_length

        n_mid_channels = 64
        norm = torch.nn.utils.spectral_norm

        embedding_size = 64
        hidden_state_size = 64
        f = 2
        self.embedding = torch.nn.Sequential(
            # 128
            SubsampleBlock(n_channels, 32, 64 // f),  # 64
            SubsampleBlock(64 // f, 64 // f, 128 // f),  # 32
            SubsampleBlock(128 // f, 128 // f, 256 // f),  # 16
            SubsampleBlock(256 // f, 256 // f, 512 // f),  # 8
            SubsampleBlock(512 // f, 128, embedding_size),  # 4
            torch.nn.AvgPool2d(4),
            torch.nn.LeakyReLU(),
        )

        self.lstm = torch.nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_state_size, batch_first=False
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_state_size, hidden_state_size // 2),
            torch.nn.Mish(),
            torch.nn.BatchNorm1d(hidden_state_size // 2),
            torch.nn.Linear(hidden_state_size // 2, embedding_size),
            torch.nn.LeakyReLU(),
        )

        self.distance_metric = torch.nn.CosineSimilarity(dim=1)

    def embed(self, images):
        return self.embedding(images)

    def forward(self, images):
        batch_size = images.shape[0]
        assert images.shape[1] == self.temporal_length
        embeddings = []
        for i in range(2):  # range(self.temporal_length):
            e = self.embed(images[:, i : (i + 1), :, :])
            embedding_size = e.shape[1]
            assert e.shape[2] == 1 and e.shape[3] == 1
            e = e[:, :, 0, 0]

            embeddings.append(e)
        embeddings = torch.stack(embeddings, dim=0)
        # print(e.shape, hidden_states.shape if hidden_states is not None else None)
        lstm_state, hidden_states = self.lstm(embeddings)
        lstm_state = lstm_state[-1]  # Last sequence state.

        predictions = self.predictor(lstm_state)

        return predictions

    def run_batch(self, images):
        batch_size, L = images.shape[:2]
        base = images[:, :-1, :, :]
        target = images[:, -1:, :, :]

        target_embedding = self.embed(target)
        target_embedding = target_embedding[
            :, :, 0, 0
        ]  # Collapse dimensions of length 1.
        # embedding_size = target_embedding.shape[1]
        predictions = self(base)
        predictions = torch.swapaxes(predictions, 0, 1)

        losses = calculate_cpc_loss([target_embedding], [predictions])

        """
        positive_predictions = self.distance_metric(
            predictions[:, 0], predictions[:, 1]
        )
        negative_predictions = self.distance_metric(
            predictions[:-1, 0], predictions[1:, 0]
        )
        assert positive_predictions.shape[0] == batch_size
        assert (
            len(positive_predictions.shape) == 1 or positive_predictions.shape[1] == 1
        )

        cpc_loss = torch.mean(negative_predictions) - torch.mean(positive_predictions)
        cpc_loss = torch.exp(cpc_loss)

        return {"cpc_loss": cpc_loss}
        """

        return losses
