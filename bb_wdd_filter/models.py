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
                    n_channels, n_mid_channels, kernel_size=3, padding=1, dilation=1
                )
            ),
            torch.nn.GroupNorm(8, n_mid_channels),
            torch.nn.GLU(dim=1),
            torch.nn.Mish(),
            # torch.nn.BatchNorm2d(n_mid_channels // 2),
            norm(
                torch.nn.Conv2d(
                    n_mid_channels // 2,
                    n_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            torch.nn.GroupNorm(8, n_out_channels),
            torch.nn.Mish(),
            # torch.nn.BatchNorm2d(n_out_channels),
        )

    def forward(self, x):
        return self.seq(x)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, n_channels=1, temporal_length=15, n_targets=3):
        super().__init__()

        self.temporal_length = temporal_length
        self.n_targets = n_targets

        n_mid_channels = 64
        norm = torch.nn.utils.spectral_norm

        embedding_size = 256
        hidden_state_size = 64
        f = 2
        self.embedding = torch.nn.Sequential(
            # 128
            SubsampleBlock(n_channels, 32, 96 // 2 // f),  # 64
            SubsampleBlock(96 // 2 // f, 128 // f, 128 // f),  # 32
            SubsampleBlock(128 // f, 128 // f, 256 // f),  # 16
            SubsampleBlock(256 // f, 256 // f, 512 // f),  # 8
            SubsampleBlock(512 // f, embedding_size, 2 * embedding_size),  # 4
            norm(
                torch.nn.Conv2d(
                    2 * embedding_size,
                    embedding_size,
                    kernel_size=4,
                )
            ),
        )

        self.lstm = torch.nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_state_size, batch_first=False
        )

        self.predictors = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    # torch.nn.Linear(hidden_state_size, hidden_state_size // 2),
                    # torch.nn.GroupNorm(1, hidden_state_size // 2),
                    # torch.nn.Mish(),
                    torch.nn.Linear(hidden_state_size, embedding_size),
                    # torch.nn.LeakyReLU(),
                )
                for _ in range(self.n_targets)
            ]
        )

        self.direction_vector_regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_state_size, hidden_state_size * 2),
            torch.nn.GroupNorm(8, hidden_state_size * 2),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_state_size * 2, 2),
            torch.nn.Tanh(),
        )

        self.vector_similarity = torch.nn.CosineSimilarity(dim=1)

    def predict_waggle_direction(self, hidden_state):
        directions = self.direction_vector_regressor(hidden_state)
        lengths = torch.linalg.vector_norm(directions, dim=1) + 1e-3
        directions = directions / lengths.unsqueeze(1)
        return directions

    def embed(self, images):
        return self.embedding(images)

    def embed_sequence(self, images, return_full_state=False, check_length=True):

        assert (
            (not check_length)
            or (self.temporal_length is None)
            or (images.shape[1] == self.temporal_length)
        )

        temporal_length = self.temporal_length or images.shape[1]

        embeddings = []
        for i in range(temporal_length):
            e = self.embed(images[:, i : (i + 1), :, :])
            embedding_size = e.shape[1]
            assert e.shape[2] == 1 and e.shape[3] == 1
            e = e[:, :, 0, 0]
            embeddings.append(e)

        embeddings = torch.stack(embeddings, dim=0)
        lstm_state, hidden_states = self.lstm(embeddings)
        if not return_full_state:
            lstm_state = lstm_state[-1]  # Last sequence state.

        return lstm_state

    def forward(self, images):

        sequential_embeddings = self.embed_sequence(images)
        predictions = [
            predictor(sequential_embeddings) for predictor in self.predictors
        ]

        return sequential_embeddings, predictions

    def run_batch(self, images, vectors):
        batch_size, L = images.shape[:2]
        base = images[:, : -len(self.predictors), :, :]

        assert base.shape[1] == self.temporal_length

        target_embeddings = []
        targets = images[:, -len(self.predictors) :, :, :]
        for idx, predictor in enumerate(self.predictors):

            target_embedding = self.embed(targets[:, idx : (idx + 1)])
            target_embedding = target_embedding[
                :, :, 0, 0
            ]  # Collapse dimensions of length 1.
            target_embeddings.append(target_embedding)

        sequential_embeddings, predictions = self(base)
        # losses = calculate_cpc_loss(target_embeddings, predictions)
        losses = dict()

        if vectors is not None:
            valid_indices = ~torch.all(torch.abs(vectors) < 1e-4, dim=1)
            if torch.any(valid_indices):
                vectors_hat = self.predict_waggle_direction(
                    sequential_embeddings[valid_indices]
                )
                divergence = 1.0 - self.vector_similarity(
                    vectors[valid_indices], vectors_hat
                )
                losses["vector_mse"] = torch.mean(divergence)

        return losses
