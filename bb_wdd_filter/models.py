import numpy as np
import torch
import torch.nn
import torch.utils
import torchvision.transforms.functional

from .loss import calculate_cpc_loss


class SubsampleBlock(torch.nn.Module):
    def __init__(
        self, n_channels, n_mid_channels=64, n_out_channels=64, subsample=True
    ):
        super().__init__()

        stride = 1 if not subsample else 2

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
                    stride=stride,
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
    def __init__(self, n_channels=1, temporal_length=15, n_targets=3, image_size=128):
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
            SubsampleBlock(
                n_channels, 32, 96 // 2 // f, subsample=image_size >= 128
            ),  # 64
            SubsampleBlock(
                96 // 2 // f, 128 // f, 128 // f, subsample=image_size >= 64
            ),  # 32
            SubsampleBlock(
                128 // f, 128 // f, 256 // f, subsample=image_size >= 32
            ),  # 16
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

        # Input size: b, embedding_size,  temporal_length
        if False:
            self.lstm = None
            self.sequential_embedding = []
            current_length = self.temporal_length
            current_hidden_size = embedding_size
            while current_length >= 4:
                self.sequential_embedding += [
                    torch.nn.Conv1d(
                        current_hidden_size,
                        current_hidden_size * 2,
                        kernel_size=3,
                        stride=1,
                    ),
                    torch.nn.GroupNorm(8, current_hidden_size * 2),
                    torch.nn.Mish(),
                ]
                current_length -= 2
                current_length //= 2

                out_size = (
                    current_hidden_size * 2
                    if current_length >= 4
                    else hidden_state_size
                )
                self.sequential_embedding += [
                    torch.nn.Conv1d(
                        current_hidden_size * 2,
                        out_size,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.GroupNorm(8, out_size),
                    torch.nn.Mish(),
                ]

                current_hidden_size = out_size

            print(current_length)
            self.sequential_embedding += [torch.nn.AvgPool1d(current_length)]
            self.sequential_embedding = torch.nn.Sequential(*self.sequential_embedding)
        else:
            self.lstm = torch.nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_state_size,
                batch_first=False,
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
        if self.training:
            embedding = torch.utils.checkpoint.checkpoint(self.embedding, images)
        else:
            embedding = self.embedding(images)
        return embedding

    def calculate_image_embeddings_for_image_sequences(self, images):

        temporal_length = self.temporal_length or images.shape[1]

        embeddings = []
        for i in range(temporal_length):
            e = self.embed(images[:, i : (i + 1), :, :])
            embedding_size = e.shape[1]
            assert e.shape[2] == 1 and e.shape[3] == 1
            e = e[:, :, 0, 0]
            embeddings.append(e)

        embeddings = torch.stack(embeddings, dim=0)

        return embeddings

    def embed_sequence(self, images, return_full_state=False, check_length=True):

        assert (
            (not check_length)
            or (self.temporal_length is None)
            or (images.shape[1] == self.temporal_length)
        )

        embeddings = self.calculate_image_embeddings_for_image_sequences(images)

        if self.lstm is not None:
            out, hidden_states = self.lstm(embeddings)

            if not return_full_state:
                out = out[-1]  # Last sequence state.

        else:
            e = torch.transpose(embeddings, 0, 1)
            e = torch.transpose(e, 1, 2)

            out = self.sequential_embedding(e)

            if not return_full_state:
                out = out[:, :, -1]

        return embeddings, out

    def forward(self, images):

        image_embeddings, sequential_embeddings = self.embed_sequence(images)
        predictions = [
            predictor(sequential_embeddings) for predictor in self.predictors
        ]

        return image_embeddings, sequential_embeddings, predictions

    def run_batch(self, images, vectors, durations=None, labels=None):
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

        image_embeddings, sequential_embeddings, predictions = self(base)
        losses = calculate_cpc_loss(target_embeddings, predictions)

        # Add rotation invariance losses.
        angles = np.arange(0, 360 - 45, 15) / 180 * np.pi
        angles = np.random.choice(angles, 2, replace=False)

        n_angles = angles.shape[0]
        rotation_loss = 0.0

        for angle in angles:
            rotated = torchvision.transforms.functional.rotate(images, angle=angle)
            rotated_embeddings = self.calculate_image_embeddings_for_image_sequences(
                rotated
            )

            difference = 1.0 - self.vector_similarity(
                image_embeddings, rotated_embeddings
            )
            # difference = torch.abs(image_embeddings - rotated_embeddings).mean()
            rotation_loss += difference.mean()

        losses["rotation_inv_loss"] = 100 * rotation_loss / n_angles

        if vectors is not None:
            valid_indices = ~torch.all(torch.abs(vectors) < 1e-4, dim=1)
            if torch.any(valid_indices):
                vectors_hat = self.predict_waggle_direction(
                    sequential_embeddings[valid_indices]
                )
                divergence = 1.0 - self.vector_similarity(
                    vectors[valid_indices], vectors_hat
                )
                losses["vector_loss"] = torch.mean(divergence)

        return losses
