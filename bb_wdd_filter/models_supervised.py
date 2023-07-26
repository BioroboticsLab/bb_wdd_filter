import numpy as np
import sklearn.metrics
import torch
import torch.nn
import torch.utils
import torchvision.transforms.functional

DEFAULT_CLASS_LABELS = ["other", "waggle", "ventilating", "activating"]


class TensorView(torch.nn.Module):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, t):
        return t.view(*self.shape)


class WDDClassificationModel(torch.nn.Module):
    def __init__(
        self,
        n_outputs=7,
        temporal_dimension=40,
        image_size=32,
        scaledown_factor=4,
        inplace=False,
    ):
        super().__init__()

        center_stride = image_size // 32
        center_padding = 2 if image_size == 32 else 0

        if temporal_dimension == 60:
            center_temporal_stride = 2
            center_temporal_kernel_size = 3
        else:
            assert temporal_dimension == 40
            center_temporal_stride = 1
            center_temporal_kernel_size = 5

        s = scaledown_factor

        self.seq = [
            torch.nn.Conv3d(1, 128 // s, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm3d(128 // s),
            torch.nn.Mish(inplace=inplace),
            # 36/56 x 28 - 56 x 60
            torch.nn.Conv3d(
                128 // s, 64 // s, kernel_size=3, stride=1, padding=0, dilation=2
            ),
            torch.nn.BatchNorm3d(64 // s),
            torch.nn.Mish(inplace=inplace),
            # 32/52 x 24 - 52 x 56
            torch.nn.Conv3d(
                64 // s,
                64 // s,
                kernel_size=(5, 3, 3),
                stride=2,
                padding=(3, 1, 1),
                dilation=(2, 1, 1),
            ),
            torch.nn.BatchNorm3d(64 // s),
            torch.nn.Mish(inplace=inplace),
            # 15/25 x 12 - 25 x 28
            torch.nn.Conv3d(
                64 // s,
                64 // s,
                kernel_size=5,
                stride=(1, center_stride, center_stride),
                padding=(2, center_padding, center_padding),
                dilation=1,
            ),
            torch.nn.BatchNorm3d(64 // s),
            torch.nn.Mish(inplace=inplace),
            # 15/25 x 12 - 25 x 12
            torch.nn.Conv3d(
                64 // s,
                128 // s,
                kernel_size=(center_temporal_kernel_size, 3, 3),
                stride=(center_temporal_stride, 2, 2),
                padding=(0, 1, 1),
                dilation=1,
            ),
            torch.nn.BatchNorm3d(128 // s),
            torch.nn.FeatureAlphaDropout(),
            torch.nn.Mish(inplace=inplace),
            # 12 x 6 - 12 x 6
            torch.nn.Conv3d(
                128 // s,
                128 // s,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=(0, 1, 1),
                dilation=(1, 2, 2),
            ),
            torch.nn.BatchNorm3d(128 // s),
            torch.nn.GLU(dim=1),
            torch.nn.Mish(inplace=inplace),
            # 5 x 4 - 5 x 4
            torch.nn.Conv3d(64 // s, n_outputs, kernel_size=(5, 4, 4)),
        ]

        self.seq = torch.nn.Sequential(*self.seq)

    @staticmethod
    def postprocess_predictions(all_outputs, return_raw=False, as_numpy=False):
        n_classes = 4

        classes_hat = all_outputs[:, :n_classes]
        vectors_hat = all_outputs[:, n_classes : (n_classes + 2)]
        durations_hat = all_outputs[:, (n_classes + 2)]

        confidences = None

        if not return_raw:
            probabilities = torch.nn.functional.softmax(classes_hat, 1)
            classes_hat = torch.argmax(probabilities, 1)
            confidences = probabilities[np.arange(probabilities.shape[0]), classes_hat]
            vectors_hat = torch.tanh(vectors_hat)
            durations_hat = torch.relu(durations_hat)

        if as_numpy:
            classes_hat = classes_hat.detach().cpu().numpy()
            vectors_hat = vectors_hat.detach().cpu().numpy()
            durations_hat = durations_hat.detach().cpu().numpy()
            if confidences is not None:
                confidences = confidences.detach().cpu().numpy()

        return classes_hat, vectors_hat, durations_hat, confidences

    def forward(self, images):
        if self.training:
            images.requires_grad = True
            output = torch.utils.checkpoint.checkpoint_sequential(self.seq, 4, images)
        else:
            output = self.seq(images)

        if self.training:
            shape_correct = (
                output.shape[2] == 1 and output.shape[3] == 1 and output.shape[4] == 1
            )
            if not shape_correct:
                raise ValueError(
                    "Incorrect output shape: {} [input shape was {}]".format(
                        output.shape, images.shape
                    )
                )
            output = output[:, :, 0, 0, 0]

        return output

    def load_state_dict(self, d):
        try:
            return super().load_state_dict(d)
        except Exception as e:
            print("Failed to load. Trying without DataParallel prefix.")
        # Strip off Wrapper & DataParallel prefix.
        d = {key.replace("model.module.", ""): v for key, v in d.items()}
        return super().load_state_dict(d)


# To support DataParallel.
class SupervisedModelTrainWrapper(torch.nn.Module):
    def __init__(self, model, class_labels=DEFAULT_CLASS_LABELS, use_wandb=True):
        super().__init__()

        self.vector_similarity = torch.nn.CosineSimilarity(dim=1)
        self.mse = torch.nn.MSELoss()
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.class_labels = class_labels
        self.use_wandb = use_wandb

        self.model = model

    def set_use_wandb(self, use_wandb=True):
        self.use_wandb = use_wandb

    def forward(self, x):
        return self.model(x)

    def calc_additional_metrics(
        self, predictions, labels, vectors_hat, vectors, durations_hat, durations
    ):
        results = dict()

        predictions = torch.nn.functional.softmax(predictions, dim=1)

        predictions = predictions.detach().cpu().numpy()
        predicted_labels = np.argmax(predictions, axis=1)

        labels = labels.detach().cpu().numpy()

        if self.use_wandb:
            import wandb

            results["train_conf"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predicted_labels,
                class_names=self.class_labels,
            )

        try:
            results["train_roc_auc"] = sklearn.metrics.roc_auc_score(
                labels,
                predictions,
                multi_class="ovr",
                labels=np.arange(len(self.class_labels)),
            )
        except:
            pass

        results["train_matthews"] = sklearn.metrics.matthews_corrcoef(
            labels, predicted_labels
        )

        results["train_balanced_accuracy"] = sklearn.metrics.balanced_accuracy_score(
            labels, predicted_labels, adjusted=True
        )

        results["train_f1_weighted"] = sklearn.metrics.f1_score(
            labels, predicted_labels, average="weighted"
        )

        if vectors_hat is not None:
            divergence = self.vector_similarity(vectors_hat, vectors)
            results["vector_cossim"] = torch.mean(divergence)

            vectors_hat = torch.tanh(vectors_hat)
            divergence = self.mse(vectors_hat, vectors)
            results["vector_mse"] = torch.mean(divergence)

        return results

    def run_batch(self, images, vectors, durations, labels):
        # print(images.dtype, vectors.dtype, durations.dtype)
        batch_size, temp_dimension = images.shape[:2]
        model = self.model

        all_outputs = model(images)
        (
            classes_hat,
            vectors_hat,
            durations_hat,
            _,
        ) = WDDClassificationModel.postprocess_predictions(all_outputs, return_raw=True)

        losses = dict()

        losses["classification_loss"] = self.classification_loss(classes_hat, labels)
        # losses["classification_loss"] = self.mse(classes_hat, labels)

        if vectors is not None:
            other_target = 0
            valid_indices = labels != other_target

            if torch.any(valid_indices):
                vectors_hat = vectors_hat[valid_indices]
                vectors = vectors[valid_indices]

                vectors_hat = torch.tanh(vectors_hat)
                # divergence = 1.0 - self.vector_similarity(vectors, vectors_hat)
                divergence = self.mse(vectors, vectors_hat)
                losses["vector_loss"] = 1.5 * torch.mean(divergence)
            else:
                vectors_hat = None
                vectors = None

        if durations is not None:
            waggle_target = 1
            valid_indices = (labels == waggle_target) & (~torch.isnan(durations))

            if torch.any(valid_indices):
                durations_hat = durations_hat[valid_indices]
                durations = durations[valid_indices]

                durations_hat = torch.relu(durations_hat)
                divergence = self.mse(durations, durations_hat)
                losses["duration_loss"] = 1.0 * torch.mean(divergence)
            else:
                durations_hat = None
                durations = None

        with torch.no_grad():
            losses["additional"] = self.calc_additional_metrics(
                classes_hat, labels, vectors_hat, vectors, durations_hat, durations
            )

        return losses
