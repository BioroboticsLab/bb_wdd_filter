from locale import normalize
import imgaug.augmenters as iaa
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas
import pickle
import PIL
import scipy.spatial.distance
import skimage.transform
import sklearn.metrics
import torchvision.transforms
import torch
import torch.utils.data
import tqdm.auto
import zipfile
import sklearn.preprocessing

class ImageNormalizer:
    def __init__(self, image_size, scale_factor):
        self.image_size = image_size
        self.scale_factor = scale_factor

        self.crop = iaa.Sequential(
            [
                iaa.Resize(scale_factor),
                iaa.CenterCropToFixedSize(image_size, image_size),
            ]
        )

        self.normalize_to_float = iaa.Sequential(
            [
                # Scale to range -1, +1
                iaa.Multiply(2.0 / 255.0),
                iaa.Add(-1.0),
            ]
        )

    def crop_images(self, images):
        images = self.crop.augment_images(images)
        return images

    def floatify_image(self, img):

        if not np.issubdtype(img.dtype, np.floating):
            assert img.max() > 1
            img = img.astype(np.float32)
        else:
            img = 255.0 * img

        img = self.normalize_to_float.augment_image(img)
        return img

    def floatify_images(self, images):
        images = [self.floatify_image(img) for img in images]
        return images

    def normalize_images(self, images):
        return self.floatify_images(self.crop_images(images))
        

class WDDDataset:
    def __init__(
        self,
        paths,
        temporal_dimension=15,
        n_targets=3,
        target_offset=2,
        images_in_archives=True,
        remap_wdd_dir=None,
        image_size=128,
        silently_skip_invalid=True,
        load_wdd_vectors=False,
        load_wdd_durations=False,
        wdd_angles_for_samples=None,
        default_image_scale=0.5,
    ):

        self.load_wdd_vectors = load_wdd_vectors
        self.load_wdd_durations = load_wdd_durations
        self.silently_skip_invalid = silently_skip_invalid
        self.images_in_archives = images_in_archives
        self.sample_gaps = False
        self.all_meta_files = []
        self.wdd_angles_for_samples = wdd_angles_for_samples

        # Count and index waggle information.
        if isinstance(paths, str):
            if paths.endswith(".pickle"):
                with open(paths, "rb") as f:
                    self.all_meta_files = pickle.load(f)["json_files"]
            else:
                paths = [paths]

        if isinstance(paths, list) and str(paths[0]).endswith(".json"):
            self.all_meta_files += paths
        else:
            if not self.all_meta_files:
                for path in paths:
                    self.all_meta_files += list(pathlib.Path(path).glob("**/*.json"))

        print("Found {} waggle folders.".format(len(self.all_meta_files)))

        if remap_wdd_dir:
            for i, path in enumerate(self.all_meta_files):
                path = str(path).replace("/mnt/thekla/", remap_wdd_dir)
                path = pathlib.Path(path)
                self.all_meta_files[i] = path

        self.temporal_dimension = temporal_dimension
        self.n_targets = n_targets
        self.target_offset = target_offset

        self.default_normalizer = ImageNormalizer(image_size=image_size,
                scale_factor=default_image_scale)
        

    def load_and_normalize_image(self, filename):
        img = WDDDataset.load_image(filename)

        img = self.default_normalizer.crop_images(img)
        img = self.default_normalizer.floatify_image(img)

        return img

    @staticmethod
    def load_image(filename):
        img = PIL.Image.open(filename)
        img = np.asarray(img)
        assert img.dtype is np.dtype(np.uint8)
        return img

    @staticmethod
    def load_images(filenames, parent=""):
        return [WDDDataset.load_image(os.path.join(parent, f)) for f in filenames]

    @staticmethod
    def load_images_from_archive(filenames, archive):
        images = []
        for fn in filenames:
            with archive.open(fn, "r") as f:
                images.append(WDDDataset.load_image(f))
        return images

    @staticmethod
    def load_metadata_for_waggle(
        waggle_metadata_path,
        temporal_dimension,
        load_images=True,
        images_in_archives=False,
        gap_factor=1,
        n_targets=0,
        target_offset=1,
        return_center_images=False,
    ):

        waggle_dir = waggle_metadata_path.parent

        with open(waggle_metadata_path, "r") as f:
            waggle_metadata = json.load(f)

        available_frames_length = len(waggle_metadata["frame_timestamps"])
        try:
            waggle_angle = waggle_metadata["waggle_angle"]
            assert np.abs(waggle_angle) < np.pi * 2.0
            waggle_duration = waggle_metadata["waggle_duration"]
        except:
            waggle_angle = np.nan
            waggle_duration = np.nan

        if temporal_dimension is not None:
            target_sequence_length = n_targets * target_offset
            sequence_length = int(
                gap_factor * temporal_dimension + target_sequence_length
            )

            if not return_center_images:
                sequence_start = np.random.randint(
                    0, available_frames_length - sequence_length
                )
            else:
                sequence_center = available_frames_length // 2
                sequence_start = sequence_center - sequence_length // 2

            assert available_frames_length >= target_sequence_length + sequence_length

        def select_images_from_list(images):

            if temporal_dimension is None:
                if return_center_images:
                    n_available_images = len(images)
                    if n_available_images > 32:
                        images = images[
                            (n_available_images // 4) : -(n_available_images // 4)
                        ]
                return images

            if len(images) != available_frames_length:
                print(
                    "N images: {}, available_frames_length: {}".format(
                        len(images), available_frames_length
                    )
                )

            assert len(images) == available_frames_length
            images = images[sequence_start : (sequence_start + sequence_length)]

            targets_start = sequence_length - target_sequence_length

            if n_targets != 0:
                targets = images[targets_start:][::target_offset]
            else:
                targets = []

            if temporal_dimension == sequence_length - target_sequence_length:
                images = images[:targets_start]
            else:
                if return_center_images:
                    mid = len(images) // 2
                    margin = temporal_dimension // 2
                    images = images[(mid - margin) : (mid + margin + 1)]
                else:
                    images = [
                        images[idx]
                        for idx in sorted(
                            np.random.choice(
                                sequence_length - target_sequence_length,
                                size=temporal_dimension,
                                replace=False,
                            )
                        )
                    ]
            return images + targets

        if images_in_archives:
            zip_file_path = os.path.join(waggle_dir, "images.zip")
            if not os.path.exists(zip_file_path):
                print("{} does not exist.".format(zip_file_path))
                return None, None

            try:
                with zipfile.ZipFile(zip_file_path, "r") as zf:
                    images = list(sorted(zf.namelist()))
                    images = select_images_from_list(images)

                    if load_images:
                        images = WDDDataset.load_images_from_archive(images, zf)
            except zipfile.BadZipFile:
                print("ZipFile corrupt: {}".format(zip_file_path))
                return None, None

        else:
            images = list(
                sorted([f for f in os.listdir(waggle_dir) if f.endswith("png")])
            )
            if len(images) == 0:
                print("No images found in folder {}.".format(waggle_dir))
            assert len(images) > 0

            images = select_images_from_list(images)
            if load_images:
                images = WDDDataset.load_images(images, waggle_dir)

        return images, waggle_angle, waggle_duration

    def __len__(self):
        return len(self.all_meta_files)

    def __getitem__(
        self,
        i,
        aug=None,
        return_just_one=False,
        normalize_to_float=False,
        return_center_images=False,
    ):
        waggle_metadata_path = self.all_meta_files[i]

        images, waggle_angle, waggle_duration = WDDDataset.load_metadata_for_waggle(
            waggle_metadata_path,
            self.temporal_dimension,
            images_in_archives=self.images_in_archives,
            n_targets=self.n_targets,
            target_offset=self.target_offset,
            return_center_images=return_center_images,
        )

        if self.wdd_angles_for_samples is not None:
            waggle_angle = self.wdd_angles_for_samples[i]

        if images is None:
            if self.silently_skip_invalid:
                return self[i + 1]
            else:
                return None, None, None
        if return_just_one:
            images = images[:1]
        # images = WDDDataset.load_images(image_filenames, parent=waggle_metadata_path.parent)
        if aug is not None:
            images, waggle_angle = aug(images, waggle_angle)
        else:
            images = self.default_normalizer.crop_images(images)

        if normalize_to_float:
            images = self.default_normalizer.floatify_images(images)

        images = np.stack(images, axis=0)  # Stack over channels.

        if self.load_wdd_vectors:
            waggle_vector = np.zeros(shape=(2,), dtype=np.float32)
            if np.isfinite(waggle_angle):
                waggle_vector[0] = np.cos(waggle_angle)
                waggle_vector[1] = np.sin(waggle_angle)
        else:
            waggle_vector = None

        if not self.load_wdd_durations:
            waggle_duration = None

        return images, waggle_vector, np.float32(waggle_duration)


class BatchSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        image_size=32,
        inflate_dataset_factor=1,
        image_scale_factor=0.5,
        augmentation_per_image=True,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.total_length = len(dataset)
        self.inflate_dataset_factor = int(inflate_dataset_factor)
        self.image_scale_factor = image_scale_factor
        self.augmentation_per_image = augmentation_per_image

        self.augmenters = None
        self.image_size = image_size

    def init_augmenters(self, current_epoch=1, total_epochs=1):

        p = np.clip(
            0.1 + np.log1p(2 * current_epoch / (max(1, total_epochs - 1))), 0, 1
        )

        # p = 0.0

        # These are applied to each image individually and must not rotate e.g. the images.
        self.quality_augmenters = iaa.Sequential(
            [
                iaa.Sometimes(0.55 * p, iaa.GammaContrast((0.9, 1.1))),
                iaa.Sometimes(0.25 * p, iaa.SaltAndPepper(0.01)),
                iaa.Sometimes(0.5 * p, iaa.AdditiveGaussianNoise(scale=(0, 0.1))),
                iaa.Sometimes(0.25 * p, iaa.GaussianBlur(sigma=(0.0, 0.5))),
                iaa.Sometimes(0.25 * p, iaa.Add(value=(-5, 5))),
            ]
        )
        self.rescale = iaa.Sequential(
            [
                # Scale to range -1, +1
                iaa.Multiply(2.0 / 255.0),
                iaa.Add(-1.0),
            ]
        )

        # These are sampled for each batch and applied to all images.
        self.augmenters = iaa.Sequential(
            [
                iaa.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=0.0,
                    shear=(-5, 5),
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                ),
                iaa.CropToFixedSize(
                    self.image_size * int(1.0 / self.image_scale_factor),
                    self.image_size * int(1.0 / self.image_scale_factor),
                    position="center",
                ),
                iaa.Resize(self.image_scale_factor),
                iaa.Sometimes(
                    0.25 * p,
                    iaa.Sequential(
                        [
                            iaa.Crop(
                                percent=(0.1, 0.25),
                                sample_independently=False,
                                keep_size=False,
                            ),
                            iaa.PadToFixedSize(
                                self.image_size,
                                self.image_size,
                                position="center",
                            ),
                        ]
                    ),
                ),
            ]
        )

        # self.augmenters = iaa.Sequential([iaa.CropToFixedSize(128, 128, position="center")])

    def __len__(self):
        return (self.total_length * self.inflate_dataset_factor) // self.batch_size

    def __getitem__(self, _):

        if self.augmenters is None:
            self.init_augmenters()

        aug = self.augmenters.to_deterministic()

        def augment_fn(images, *args):
            nonlocal aug
            img_aug = self.quality_augmenters
            if not self.augmentation_per_image:
                # Apply the same augmentation to the whole sequence.
                img_aug = img_aug.to_deterministic()
            images = img_aug.augment_images(images)
            images = self.rescale.augment_images(
                [img.astype(np.float32) for img in images]
            )
            images, angles = BatchSampler.augment_sequence(aug, images, *args)
            return images, angles

        samples, angles, durations = [], [], []
        has_labels = False
        labels = []

        for _ in range(self.batch_size):
            idx = np.random.randint(self.total_length)
            sample_data = self.dataset.__getitem__(idx, aug=augment_fn)
            label = None
            if len(sample_data) == 2:
                images, angle, duration = sample_data
            else:
                images, angle, duration, label = sample_data
                has_labels = True

            samples.append(images)
            angles.append(angle)
            durations.append(duration)
            labels.append(label)

        samples = np.stack(samples, axis=0)
        angles = np.stack(angles, axis=0)
        durations = np.stack(durations, axis=0)

        if not has_labels:
            return samples, angles, durations

        labels = np.stack(labels, axis=0)
        return samples, angles, durations, labels

    @classmethod
    def augment_sequence(self, aug, images, angle, rotate=True):

        rotation = np.random.randint(0, 360)

        for idx, img in enumerate(images):
            if rotate:
                img = skimage.transform.rotate(img, rotation)
            images[idx] = aug.augment_image(img)

        return images, angle + rotation / 180.0 * np.pi


class ValidationDatasetEvaluator:
    def __init__(
        self,
        gt_data_path,
        remap_paths_to="/mnt/thekla/",
        images_in_archives=False,
        image_size=128,
        raw_paths=None,
        temporal_dimension=None,
        return_indices=False,
    ):

        if raw_paths is None:
            self.gt_data_df, paths = ValidationDatasetEvaluator.load_ground_truth_data(
                gt_data_path, remap_paths_to=remap_paths_to
            )
        else:
            paths = raw_paths

        self.dataset = WDDDataset(
            paths,
            images_in_archives=images_in_archives,
            temporal_dimension=temporal_dimension,
            image_size=image_size,
            n_targets=0,
            silently_skip_invalid=False,
        )

        self.return_indices = return_indices

    @staticmethod
    def load_ground_truth_data(gt_data_path, remap_paths_to=""):
        if isinstance(gt_data_path, str):
            with open(gt_data_path, "rb") as f:
                wdd_gt_data = pickle.load(f)
                gt_data_df = [(key,) + v for key, v in wdd_gt_data.items()]
        else:
            gt_data_df = gt_data_path

        gt_data_df = pandas.DataFrame(
            gt_data_df, columns=["waggle_id", "label", "gt_angle", "path"]
        )
        paths = list(gt_data_df.path.values)

        if remap_paths_to:

            def rewrite(p):
                p = str(p).replace("/mnt/curta/storage/beesbook/wdd/", remap_paths_to)
                p = pathlib.Path(p)
                return p

            paths = [rewrite(p) for p in paths]

        return gt_data_df, paths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        batch_images, vectors, durations = self.dataset.__getitem__(
            i, normalize_to_float=True, return_center_images=True
        )

        if not self.return_indices:
            return batch_images
        return i, batch_images

    def get_images_and_embeddings(
        self,
        model,
        use_last_state=False,
        show_progress=True,
        get_sample_images=True,
        augment_images=False,
    ):

        augmentations = [None]
        if augment_images:
            augmentations = [
                None,
                iaa.Fliplr(1.0),
                iaa.Flipud(1.0),
                iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
            ]

        model.eval()
        with torch.no_grad():
            embeddings = []
            sample_images = []
            labels = []

            trange = range(len(self.dataset))
            if show_progress:
                trange = tqdm.auto.tqdm(trange)

            for i in trange:
                original_batch_images = self[i]

                for aug in augmentations:
                    # Collapse batch dimension.
                    batch_images = original_batch_images.copy()

                    if aug is not None:
                        batch_images = aug.augment_images(batch_images)

                    # Add batch dimension.
                    batch_images = batch_images[None, :, :, :]

                    if get_sample_images:
                        temp_dimension = batch_images.shape[0]
                        sample_images.append(batch_images[0, temp_dimension // 2])

                    batch_images = torch.from_numpy(batch_images).cuda()

                    _, embedding = model.embed_sequence(
                        batch_images,
                        return_full_state=not use_last_state,
                        check_length=False,
                    )

                    if not use_last_state and model.lstm is not None:
                        embedding = torch.mean(embedding[:, 0], dim=0)

                    embedding = embedding.detach().cpu().numpy().flatten()

                    embeddings.append(embedding)
                    labels.append(self.gt_data_df.label.iloc[i])

        embeddings = np.array(embeddings)

        return sample_images, embeddings, labels

    def plot_embeddings(self, sample_images, embeddings, labels, **kwargs):

        from bb_wdd_filter.visualization import plot_embeddings

        return plot_embeddings(
            embeddings=embeddings,
            indices=np.arange(len(self.dataset)),
            dataset=self.dataset,
            images=sample_images,
            labels=labels,
            **kwargs,
        )

    def calculate_scores(self, embeddings, labels):

        import sklearn.linear_model
        import sklearn.preprocessing
        import sklearn.dummy

        unique_labels = list(sorted(np.unique(labels)))
        label_encoder = lambda l: np.array([unique_labels.index(x) for x in l])
        reg_model = sklearn.linear_model.LogisticRegression()

        X = embeddings
        Y = label_encoder(labels)

        scores = dict()

        from sklearn.metrics import make_scorer
        import sklearn.metrics

        scorers = dict(
            accuracy=make_scorer(sklearn.metrics.accuracy_score),
            f1=make_scorer(sklearn.metrics.f1_score, average="macro"),
            roc_auc_score=make_scorer(
                sklearn.metrics.roc_auc_score, multi_class="ovr", needs_proba=True
            ),
        )

        for label in ("all", "waggle"):
            _Y = Y
            if label != "all":
                target_label = unique_labels.index(label)
                _Y = (Y == target_label).astype(int)

            cv_results = sklearn.model_selection.cross_validate(
                reg_model, X, _Y, scoring=scorers, cv=10
            )

            for metric_name, metric_results in cv_results.items():
                if not metric_name.startswith("test_"):
                    continue
                scores[f"{label}_{metric_name}"] = np.mean(metric_results)

        return scores

    def evaluate(self, model, embed_kwargs={}, plot_kwargs={}):

        images, embeddings, labels = self.get_images_and_embeddings(
            model, **embed_kwargs
        )
        scores = self.calculate_scores(embeddings, labels)
        plot = self.plot_embeddings(images, embeddings, labels, **plot_kwargs)

        return scores, plot


class SupervisedDataset:
    def __init__(
        self,
        gt_paths,
        image_size=32,
        temporal_dimension=40,
        remap_paths_to="/mnt/thekla/",
        images_in_archives=False,
        **kwargs,
    ):

        self.gt_data_df, self.paths = ValidationDatasetEvaluator.load_ground_truth_data(
            gt_paths, remap_paths_to=remap_paths_to
        )
        self.dataset = WDDDataset(
            self.paths,
            images_in_archives=images_in_archives,
            temporal_dimension=temporal_dimension,
            image_size=image_size,
            n_targets=0,
            silently_skip_invalid=False,
            wdd_angles_for_samples=self.gt_data_df.gt_angle.values,
            **kwargs,
        )

        labels = self.gt_data_df.label.copy()
        labels[labels == "trembling"] = "other"
        self.all_labels = ["other", "waggle", "ventilating", "activating"]
        label_mapper = {s: i for i, s in enumerate(self.all_labels)}
        self.Y = np.array([label_mapper[l] for l in labels])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i, **kwargs):
        images, vector, duration = self.dataset.__getitem__(i, **kwargs)
        label = self.Y[i]

        # Add empty channel dimension.
        images = np.expand_dims(images, 0)

        return images, vector, duration, label


class SupervisedValidationDatasetEvaluator:
    def __init__(
        self,
        gt_data_path,
        remap_paths_to="/mnt/thekla/",
        images_in_archives=False,
        image_size=128,
        temporal_dimension=None,
        return_indices=False,
        default_image_scale=0.25,
        class_labels=["other", "waggle", "ventilating", "activating"]
    ):

        self.dataset = SupervisedDataset(
            gt_data_path,
            images_in_archives=images_in_archives,
            image_size=image_size,
            load_wdd_vectors=True,
            load_wdd_durations=True,
            remap_paths_to=remap_paths_to,
            default_image_scale=default_image_scale,
        )

        self.return_indices = return_indices
        self.class_labels = class_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):

        item = self.dataset.__getitem__(
            i, normalize_to_float=True, return_center_images=True
        )

        if self.return_indices:
            return i, item
        return item

    def evaluate(self, model, plot_kwargs=dict()):

        dataloader = torch.utils.data.DataLoader(
            self, num_workers=0, batch_size=16, shuffle=False, drop_last=False
        )

        all_classes_hat = []
        all_vectors_hat = []
        all_durations_hat = []
        all_classes = self.dataset.Y
        all_vectors = []
        all_durations = []

        for (images, vectors, durations, _) in dataloader:
            predictions = model(images.cuda())
            assert predictions.shape[2] == 1
            assert predictions.shape[3] == 1
            assert predictions.shape[4] == 1
            predictions = predictions[:, :, 0, 0, 0]

            n_classes = 4
            classes_hat = predictions[:, :n_classes]
            vectors_hat = predictions[:, n_classes : (n_classes + 2)]
            durations_hat = predictions[:, (n_classes + 2) : (n_classes + 3)]

            vectors_hat = torch.tanh(vectors_hat)
            durations_hat = torch.relu(durations_hat)

            classes_hat = torch.nn.functional.softmax(classes_hat, dim=1)

            classes_hat = classes_hat.detach().cpu().numpy()
            vectors_hat = vectors_hat.detach().cpu().numpy()
            durations_hat = durations_hat.detach().cpu().numpy()

            all_classes_hat.append(classes_hat)
            all_vectors_hat.append(vectors_hat)
            all_durations_hat.append(durations_hat)
            all_vectors.append(vectors)
            all_durations.append(durations)

        all_classes_hat = np.concatenate(all_classes_hat, axis=0)
        all_classes_hat_argmax = np.argmax(all_classes_hat, axis=1)
        all_vectors_hat = np.concatenate(all_vectors_hat, axis=0)
        all_durations_hat = np.concatenate(all_durations_hat, axis=0)
        all_vectors = np.concatenate(all_vectors, axis=0)
        all_durations = np.concatenate(all_durations, axis=0)

        metrics = dict()
        metrics["test_balanced_accuracy"] = sklearn.metrics.balanced_accuracy_score(
            all_classes, all_classes_hat_argmax, adjusted=True
        )
        try:
            metrics["test_roc_auc_score"] = sklearn.metrics.roc_auc_score(
                all_classes, all_classes_hat, multi_class="ovr"
            )
        except ValueError as e:
            metrics["test_roc_auc_score"] = np.nan
            
        metrics["test_matthews"] = sklearn.metrics.matthews_corrcoef(
            all_classes, all_classes_hat_argmax
        )
        metrics["test_f1_weighted"] = sklearn.metrics.f1_score(
            all_classes, all_classes_hat_argmax, average="weighted"
        )

        metrics["test_angle_cosine"] = 1.0 - np.mean(
            [
                scipy.spatial.distance.cosine(a, b)
                for (a, b) in zip(all_vectors, all_vectors_hat)
            ]
        )

        for i in range(1, len(self.class_labels)):
            label = self.class_labels[i]
            Y_hat = all_classes_hat_argmax == i
            Y = all_classes == i

            metrics[f"test_precision_{label}"] = sklearn.metrics.precision_score(Y, Y_hat)
            metrics[f"test_recall_{label}"] = sklearn.metrics.recall_score(Y, Y_hat)

            

        idx = ~pandas.isnull(all_durations)
        all_durations = all_durations[idx]
        all_durations_hat = all_durations_hat[idx]
        metrics["test_duration_mse"] = sklearn.metrics.mean_squared_error(
            all_durations, all_durations_hat
        )

        return metrics


class WDDDatasetWithIndicesAndNormalized:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset.__getitem__(
            i,
            return_just_one=False,
            normalize_to_float=True,
            return_center_images=True,
        )
        return i, item
