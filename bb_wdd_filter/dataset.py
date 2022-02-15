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
import torchvision.transforms
import torch
import tqdm.auto
import zipfile


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
    ):

        self.load_wdd_vectors = load_wdd_vectors
        self.silently_skip_invalid = silently_skip_invalid
        self.images_in_archives = images_in_archives
        self.sample_gaps = False
        self.all_meta_files = []

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

        self.default_crop = iaa.Sequential(
            [iaa.Resize(0.5), iaa.CenterCropToFixedSize(image_size, image_size)]
        )

        self.normalize_to_float = iaa.Sequential(
            [
                # Scale to range -1, +1
                iaa.Multiply(2.0 / 255.0),
                iaa.Add(-1.0),
            ]
        )

    def load_and_normalize_image(self, filename):
        img = WDDDataset.load_image()

        img = self.default_crop.augment_images(img)
        assert img.max() > 1
        img = img.astype(np.float32)
        img = self.normalize_to_float.augment_image(img)

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
        except:
            waggle_angle = np.nan

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
                # Just start taking X images, starting in the middle.
                sequence_center = available_frames_length // 2
                if sequence_center >= target_sequence_length:
                    sequence_start = sequence_center
                else:  # Or take the center X images.
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
            images = select_images_from_list(images)
            if load_images:
                images = WDDDataset.load_images(images, waggle_dir)

        return images, waggle_angle

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

        images, waggle_angle = WDDDataset.load_metadata_for_waggle(
            waggle_metadata_path,
            self.temporal_dimension,
            images_in_archives=self.images_in_archives,
            n_targets=self.n_targets,
            target_offset=self.target_offset,
            return_center_images=return_center_images,
        )
        if images is None:
            if self.silently_skip_invalid:
                return self[i + 1]
            else:
                return None, None
        if return_just_one:
            images = images[:1]
        # images = WDDDataset.load_images(image_filenames, parent=waggle_metadata_path.parent)
        if aug is not None:
            images, waggle_angle = aug(images, waggle_angle)
        else:
            images = self.default_crop.augment_images(images)

        if normalize_to_float:
            assert images[0].max() > 1
            images = [img.astype(np.float32) for img in images]
            images = self.normalize_to_float.augment_images(images)

        images = np.stack(images, axis=0)  # Stack over channels.

        if self.load_wdd_vectors:
            waggle_vector = np.zeros(shape=(2,), dtype=np.float32)
            if np.isfinite(waggle_angle):
                waggle_vector[0] = np.cos(waggle_angle)
                waggle_vector[1] = np.sin(waggle_angle)
        else:
            waggle_vector = None

        return images, waggle_vector


class BatchSampler:
    def __init__(self, dataset, batch_size, image_size=32):
        self.batch_size = batch_size
        self.dataset = dataset
        self.total_length = len(dataset)

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
                iaa.Sometimes(0.55 * p, iaa.GammaContrast((0.75, 1.25))),
                iaa.Sometimes(0.25 * p, iaa.SaltAndPepper(0.05)),
                iaa.Sometimes(0.5 * p, iaa.AdditiveGaussianNoise(scale=(0, 0.2))),
                iaa.Sometimes(0.25 * p, iaa.GaussianBlur(sigma=(0.0, 1.5))),
                iaa.Add(value=(-15, 15)),
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
                iaa.Sometimes(
                    0.25 * p,
                    iaa.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(0, 360),
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    ),
                ),
                iaa.CropToFixedSize(
                    self.image_size * 2, self.image_size * 2, position="center"
                ),
                iaa.Resize(0.5),
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
        return self.total_length // self.batch_size

    def __getitem__(self, _):

        if self.augmenters is None:
            self.init_augmenters()

        aug = self.augmenters.to_deterministic()

        def augment_fn(images, *args):
            nonlocal aug
            images = self.quality_augmenters.augment_images(images)
            images = self.rescale.augment_images(
                [img.astype(np.float32) for img in images]
            )
            images, angles = BatchSampler.augment_sequence(aug, images, *args)
            return images, angles

        samples, angles = [], []

        for _ in range(self.batch_size):
            idx = np.random.randint(self.total_length)
            images, angle = self.dataset.__getitem__(idx, aug=augment_fn)
            samples.append(images)
            angles.append(angle)

        return np.stack(samples, axis=0), np.stack(angles, axis=0)

    @classmethod
    def augment_sequence(self, aug, images, angle):

        random_state = aug.random_state.duplicate(12)[6]
        if len(aug) > 1:
            rotation = (
                aug[0].then_list[0].rotate.draw_sample(random_state=random_state.copy())
            )
            rotation = rotation / 180.0 * np.pi
            assert np.abs(rotation) <= np.pi * 2.0
        else:
            rotation = 0.0

        for idx, img in enumerate(images):
            images[idx] = aug.augment_image(img)

        return images, angle + rotation


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
            with open(gt_data_path, "rb") as f:
                wdd_gt_data = pickle.load(f)
            self.gt_data_df = [(key,) + v for key, v in wdd_gt_data.items()]
            self.gt_data_df = pandas.DataFrame(
                self.gt_data_df, columns=["waggle_id", "label", "gt_angle", "path"]
            )
            paths = list(self.gt_data_df.path.values)

            if remap_paths_to:

                def rewrite(p):
                    p = str(p).replace(
                        "/mnt/curta/storage/beesbook/wdd/", remap_paths_to
                    )
                    p = pathlib.Path(p)
                    return p

                paths = [rewrite(p) for p in paths]

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        batch_images, vectors = self.dataset.__getitem__(
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

                    if not use_last_state:
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
