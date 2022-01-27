from locale import normalize
import imgaug.augmenters as iaa
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import PIL
import torchvision.transforms
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
    ):

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

        self.default_crop = iaa.Sequential(iaa.CenterCropToFixedSize(128, 128))

        self.normalize_to_float = iaa.Sequential(
            [
                # Scale to range -1, +1
                iaa.Multiply(2.0 / 255.0),
                iaa.Add(-1.0),
            ]
        )

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
            sequence_start = np.random.randint(
                0, available_frames_length - sequence_length
            )

            assert available_frames_length >= target_sequence_length + sequence_length

        def select_images_from_list(images):

            if temporal_dimension is None:
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

    def __getitem__(self, i, aug=None, return_just_one=False, normalize_to_float=False):
        waggle_metadata_path = self.all_meta_files[i]

        images, waggle_angle = WDDDataset.load_metadata_for_waggle(
            waggle_metadata_path,
            self.temporal_dimension,
            images_in_archives=self.images_in_archives,
            n_targets=self.n_targets,
            target_offset=self.target_offset,
        )
        if images is None:
            return self[i + 1]
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
        waggle_vector = np.zeros(shape=(2,), dtype=np.float32)
        if np.isfinite(waggle_angle):
            waggle_vector[0] = np.cos(waggle_angle)
            waggle_vector[1] = np.sin(waggle_angle)

        return images, waggle_vector


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.total_length = len(dataset)

        self.augmenters = None

    def init_augmenters(self, current_epoch=1, total_epochs=1):

        p = np.clip(
            0.1 + np.log1p(2 * current_epoch / (max(1, total_epochs - 1))), 0, 1
        )
        p = 0.0

        # These are applied to each image individually and must not rotate e.g. the images.
        self.quality_augmenters = iaa.Sequential(
            [
                iaa.Sometimes(0.75 * p, iaa.GammaContrast((0.75, 1.25))),
                iaa.Sometimes(0.5 * p, iaa.SaltAndPepper(0.05)),
                iaa.Sometimes(0.75 * p, iaa.AdditiveGaussianNoise(scale=(0, 0.2))),
                iaa.Sometimes(0.75 * p, iaa.GaussianBlur(sigma=(0.0, 1.5))),
                iaa.Add(value=(-30, 30)),
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
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(0, 360),
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                ),
                iaa.CropToFixedSize(128, 128, position="center"),
                iaa.Sometimes(
                    0.5 * p,
                    iaa.Sequential(
                        [
                            iaa.Crop(
                                percent=(0.1, 0.25),
                                sample_independently=False,
                                keep_size=False,
                            ),
                            iaa.PadToFixedSize(128, 128, position="center"),
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
            rotation = aug[0].rotate.draw_sample(random_state=random_state.copy())
            rotation = rotation / 180.0 * np.pi
            assert np.abs(rotation) <= np.pi * 2.0
        else:
            rotation = 0.0

        for idx, img in enumerate(images):
            images[idx] = aug.augment_image(img)

        return images, angle + rotation
