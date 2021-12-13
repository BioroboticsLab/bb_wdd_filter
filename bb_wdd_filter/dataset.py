import imgaug.augmenters as iaa
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import zipfile


class WDDDataset:
    def __init__(
        self, paths, temporal_dimension=21, images_in_archives=True, remap_wdd_dir=None
    ):

        self.images_in_archives = images_in_archives

        self.all_meta_files = []

        # Count and index waggle information.
        if isinstance(paths, str):
            if paths.endswith(".pickle"):
                with open(paths, "rb") as f:
                    self.all_meta_files = pickle.load(f)["json_files"]
            else:
                paths = [paths]
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

        self.default_crop = iaa.Sequential(iaa.CropToFixedSize(128, 128))

    @staticmethod
    def load_image(filename):
        img = plt.imread(filename).astype(np.float32)
        if img.max() > 1.0 + 1e-4:
            img /= 255.0
        if len(img.shape) > 2:
            img = img[:, :, 0]

        img *= 2.0
        img -= 1.0

        return img

    @staticmethod
    def load_images(filenames, parent=""):
        return [WDDDataset.load_image(os.path.join(parent, f)) for f in filenames]

    @staticmethod
    def load_images_from_archive(filenames, archive):
        images = []
        for fn in filenames:
            with archive.open(fn, "r") as f:
                images.append(plt.imread(f))
        return images

    @staticmethod
    def load_metadata_for_waggle(
        waggle_metadata_path,
        temporal_dimension,
        load_images=True,
        images_in_archives=False,
    ):
        waggle_dir = waggle_metadata_path.parent

        with open(waggle_metadata_path, "r") as f:
            waggle_metadata = json.load(f)

        available_frames_length = len(waggle_metadata["frame_timestamps"])
        try:
            waggle_angle = waggle_metadata["waggle_angle"]
        except:
            waggle_angle = np.nan

        sequence_length = 3 * temporal_dimension
        sequence_start = np.random.randint(0, available_frames_length - sequence_length)

        def select_images_from_list(images):
            assert len(images) == available_frames_length
            images = images[sequence_start : (sequence_start + sequence_length)]
            images = [
                images[idx]
                for idx in sorted(
                    np.random.choice(
                        sequence_length, size=temporal_dimension, replace=False
                    )
                )
            ]
            return images

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

    def __getitem__(self, i, aug=None, return_just_one=False):
        waggle_metadata_path = self.all_meta_files[i]

        images, waggle_angle = WDDDataset.load_metadata_for_waggle(
            waggle_metadata_path,
            self.temporal_dimension,
            images_in_archives=self.images_in_archives,
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

        images = np.stack(images, axis=0)  # Stack over channels.
        waggle_vector = np.zeros(shape=(2,))
        if np.isfinite(waggle_angle):
            waggle_angle / 180 * np.pi
            waggle_vector[0] = np.cos(waggle_angle)
            waggle_vector[1] = np.sin(waggle_angle)

        return images, waggle_vector


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.total_length = len(dataset)

        self.augmenters = iaa.Sequential(
            [
                iaa.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(0, 360),
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                ),
                iaa.CropToFixedSize(128, 128)
                # iaa.Sometimes(0.25, iaa.arithmetic.JpegCompression((0, 30))),
            ]
        )

    def __len__(self):
        return self.total_length // self.batch_size

    def __getitem__(self, _):
        aug = self.augmenters.to_deterministic()

        def augment_fn(*args):
            return BatchSampler.augment_sequence(aug, *args)

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
        rotation = aug[0].rotate.draw_sample(random_state=random_state.copy())

        for idx, img in enumerate(images):
            images[idx] = aug.augment_image(img)
        return images, angle + rotation
