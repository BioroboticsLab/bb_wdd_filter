import argparse

import pickle
import numpy as np
import os
import torch.nn

import bb_wdd_filter.dataset
import bb_wdd_filter.models_supervised
import bb_wdd_filter.trainer_supervised
import bb_wdd_filter.visualization


def run(
    gt_data_paths,
    checkpoint_path=None,
    continue_training=True,
    cache_images=False,
    epochs=1000,
    remap_wdd_dir=None,
    image_size=32,
    multi_gpu=False,
    image_scale=0.5,
    batch_size="auto",
    max_lr=0.002 * 8,
    wandb_entity=None,
    dataset_scale_factors=None,
    wandb_project="wdd-image-classification",
):
    """
    Arguments:
        gt_data_paths (list(string))
            Paths to the .pickle files containing the ground-truth labels and paths.
        dataset_scale_factors (list(float))
            Scaling factors in case of using multiple datasets with different resolutions.
            Same length as gt_data_paths if given.
        remap_wdd_dir (string, optional)
            Prefix of the path where the image data is saved. The paths in gt_data_path
            will be changed to point to this directory instead.
        checkpoint_path (string, optional)
            Filename to which the model will be saved regularly during training.
            The model will be saved on every epoch AND every X batches.
        continue_training (bool)
            Whether to try to continue training from last checkpoint. Will use the same
            wandb run ID. Auto set to "false" in case no checkpoint is found.
        cache_images (bool)
            Whether to create H5 files that contain the uncompressed images. Needs write access to the dataset location.
        epochs (int)
            Number of epochs to train for.
            As the model is saved after every epoch in 'checkpoint_path' and as the logs are
            streamed live to wandb.ai, it's save to interrupt the training after any epoch.
        image_size (int)
            Width and height of images that are passed to the model.
        image_scale (float)
            Scale factor for the data. E.g. 0.5 will scale the images to half resolution.
            That allows for a wider FoV for the model by sacrificing some resolution.
        max_lr (float)
            The training uses a learning rate scheduler (OneCycleLR) for each epoch
            where max_lr constitutes the peak learning rate.
        wandb_entity (string, optional)
            User name for wandb.ai that the training will log data to.
        wandb_project (string)
            Project name for wandb.ai.

    """

    train_datasets = []
    test_datasets = []

    if dataset_scale_factors is None:
        dataset_scale_factors = [None] * len(gt_data_paths)
    else:
        if len(dataset_scale_factors) != len(gt_data_paths):
            raise ValueError(
                "Number of scaling factors need to be the same as number of datasets."
            )
    dataset_scale_factors = [
        (s if (s is not None) else 1.0) for s in dataset_scale_factors
    ]

    for dataset_index, (gt_data_path, gt_scaling_factor) in enumerate(
        zip(gt_data_paths, dataset_scale_factors)
    ):
        suffix_indicator = ".pickle:"
        dataset_name_index = gt_data_path.find(suffix_indicator)
        dataset_name = "dataset{}".format(dataset_index)

        if dataset_name_index != -1:
            dataset_name = gt_data_path[(dataset_name_index + len(suffix_indicator)) :]
            gt_data_path = gt_data_path[
                : (dataset_name_index + len(suffix_indicator) - 1)
            ]

        with open(gt_data_path, "rb") as f:
            wdd_gt_data = pickle.load(f)
            gt_data_df = [(key,) + v for key, v in wdd_gt_data.items()]

        all_indices = np.arange(len(gt_data_df))
        test_indices = all_indices[::10]
        train_indices = [idx for idx in all_indices if not (idx in test_indices)]

        print("Train set:")
        dataset = bb_wdd_filter.dataset.SupervisedDataset(
            [gt_data_df[idx] for idx in train_indices],
            image_size=image_size,
            load_wdd_vectors=True,
            load_wdd_durations=True,
            remap_paths_to=remap_wdd_dir,
            forced_scale_factor=gt_scaling_factor,
            create_cache_on_startup=cache_images,
            temporal_subsampling_probability=0.2,
        )

        eval_dataset_kwargs = dict(
            name=dataset_name,
            gt_paths=[gt_data_df[idx] for idx in test_indices],
            image_size=image_size,
            remap_paths_to=remap_wdd_dir,
            default_image_scale=gt_scaling_factor * image_scale,
            create_cache_on_startup=cache_images,
        )

        train_datasets.append(dataset)
        test_datasets.append(eval_dataset_kwargs)

    if len(train_datasets) == 1:
        dataset = train_datasets[0]
    else:
        dataset = bb_wdd_filter.dataset.ChainDatasetWithKwargsForwarding(
            *train_datasets
        )

    wandb_config = None
    use_wandb = False
    if wandb_entity:
        # Project name is fixed so far.
        # This provides a logging interface to wandb.ai.
        wandb_config = dict(project=wandb_project, entity=wandb_entity)
        use_wandb = True

    print("Test set:")
    # The evaluator's job is to regularly evaluate the training progress on the test dataset.
    # It will calculate additional statistics that are logged over the wandb connection.
    evaluator = bb_wdd_filter.dataset.SupervisedValidationDatasetEvaluator(
        test_datasets, use_wandb=use_wandb
    )

    model = bb_wdd_filter.models_supervised.WDDClassificationModel(
        image_size=image_size
    )

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    if batch_size == "auto":
        # The batch size here is calculated so that it fits on two RTX 2080 Ti in multi-GPU mode.
        # Note that a smaller batch size might also need a smaller learning rate.
        factor = 1
        if multi_gpu:
            factor = 2
        batch_size = int((64 * 7 * factor) / ((image_size * image_size) / (32 * 32)))
    else:
        batch_size = int(batch_size)

    print(
        "N pars: ",
        str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "batch size: ",
        batch_size,
    )

    trainer = bb_wdd_filter.trainer_supervised.SupervisedTrainer(
        dataset,
        model,
        wandb_config=wandb_config,
        save_path=checkpoint_path,
        batch_size=batch_size,
        num_workers=8,
        continue_training=continue_training,
        image_size=image_size,
        batch_sampler_kwargs=dict(
            image_scale_factor=image_scale,
            inflate_dataset_factor=1000,
            augmentation_per_image=False,
        ),
        test_set_evaluator=evaluator,
        eval_test_set_every_n_samples=2000,
        save_every_n_samples=200000,
        max_lr=max_lr,
        batches_to_reach_maximum_augmentation=1000,
    )

    trainer.run_epochs(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path", type=str, default="./ground_truth_wdd_angles.pickle", nargs="+"
    )
    parser.add_argument("--dataset-scale-factor", type=float, default=[1.0], nargs="+")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./wdd_filtering_supervised_model.pt",
    )
    parser.add_argument("--remap-wdd-dir", type=str, default="")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=0.002 * 8)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="wdd-image-classification")
    parser.add_argument("--cache-images", action="store_true")
    args = parser.parse_args()

    continue_training = args.continue_training
    if continue_training and args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print("Can not continue training, as no file found at checkpoint location.")
            continue_training = False

    run(
        gt_data_paths=args.index_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        continue_training=continue_training,
        cache_images=args.cache_images,
        dataset_scale_factors=args.dataset_scale_factor,
        remap_wdd_dir=args.remap_wdd_dir,
        multi_gpu=args.multi_gpu,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
