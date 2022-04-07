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
    gt_data_path,
    checkpoint_path=None,
    continue_training=True,
    epochs=1000,
    remap_wdd_dir=None,
    image_size=32,
    images_in_archives=False,
    multi_gpu=False,
    image_scale=0.5,
):

    with open(gt_data_path, "rb") as f:
        wdd_gt_data = pickle.load(f)
        gt_data_df = [(key,) + v for key, v in wdd_gt_data.items()]

    all_indices = np.arange(len(gt_data_df))
    test_indices = all_indices[::10]
    train_indices = [idx for idx in all_indices if not (idx in test_indices)]
    print("Train set:")
    dataset = bb_wdd_filter.dataset.SupervisedDataset(
        [gt_data_df[idx] for idx in train_indices],
        images_in_archives=images_in_archives,
        image_size=image_size,
        load_wdd_vectors=True,
        load_wdd_durations=True,
        remap_paths_to=remap_wdd_dir,
    )
    print("Test set:")
    evaluator = bb_wdd_filter.dataset.SupervisedValidationDatasetEvaluator(
        [gt_data_df[idx] for idx in test_indices],
        images_in_archives=images_in_archives,
        image_size=image_size,
        remap_paths_to=remap_wdd_dir,
        default_image_scale=image_scale,
    )

    model = bb_wdd_filter.models_supervised.WDDClassificationModel(
        image_size=image_size
    )

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    batch_size = int((32 * 4 * 2) / ((image_size * image_size) / (32 * 32)))
    print(
        "N pars: ",
        str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "batch size: ",
        batch_size,
    )
    trainer = bb_wdd_filter.trainer_supervised.SupervisedTrainer(
        dataset,
        model,
        wandb_config=dict(project="wdd-image-classification", entity="d_d"),
        save_path=checkpoint_path,
        batch_size=batch_size,
        num_workers=8,
        continue_training=continue_training,
        image_size=image_size,
        batch_sampler_kwargs=dict(
            image_scale_factor=image_scale,
            inflate_dataset_factor=100,
            augmentation_per_image=False,
        ),
        test_set_evaluator=evaluator,
        eval_test_set_every_n_samples=2000,
        save_every_n_samples=50000,
        max_lr=0.001 * 8,
        batches_to_reach_maximum_augmentation=1000,
    )

    trainer.run_epochs(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path",
        type=str,
        # default="/home/mi/dormagen/temp/mnt/curta/storage/david/data/wdd/WDD2021_index.pickle",
        default="/home/mi/dormagen/temp/mnt/curta/storage/david/data/wdd/ground_truth_wdd_angles.pickle",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/mi/dormagen/Downloads/wdd_filtering_supervised_model.pt",
    )
    parser.add_argument("--remap-wdd-dir", type=str, default="")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--images-in-archives", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    continue_training = args.continue_training
    if continue_training and args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print("Can not continue training, as no file found at checkpoint location.")
            continue_training = False

    run(
        gt_data_path=args.index_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        continue_training=continue_training,
        remap_wdd_dir=args.remap_wdd_dir,
        images_in_archives=args.images_in_archives,
        multi_gpu=args.multi_gpu,
    )
