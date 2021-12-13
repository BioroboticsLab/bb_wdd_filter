import argparse

import os

import bb_wdd_filter.dataset
import bb_wdd_filter.models
import bb_wdd_filter.trainer
import bb_wdd_filter.visualization


def run(
    index_path,
    checkpoint_path=None,
    continue_training=True,
    epochs=10,
    remap_wdd_dir=None,
):
    dataset = bb_wdd_filter.dataset.WDDDataset(index_path, remap_wdd_dir=remap_wdd_dir)
    model = bb_wdd_filter.models.EmbeddingModel().cuda()

    print(
        "N pars: ", str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    trainer = bb_wdd_filter.trainer.Trainer(
        dataset,
        model,
        wandb_config=dict(project="wdd-image-classification", entity="d_d"),
        save_path=checkpoint_path,
        continue_training=continue_training
    )

    trainer.run_epochs(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path",
        type=str,
        default="/home/mi/dormagen/temp/mnt/curta/storage/david/data/wdd/WDD2021_index.pickle",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/mi/dormagen/Downloads/wdd_filtering_model.pt",
    )
    parser.add_argument("--remap-wdd-dir", type=str, default="")
    parser.add_argument("--continue-training", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    continue_training = args.continue_training
    if continue_training and args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print("Can not continue training, as no file found at checkpoint location.")
            continue_training = False

    run(
        index_path=args.index_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        continue_training=continue_training,
        remap_wdd_dir=args.remap_wdd_dir,
    )
