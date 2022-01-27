import madgrad
import numpy as np
import pathlib
import shutil
import torch
import tqdm.auto

from .dataset import BatchSampler
from .visualization import plot_embeddings, sample_embeddings


class Trainer:
    def __init__(
        self,
        dataset,
        model,
        batch_size=32,
        use_wandb=None,
        wandb_config=dict(),
        save_path="warn",
        save_every_n_batches=5000,
        num_workers=16,
        continue_training=True,
    ):
        def init_worker(ID):

            import torch
            import numpy as np

            np.random.seed(torch.initial_seed() % 2 ** 32)

            import imgaug

            imgaug.seed((torch.initial_seed() + 1) % 2 ** 32)

        self.dataset = dataset
        self.batch_sampler = BatchSampler(dataset, batch_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.batch_sampler,
            num_workers=num_workers,
            batch_size=None,
            batch_sampler=None,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=init_worker,
        )
        self.model = model

        self.optimizer = madgrad.MADGRAD(self.model.parameters(), lr=0.001)

        self.use_wandb = use_wandb
        self.wandb_config = wandb_config
        if self.use_wandb is None:
            self.use_wandb = len(wandb_config) > 0

        if self.use_wandb:
            import wandb

            self.id = wandb.util.generate_id()
            self.wandb_initialized = False
        else:
            self.id = None

        if save_path == "warn":
            print("Warning: No model save path given. Model will not be saved.")
            save_path = None

        self.save_path = save_path
        self.save_every_n_batches = save_every_n_batches
        self.total_batches = 0
        self.total_epochs = 0

        self.continue_training = continue_training

        if continue_training:
            self.load_checkpoint()

    def run_batch(self, images, vectors):

        current_state = dict()

        self.model.train()
        images = images.cuda(non_blocking=True)
        if vectors is not None:
            vectors = vectors.cuda(non_blocking=True)
        self.optimizer.zero_grad()
        losses = self.model.run_batch(images, vectors)

        total_loss = None
        for loss_name, value in losses.items():
            if value.requires_grad:
                if total_loss is None:
                    total_loss = value
                else:
                    total_loss += value

            current_state[loss_name] = float(value.detach().cpu().numpy())
        total_loss.backward()
        self.optimizer.step()

        return current_state

    def run_epoch(self):

        if self.use_wandb:
            import wandb

            if not self.wandb_initialized:
                self.wandb_initialized = True
                wandb.init(
                    id=self.id,
                    resume="allow" if self.continue_training else False,
                    **self.wandb_config
                )

                config = wandb.config
                config["optimizer"] = type(self.optimizer).__name__

        n_batches = len(self.batch_sampler)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 0.0001, total_steps=n_batches
        )

        for _, batch in enumerate(tqdm.auto.tqdm(self.dataloader, leave=False)):

            if self.total_batches % 100 == 0:
                # Scale augmentation.
                batches_to_reach_maximum_augmentation = 2000
                self.batch_sampler.init_augmenters(
                    current_epoch=self.total_batches,
                    total_epochs=batches_to_reach_maximum_augmentation,
                )

            loss_info = self.run_batch(*batch)

            self.total_batches += 1

            if (self.total_batches + 1) % (self.save_every_n_batches + 1) == 0:
                if self.save_path is not None:
                    tqdm.auto.tqdm.write(
                        "Saving model state at batch {}..".format(self.total_batches)
                    )
                    self.save_state()

                if self.use_wandb:
                    e, idx = sample_embeddings(self.model, self.dataset)
                    img = plot_embeddings(
                        e, idx, self.dataset, scatterplot=False, display=False
                    )
                    loss_info["embedding"] = wandb.Image(img)

            scheduler.step()

            if self.use_wandb:
                loss_info["learning_rate"] = scheduler._last_lr
                wandb.log(loss_info)

    def run_epochs(self, n):
        for i in range(n):

            self.run_epoch()
            self.total_epochs += 1

            if self.save_path is not None:
                print("Saving model state after epoch {}..".format(i), flush=True)
                self.save_state(copy_suffix="_epoch{:03d}".format(self.total_epochs))

    def save_state(self, copy_suffix=None):
        state = dict(
            model=self.model.state_dict(),
            wandb_id=self.id,
            total_batches=self.total_batches,
            total_epochs=self.total_epochs,
        )
        torch.save(state, self.save_path)

        if copy_suffix:
            ext = pathlib.Path(self.save_path).suffix
            copy_path = self.save_path[: -len(ext)] + str(copy_suffix) + ext
            shutil.copy(self.save_path, copy_path)

    def load_checkpoint(self):
        print("Loading last checkpoint...")
        state = torch.load(self.save_path)

        self.id = state["wandb_id"]
        self.total_batches = state["total_batches"]
        self.total_epochs = state["total_epochs"]
        self.model.load_state_dict(state["model"])
