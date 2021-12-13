import madgrad
import numpy as np
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
        save_path=None,
        save_every_n_batches=5000,
        num_workers=16,
        continue_training=True,
    ):

        self.dataset = dataset
        self.batch_sampler = BatchSampler(dataset, batch_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.batch_sampler,
            num_workers=num_workers,
            batch_size=None,
            batch_sampler=None,
            pin_memory=True,
            shuffle=True,
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

        self.save_path = save_path
        self.save_every_n_batches = save_every_n_batches

        self.continue_training = continue_training

        if continue_training:
            self.load_checkpoint()

    def run_batch(self, images, vectors):

        current_state = dict()

        self.model.train()
        images = images.cuda(non_blocking=True)
        self.optimizer.zero_grad()
        losses = self.model.run_batch(images)

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

        for batch_idx, batch in enumerate(tqdm.auto.tqdm(self.dataloader, leave=False)):

            loss_info = self.run_batch(*batch)

            if batch_idx % self.save_every_n_batches == 0 and batch_idx > 0:
                if self.save_path is not None:
                    self.save_state()

                if self.use_wandb:
                    e, idx = sample_embeddings(self.model, self.dataset)
                    img = plot_embeddings(
                        e, idx, self.dataset, scatterplot=False, display=False
                    )
                    loss_info["embedding"] = wandb.Image(img)

            if self.use_wandb:
                wandb.log(loss_info)

    def run_epochs(self, n):
        for i in range(n):
            self.run_epoch()

    def save_state(self):
        state = dict(model=self.model.state_dict(), wandb_id=self.id)
        torch.save(state, self.save_path)

    def load_checkpoint(self):
        print("Loading last checkpoint...")
        state = torch.load(self.save_path)
        if "wandb_id" in state:
            self.id = state["wandb_id"]
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
