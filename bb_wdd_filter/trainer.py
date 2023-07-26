import madgrad
import numpy as np
import pandas
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
        save_every_n_batches=None,
        save_every_n_samples=25000,
        eval_test_set_every_n_samples=None,
        num_workers=16,
        continue_training=True,
        image_size=128,
        test_set_evaluator=None,
        batch_sampler_kwargs=dict(),
        max_lr=0.001,
        batches_to_reach_maximum_augmentation=2000,
        run_batch_fn=None,
    ):
        def init_worker(ID):
            import torch
            import numpy as np

            np.random.seed(torch.initial_seed() % 2**32)

            import imgaug

            imgaug.seed((torch.initial_seed() + 1) % 2**32)

        self.dataset = dataset
        self.batch_sampler = BatchSampler(
            dataset, batch_size, image_size=image_size, **batch_sampler_kwargs
        )
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
        self.max_lr = max_lr
        self.batches_to_reach_maximum_augmentation = (
            batches_to_reach_maximum_augmentation
        )
        self.run_batch_fn = run_batch_fn

        self.use_wandb = use_wandb
        self.wandb_config = wandb_config
        if self.use_wandb is None:
            self.use_wandb = wandb_config is not None and len(wandb_config) > 0

        if self.use_wandb:
            import wandb

            self.id = wandb.util.generate_id()
            self.wandb_initialized = False
        else:
            self.id = None

        if save_path == "warn":
            print("Warning: No model save path given. Model will not be saved.")
            save_path = None

        if save_every_n_batches is None:
            save_every_n_batches = save_every_n_samples // batch_size
        if eval_test_set_every_n_samples is None:
            self.eval_test_set_every_n_batches = save_every_n_batches
        else:
            self.eval_test_set_every_n_batches = (
                eval_test_set_every_n_samples // batch_size
            )

        self.save_path = save_path
        self.save_every_n_batches = save_every_n_batches
        self.test_set_evaluator = test_set_evaluator
        self.total_batches = 0
        self.total_epochs = 0

        self.continue_training = continue_training

        if continue_training:
            self.load_checkpoint()

    def is_using_wandb(self):
        return self.use_wandb
    
    def run_batch(self, images, vectors, durations=None, labels=None):
        current_state = dict()

        self.model.train()
        images = images.cuda(non_blocking=True)
        if vectors is not None and not np.any(pandas.isnull(vectors)):
            vectors = vectors.cuda(non_blocking=True)
        else:
            vectors = None

        if durations is not None and not np.all(pandas.isnull(durations)):
            durations = durations.cuda(non_blocking=True)
        else:
            durations = None

        if labels is not None:
            labels = labels.cuda(non_blocking=True)

        self.optimizer.zero_grad()
        if self.run_batch_fn is not None:
            losses = self.run_batch_fn(self.model, images, vectors, durations, labels)
        else:
            losses = self.model.run_batch(images, vectors, durations, labels)

        total_loss = 0.0
        for loss_name, value in losses.items():
            if isinstance(value, dict):
                current_state = {**current_state, **value}
                continue

            if value.requires_grad:
                total_loss += value

            current_state[loss_name] = float(value.detach().cpu().numpy())

        total_loss.backward()
        self.optimizer.step()

        return current_state

    def check_init_wandb(self):
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

    def check_scale_augmenters(self):
        if self.total_batches % 100 == 0:
            # Scale augmentation.
            self.batch_sampler.init_augmenters(
                current_epoch=self.total_batches,
                total_epochs=self.batches_to_reach_maximum_augmentation,
            )

    def save_at_n_batches(self):
        if self.save_path is not None:
            tqdm.auto.tqdm.write(
                "Saving model state at batch {}..".format(self.total_batches)
            )
            self.save_state()

    def sample_and_save_embedding(self):
        import wandb

        self.model.eval()

        loss_info = dict()

        if self.test_set_evaluator is not None:
            scores, plot = self.test_set_evaluator.evaluate(
                self.model, plot_kwargs=dict(display=False)
            )
            loss_info = {**loss_info, **scores}
            loss_info["embedding"] = wandb.Image(plot)

        else:
            e, idx = sample_embeddings(self.model, self.dataset)
            img = plot_embeddings(
                e, idx, self.dataset, scatterplot=False, display=False
            )
            loss_info["embedding"] = wandb.Image(img)

        self.model.train()

        return loss_info

    def run_epoch(self):
        if self.use_wandb:
            import wandb

        self.check_init_wandb()

        n_batches = len(self.batch_sampler)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, self.max_lr, total_steps=n_batches
        )

        for _, batch in enumerate(tqdm.auto.tqdm(self.dataloader, leave=False)):
            self.check_scale_augmenters()

            loss_info = self.run_batch(*batch)

            self.total_batches += 1

            if (self.total_batches + 1) % (self.save_every_n_batches + 1) == 0:
                self.save_at_n_batches()

            if (self.total_batches + 1) % (self.eval_test_set_every_n_batches + 1) == 0:
                additional_vars = None

                if self.use_wandb:
                    with torch.no_grad():
                        additional_vars = self.sample_and_save_embedding()

                if additional_vars is not None:
                    loss_info = {**loss_info, **additional_vars}

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
        model_state_dict = self.model.state_dict()

        state = dict(
            model=model_state_dict,
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
