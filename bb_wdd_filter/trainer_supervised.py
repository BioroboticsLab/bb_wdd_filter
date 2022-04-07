import madgrad
import numpy as np
import pandas
import pathlib
import shutil
import torch
import tqdm.auto

from .trainer import Trainer
from .dataset import BatchSampler
from .visualization import plot_embeddings, sample_embeddings
from .models_supervised import SupervisedModelTrainWrapper


class SupervisedTrainer(Trainer):
    def __init__(self, dataset, model, *args, batch_sampler_kwargs=dict(), **kwargs):

        model = SupervisedModelTrainWrapper(model)

        super().__init__(
            dataset,
            model,
            *args,
            batch_sampler_kwargs={
                **dict(inflate_dataset_factor=50),
                **batch_sampler_kwargs,
            },
            **kwargs
        )

    def sample_and_save_embedding(self):

        self.model.eval()

        loss_info = dict()

        if self.test_set_evaluator is not None:
            scores = self.test_set_evaluator.evaluate(
                self.model, plot_kwargs=dict(display=False)
            )
            loss_info = {**loss_info, **scores}

        self.model.train()

        return loss_info
