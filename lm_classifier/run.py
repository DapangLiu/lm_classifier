#! /usr/bin/env python3
# coding=utf-8

# Ruibo Liu @Dartmouth College
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 4, 6"

from lm_classifier.data import RLHFDataModule
from lm_classifier.models import TsfBasedClassifier

import torch
from pytorch_lightning import Trainer, seed_everything

seed_everything(42)

root_path = "/home/ruibo/Research/lm_classifier/assets"

dm = RLHFDataModule(model_name_or_path="roberta-large",
                    root_path=root_path,
                    task_name="helpful-base",
                    train_batch_size=16,
                    eval_batch_size=16)
dm.setup("fit")
model = TsfBasedClassifier(
    model_name_or_path="roberta-large",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
)

trainer = Trainer(
    max_epochs=3,
    accelerator="auto",
    strategy="ddp",
    devices=4 if torch.cuda.is_available() else None,  # limiting got iPython runs
)
trainer.fit(model, datamodule=dm)
