# Copyright (c) 2024, Rhino HealthTech, Inc.
# Original file modified by Rhino Health to adapt it to the Rhino Health Federated Computing Platform.

# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
from pathlib import Path

import pandas as pd
import torch
import torchvision
from network import PneumoniaModel
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import (
    make_model_learnable,
    model_learnable_to_dxo,
)
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import (
    PTModelPersistenceFormatManager,
)
from pt_constants import PTConstants
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    Resize,
    ToTensor,
)


class PneumoniaTrainer(Executor):
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        test_set_percentage=20.0,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
    ):
        """
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super().__init__()

        self._lr = lr
        self._epochs = epochs
        self._test_set_percentage = test_set_percentage
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # Training setup
        self.model = PneumoniaModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # Read datasets.
        dataset_dirs = [
            x for x in Path("/input/datasets/").iterdir() if x.resolve().is_dir()
        ]
        dataset_dfs = [
            pd.read_csv(dataset_dir / "dataset.csv") for dataset_dir in dataset_dirs
        ]
        for dataset_dir, df in zip(dataset_dirs, dataset_dfs):
            df["JPG file_abspath"] = dataset_dir / "file_data" / df["JPG file"]

        # Random train/test split.
        combined_df = pd.concat(dataset_dfs)
        train_df, test_df = train_test_split(
            combined_df, test_size=self._test_set_percentage / 100
        )
        train_image_file_paths = train_df["JPG file_abspath"]
        test_image_file_paths = test_df["JPG file_abspath"]

        # Load datasets by creating directories with symlinks to actual images.
        train_dataset_folder = Path("/tmp/train_images_symlinks")
        for image_file_path in train_image_file_paths:
            symlink_path = train_dataset_folder / image_file_path.relative_to(
                Path("/input/datasets/")
            )
            symlink_path.parent.mkdir(parents=True, exist_ok=True)
            symlink_path.symlink_to(image_file_path)
        test_dataset_folder = Path("/tmp/test_images_symlinks")
        for image_file_path in test_image_file_paths:
            symlink_path = test_dataset_folder / image_file_path.relative_to(
                Path("/input/datasets/")
            )
            symlink_path.parent.mkdir(parents=True, exist_ok=True)
            symlink_path.symlink_to(image_file_path)

        # Create data loaders for the training and testing sets.
        transforms = Compose(
            [
                Resize(size=(256, 256)),
                RandomRotation(degrees=(-20, +20)),
                CenterCrop(size=224),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._train_dataset = torchvision.datasets.ImageFolder(
            train_dataset_folder, transform=transforms
        )
        self._test_dataset = torchvision.datasets.ImageFolder(
            test_dataset_folder, transform=transforms
        )
        self._train_loader = DataLoader(self._train_dataset, batch_size=4, shuffle=True)
        self._test_loader = DataLoader(self._test_dataset, batch_size=4, shuffle=True)

        self._n_iterations = len(self._train_loader)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        self.tb_writer = SummaryWriter("/tb-logs")

    def local_train(self, fl_ctx, abort_signal):
        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            self.epoch_global = self.epoch_of_start_time + epoch
            running_loss = 0.0
            images_count = 0
            for i, (batch_images, batch_labels) in enumerate(self._train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(batch_images)
                cost = self.loss(predictions, batch_labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy()
                images_count += batch_images.size()[0]

                if (i + 1) % 3000 == 0:
                    self.log_info(
                        fl_ctx,
                        f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, "
                        f"Loss: {running_loss/images_count}",
                    )

            if len(self._train_dataset) > 0 and (i + 1) % 3000 != 0:
                self.log_info(
                    fl_ctx,
                    f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, "
                    f"Loss: {running_loss/images_count}",
                )

            self.tb_writer.add_scalar(
                "local_train_loss_per_epoch",
                running_loss / images_count,
                self.epoch_global,
            )

        self.tb_writer.flush()

    def local_valid(self, fl_ctx, abort_signal):
        self.model.eval()
        total_loss = 0.0
        total_num_images = 0
        with torch.no_grad():
            for i, (batch_images, batch_labels) in enumerate(self._test_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return None
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                predictions = self.model(batch_images)
                total_loss += (
                    self.loss(predictions, batch_labels).cpu().detach().numpy()
                )
                total_num_images += batch_images.size()[0]
        average_loss = total_loss / total_num_images
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.tb_writer.add_scalar(
            "global_model_test_loss_per_round", average_loss, current_round
        )
        return average_loss

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model params
                try:
                    dxo = from_shareable(shareable)
                except Exception:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_files kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx,
                        f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.",
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor.
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}

                # Set the model params
                self.model.load_state_dict(state_dict=torch_weights)

                # Validation with new global model.
                global_metric = self.local_valid(fl_ctx, abort_signal)

                # Run training.
                self.local_train(fl_ctx, abort_signal)
                self.epoch_of_start_time += self._epochs

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(
                    data_kind=DataKind.WEIGHTS,
                    data=new_weights,
                    meta={
                        MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations,
                        MetaKey.INITIAL_METRICS: global_metric,
                    },
                )
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception:
            self.log_exception(fl_ctx, "Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = (
            fl_ctx.get_engine()
            .get_workspace()
            .get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        )
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = (
            fl_ctx.get_engine()
            .get_workspace()
            .get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        )
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(
            exclude_vars=self._exclude_vars
        )
        return ml
