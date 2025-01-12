# Copyright (c) 2023, Rhino HealthTech, Inc.
# Original file modified by Rhino Health to adapt it to the Rhino Health Federated Computing Platform.

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import time

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception

from coeff_optimizer import OPTIMIZERS

INVALID_SIGNS = [
    '<', '>', '=',
    '+', '-', '*', '/', '//', '%', '**',
    '(', ')', ':', '~', '|', '^', ',',
    '.', "'", '"', '@', '#', '$',
    '[', ']', '{', '}', '?', '!', ' '
]  # These are signs that will fail the smf.glm function if they appear in the column name

class GLMTrainer(Executor):
    def __init__(
        self,
        delta=1,
        sleep_time=0,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        model_name="best_numpy.npy",
        model_dir="model",
        x_values=None,
        y_values=None,
        glm_type=None,
        offset=None,
        add_intercept=True,
        formula=None,
        cast_to_string_fields=None,
        method=None
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        super().__init__()

        if not (isinstance(delta, float) or isinstance(delta, int)):
            raise TypeError("delta must be an instance of float or int.")

        self._delta = delta
        self._model_name = model_name
        self._model_dir = model_dir
        self._sleep_time = sleep_time
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self.x_values = x_values  # List of x values to use in the model
        self.y_values = y_values  # List of the y values to use in the model
        self.formula = formula  # A formula to use in the model (instead of x_values and y_values)
        self._add_intercept = add_intercept  # Whether to add an intercept column to the data (default True)
        self._cast_to_string_fields = cast_to_string_fields  # Fields to explicitly cast to string (e.g. categorical
                                                             # fields that have numeric values)
        self.glm_type = glm_type  # GLM family type (e.g. Binomial, Gaussian, Poisson, etc.)
        self.family_class = getattr(sm.families, self.glm_type)
        self.method = method
        self.offset = offset
        self.site_info = dict()
        print(f"Initialized Federated Client. {x_values=}, {y_values=}, {formula=}, {glm_type=}, {add_intercept=}, {cast_to_string_fields=}")

        # Load dataset
        datasets_path = '/input/datasets'
        dataset_uid = next(os.walk(datasets_path))[1][0]
        dataset_path = f'{datasets_path}/{dataset_uid}/dataset.csv'
        self.data = pd.read_csv(dataset_path)
        if self._cast_to_string_fields:
            print(f"Casting fields {self._cast_to_string_fields} to string.")
            for field in self._cast_to_string_fields:
                self.data[field] = self.data[field].astype(str)
        self.data_x = None
        self.data_y = None
        if self.x_values and self.y_values:
            self.data_x = self.data[self.x_values]
            self.data_y = self.data[self.y_values]

        # Add Intercept
        if self._add_intercept:
            print("Adding intercept column to data.")
            self.data["Intercept"] = 1
            if self.data_x is not None:
                self.data_x["Intercept"] = 1

    def _validate_input(self):
        """
        Validate the input parameters for the model, including:
        - Ensure that either formula or x_values and y_values are provided.
        - Ensure that the supplied variables (either from formula or x_values and y_values) are present in the dataset.
        - Ensure that the supplied variables do not contain invalid signs that will cause the model to fail.
        - Ensure that the offset is only used with the Poisson distribution family.
        """
        # Validate either formula or explicit columns are supplied
        if not self.formula and not (self.x_values and self.y_values):
            print("Either formula or x_values and y_values must be provided.")
            raise ValueError("Either formula or x_values and y_values must be provided.")

        # Validate formula structure
        if self.formula:
            formula = self.formula.replace(" ", "")
            dependent_var, independent_vars = formula.split("~")
            formula_parts = independent_vars.split("+") + [dependent_var]
        else:
            formula_parts = self.x_values + self.y_values
        missing_parts = [part for part in formula_parts if part not in self.data.columns]
        if missing_parts:
            raise ValueError(
                f"The given {"formula" if formla else "y_values or x values"} contains variables that are missing from the data columns: {missing_parts}")
        if any(sign in part for part in formula_parts for sign in INVALID_SIGNS):
            raise ValueError(f"Column headers with the signs {INVALID_SIGNS} are invalid, please modify the dataset columns and the model's formula.")

        # Validate use of offset
        if self.offset and self.family_class != sm.families.Poisson:
            print("Offset is only supported for Poisson distribution family.")
            raise ValueError("Offset is only supported for Poisson distribution family.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        pass

    def _train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # First we extract DXO from the shareable.
        try:
            incoming_dxo = from_shareable(shareable)
        except Exception as e:
            self.system_panic(
                f"Unable to convert shareable to model definition. Exception {secure_format_exception(e)}", fl_ctx
            )
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Information about workflow is retrieved from the shareable header.
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        # Ensure that data is of type weights. Extract model data.
        if incoming_dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Model DXO should be of kind DataKind.WEIGHTS.", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)
        np_data = incoming_dxo.data

        # Display properties.
        self.log_info(fl_ctx, f"Incoming data kind: {incoming_dxo.data_kind}")
        self.log_info(fl_ctx, f"Model: \n{np_data}")
        self.log_info(fl_ctx, f"Current Round: {current_round}")
        self.log_info(fl_ctx, f"Total Rounds: {total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        try:
            optimizer = OPTIMIZERS[self.method]()
            optimizer.get_local_coeffs(current_round, np_data, self.formula, self.offset, self.family_class, self.log_warning, self.data, self.data_y, self.data_x, self.site_info)

        except KeyError:
            self.log_error(fl_ctx, f"No optimizer found for method: {self.method}.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # We check abort_signal regularly to make sure
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save local numpy model
        try:
            self._save_local_model(fl_ctx, np_data)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in saving local model: {secure_format_exception(e)}.")

        # Checking abort signal again.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Prepare a DXO for our updated model. Create shareable and return
        outgoing_dxo = DXO(data_kind=incoming_dxo.data_kind, data=np_data, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1})
        return outgoing_dxo.to_shareable()

    def _submit_model(self, fl_ctx: FLContext, abort_signal: Signal):
        # Retrieve the local model saved during training.
        np_data = None
        try:
            np_data = self._load_local_model(fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load model: {secure_format_exception(e)}")

        # Checking abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Create DXO and shareable from model data.
        model_shareable = Shareable()
        if np_data:
            outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=np_data)
            model_shareable = outgoing_dxo.to_shareable()
        else:
            # Set return code.
            self.log_error(fl_ctx, "local model not found.")
            model_shareable.set_return_code(ReturnCode.EXECUTION_RESULT_ERROR)

        return model_shareable

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Any long tasks should check abort_signal regularly. Otherwise, abort client
        # will not work.
        count, interval = 0, 0.5
        while count < self._sleep_time:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            time.sleep(interval)
            count += interval

        self.log_info(fl_ctx, f"Task name: {task_name}")
        try:
            if task_name == self._train_task_name:
                return self._train(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
            elif task_name == self._submit_model_task_name:
                return self._submit_model(fl_ctx=fl_ctx, abort_signal=abort_signal)
            else:
                # If unknown task name, set RC accordingly.
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in GLMTrainer execute: {secure_format_exception(e)}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _load_local_model(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_path = os.path.join(run_dir, self._model_dir)
        model_load_path = os.path.join(model_path, self._model_name)
        try:
            model = np.load(model_load_path)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load local model: {secure_format_exception(e)}")
            return None

        return model

    def _save_local_model(self, fl_ctx: FLContext, model: dict):
        # Save local model
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_path = os.path.join(run_dir, self._model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_save_path = os.path.join(model_path, self._model_name)
        np.save(model_save_path, model)
        self.log_info(fl_ctx, f"Saved numpy model to: {model_save_path}")
