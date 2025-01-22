import datetime
from pathlib import Path

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class PTFileModelPersistorAllCheckpoints(PTFileModelPersistor):
    # Used for creating multiple model params files
    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        super().save_model(ml, fl_ctx)
        dest_dir = Path("/output/model_parameters")
        dest_dir.mkdir(parents=False, exist_ok=True)
        timestr = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.save_model_file(f"{dest_dir}/checkpoint_{timestr}.pt")
        self.logger.info(f"Saved model params file to {dest_dir}/checkpoint_{timestr}.pt")
