from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from tensorboardX import SummaryWriter


# This is based on InTimeModelSelector.
class AggregateValidationMetricTbLogger(FLComponent):
    tb_writer: SummaryWriter = None

    def __init__(
        self, weigh_by_local_iter=False, aggregation_weights=None, validation_metric_name=MetaKey.INITIAL_METRICS
    ):
        """Handler to write global aggregated validation metrics to TensorBoard logs.

        Args:
            weigh_by_local_iter (bool, optional): whether the metrics should be weighted by trainer's iteration number.
            aggregation_weights (dict, optional): a mapping of client name to float for aggregation. Defaults to None.
            validation_metric_name (str, optional): key used to save initial validation metric in the DXO meta properties (defaults to MetaKey.INITIAL_METRICS).
        """
        super().__init__()

        self.weigh_by_local_iter = weigh_by_local_iter
        self.validation_metric_name = validation_metric_name
        self.aggregation_weights = aggregation_weights or {}

        self.logger.info(f"model selection weights control: {aggregation_weights}")
        self._reset_stats()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._startup()
        elif event_type == EventType.END_RUN:
            self._shutdown()
        elif event_type == AppEventType.ROUND_STARTED:
            self._reset_stats()
        elif event_type == AppEventType.BEFORE_CONTRIBUTION_ACCEPT:
            self._before_accept(fl_ctx)
        elif event_type == AppEventType.BEFORE_AGGREGATION:
            self._before_aggregate(fl_ctx)

    def _startup(self):
        self._reset_stats()
        self.tb_writer = SummaryWriter("/tb-logs")

    def _shutdown(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer = None

    def _reset_stats(self):
        self.validation_metric_values_and_weights = []

    def _before_accept(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        shareable: Shareable = peer_ctx.get_prop(FLContextKey.SHAREABLE)
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.COLLECTION):
            self.log_debug(fl_ctx, "cannot handle {}".format(dxo.data_kind))
            return False

        if dxo.data is None:
            self.log_debug(fl_ctx, "no data to filter")
            return False

        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        client_name = peer_ctx.get_identity_name(default="?")

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding shareable from {client_name} for round: {contribution_round}. Current round is: {current_round}",
            )
            return False

        validation_metric = dxo.get_meta_prop(self.validation_metric_name)
        if validation_metric is None:
            self.log_debug(fl_ctx, f"validation metric not existing in {client_name}")
            return False
        else:
            self.log_info(fl_ctx, f"validation metric {validation_metric} from client {client_name}")

        if self.weigh_by_local_iter:
            iteration_weight = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        else:
            iteration_weight = 1.0

        aggregation_weight = self.aggregation_weights.get(client_name, 1.0)
        self.log_debug(fl_ctx, f"aggregation weight: {aggregation_weight}")

        weight = iteration_weight * aggregation_weight
        self.validation_metric_values_and_weights.append((validation_metric, weight))
        return True

    def _before_aggregate(self, fl_ctx: FLContext):
        if not self.validation_metric_values_and_weights:
            self.log_debug(fl_ctx, "nothing accumulated")
            return
        weighted_sum = sum(val * weight for val, weight in self.validation_metric_values_and_weights)
        sum_of_weights = sum(weight for _val, weight in self.validation_metric_values_and_weights)
        self.val_metric = weighted_sum / sum_of_weights
        self.logger.debug(f"weighted validation metric {self.val_metric}")

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.tb_writer.add_scalar("global_model_test_loss_per_round", self.val_metric, current_round)

        self._reset_stats()
