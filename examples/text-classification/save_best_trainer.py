import logging
import os
import shutil
import warnings

import numpy as np
import torch
import transformers
from transformers.utils import logging
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend
from transformers.file_utils import (
    is_torch_tpu_available
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)

class SaveBestTrainer(transformers.Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model, f"Module {model.module} should be a reference to self.model"
        else:
            assert model is self.model, f"Model {model} should be a reference to self.model"
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            run_id = trial.number if self.hp_search_backend == HPSearchBackend.OPTUNA else tune.get_trial_id()
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
        else:
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

            self.store_flos()
        self.save_model(output_dir)

        # Save optimizer and scheduler
        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)
        elif self.is_world_process_zero():
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(keep=[self.state.best_model_checkpoint])

    def _rotate_checkpoints(self, keep) -> None:
        logger.info("Only keeping best checkpoint {}".format(keep))
        checkpoints_sorted = self._sorted_checkpoints()
        for checkpoint in checkpoints_sorted:
            if checkpoint not in keep:
                logger.info("Deleting checkpoint [{}] because it is not the best".format(checkpoint))
                shutil.rmtree(checkpoint)
