from typing import TYPE_CHECKING

import torch.nn as nn
from transformers.utils import is_sagemaker_mp_enabled

from swift.utils import get_logger

from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer

logger = get_logger()


class OptimizerCallback:
    """
    Callback for creating and managing optimizer and learning rate scheduler.

    This callback provides hooks for customizing the creation of optimizers and
    learning rate schedulers during the training process. It delegates to the
    trainer's methods by default but can be subclassed to implement custom
    optimization strategies.

    Args:
        args (TrainingArguments): The training arguments containing hyperparameters
            and configuration settings.
        trainer (Trainer): The trainer instance that will use this callback.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Create both optimizer and learning rate scheduler for training.

        This method initializes the optimizer and scheduler by calling their
        respective creation methods and assigns them to the trainer instance.

        Args:
            num_training_steps (int): The total number of training steps, used
                for scheduler configuration (e.g., warmup steps, decay schedule).

        Returns:
            None: The optimizer and scheduler are set directly on the trainer.
        """
        trainer = self.trainer
        trainer.optimizer = self.create_optimizer()
        trainer.scheduler = self.create_scheduler(num_training_steps, trainer.optimizer)

    # def create_optimizer(self) -> Optimizer:
    #     return self.trainer.create_optimizer()

    def create_optimizer(self) -> Optimizer:
        """
        Fast-path default optimizer creation.

        Compared with the upstream Trainer implementation, this version:
        1. Converts `decay_parameters` to a set for O(1) membership checks.
        2. Traverses `named_parameters()` once instead of twice.

        For very large MoE models with many parameter names, this can
        significantly reduce startup time spent in optimizer grouping.
        """
        trainer = self.trainer
        if trainer.optimizer is not None:
            return trainer.optimizer

        # Keep upstream behavior for SageMaker MP environments.
        if is_sagemaker_mp_enabled():
            return trainer.create_optimizer()

        opt_model = trainer.model
        decay_parameters = set(trainer.get_decay_parameter_names(opt_model))
        decay_params = []
        no_decay_params = []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if name in decay_parameters:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optimizer_grouped_parameters = [{
            'params': decay_params,
            'weight_decay': trainer.args.weight_decay,
        }, {
            'params': no_decay_params,
            'weight_decay': 0.0,
        }]

        try:
            if trainer.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(trainer.args, opt_model)

            # Keep compatibility with special optimizer kwargs conventions.
            if 'params' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('params')
            if 'model' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('model')
            if 'optimizer_dict' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('optimizer_dict')

            trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Keep compatibility with bitsandbytes 8bit embedding override.
            if 'bitsandbytes' in str(optimizer_cls) and optimizer_kwargs.get('optim_bits', None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        manager.register_module_override(module, 'weight', {'optim_bits': 32})
        except Exception:
            # If any corner case is missed, fallback to the upstream path.
            logger.warning('Fast optimizer grouping failed, fallback to Trainer.create_optimizer().')
            return trainer.create_optimizer()

        return trainer.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer) -> LRScheduler:
        return self.trainer.create_scheduler(num_training_steps, optimizer)
