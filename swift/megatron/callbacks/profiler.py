# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch.distributed as dist
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger
from .base import MegatronCallback

logger = get_logger()


class NPUProfilerCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.prof = None
        self.rank = 0
        self.enabled = False
        self.stopped = False

    @staticmethod
    def _get_export_type(torch_npu_profiler, export_type):
        mapping = {
            'text': torch_npu_profiler.ExportType.Text,
            'db': torch_npu_profiler.ExportType.Db,
        }
        if isinstance(export_type, str):
            export_type = [export_type]
        return [mapping[item] for item in export_type]

    @staticmethod
    def _get_profiler_level(torch_npu_profiler, level):
        return {
            'level0': torch_npu_profiler.ProfilerLevel.Level0,
            'level1': torch_npu_profiler.ProfilerLevel.Level1,
            'level2': torch_npu_profiler.ProfilerLevel.Level2,
        }[level]

    @staticmethod
    def _get_aic_metrics(torch_npu_profiler, metrics):
        mapping = {
            'none': torch_npu_profiler.AiCMetrics.AiCoreNone,
            'pipe_utilization': torch_npu_profiler.AiCMetrics.PipeUtilization,
            'arithmetic_utilization': torch_npu_profiler.AiCMetrics.ArithmeticUtilization,
            'memory': torch_npu_profiler.AiCMetrics.Memory,
            'memory_l0': torch_npu_profiler.AiCMetrics.MemoryL0,
            'memory_ub': torch_npu_profiler.AiCMetrics.MemoryUB,
            'resource_conflict_ratio': torch_npu_profiler.AiCMetrics.ResourceConflictRatio,
            'l2_cache': torch_npu_profiler.AiCMetrics.L2Cache,
        }
        if hasattr(torch_npu_profiler.AiCMetrics, 'MemoryAccess'):
            mapping['memory_access'] = torch_npu_profiler.AiCMetrics.MemoryAccess
        if metrics not in mapping:
            raise RuntimeError(f'Current torch_npu.profiler does not support aic_metrics={metrics!r}.')
        return mapping[metrics]

    def _get_schedule_kwargs(self):
        args = self.args
        return {
            'wait': args.npu_profile_wait,
            'warmup': args.npu_profile_warmup,
            'active': args.npu_profile_active,
            'repeat': args.npu_profile_repeat,
            'skip_first': args.npu_profile_skip_first,
        }

    def _get_schedule_stop_step(self):
        args = self.args
        if args.npu_profile_repeat == 0:
            return None
        return args.npu_profile_skip_first + (
            args.npu_profile_wait + args.npu_profile_warmup + args.npu_profile_active) * args.npu_profile_repeat

    def _stop(self):
        if self.prof is None or self.stopped:
            return
        self.prof.stop()
        self.stopped = True
        logger.info(f'NPU profiling stopped on rank {self.rank}.')

    def on_train_begin(self):
        args = self.args
        if not args.npu_profile:
            return
        if not is_torch_npu_available():
            return
        if not dist.is_available() or not dist.is_initialized():
            self.rank = 0
        else:
            self.rank = dist.get_rank()
        if self.rank not in args.npu_profile_ranks:
            return

        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError('`torch_npu` is required for NPU profiling.') from exc
        if not hasattr(torch_npu, 'profiler'):
            raise RuntimeError('`torch_npu.profiler` is required for NPU profiling.')

        torch_npu_profiler = torch_npu.profiler
        output_dir = args.npu_profile_output_dir or args.tensorboard_dir
        os.makedirs(output_dir, exist_ok=True)
        experimental_config = torch_npu_profiler._ExperimentalConfig(
            export_type=self._get_export_type(torch_npu_profiler, args.npu_profile_export_type),
            profiler_level=self._get_profiler_level(torch_npu_profiler, args.npu_profile_level),
            aic_metrics=self._get_aic_metrics(torch_npu_profiler, args.npu_profile_aic_metrics),
            l2_cache=False,
            msprof_tx=False,
            data_simplification=False,
            record_op_args=False,
            op_attr=False,
        )
        self.prof = torch_npu_profiler.profile(
            activities=[
                torch_npu_profiler.ProfilerActivity.CPU,
                torch_npu_profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu_profiler.schedule(**self._get_schedule_kwargs()),
            on_trace_ready=torch_npu_profiler.tensorboard_trace_handler(
                output_dir, worker_name=f'rank{self.rank}'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        )
        self.prof.start()
        self.enabled = True
        logger.info(
            f'NPU profiling started on rank {self.rank}: dir={output_dir}, '
            f'schedule={self._get_schedule_kwargs()}, '
            f'level={args.npu_profile_level}, export_type={args.npu_profile_export_type}, '
            f'aic_metrics={args.npu_profile_aic_metrics}.')

    def on_step_end(self):
        if not self.enabled or self.stopped:
            return
        self.prof.step()
        stop_step = self._get_schedule_stop_step()
        if stop_step is not None and self.state.iteration >= stop_step:
            self._stop()

    def on_train_end(self):
        self._stop()
