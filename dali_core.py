from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as dali_types

import nvidia.dali.plugin.pytorch as dali_pytorch
import torch

from typing import Literal, List, Optional, Tuple

class InputPipelineCore(object):
    def __init__(self,
                 batch_size: int, 
                 sample_count: int,
                 device: int,
                 build_now: bool = True,
                 threads: int = None):
        self._batch_size = batch_size
        self._sample_count = sample_count
        self._device = device
        self._threads = threads

        if build_now:
            self._build()

    def _build(self):
        self._pipeline = self._build_pipeline()

    def __len__(self):
        if hasattr(self, "_total_sample_count"):
            return self._total_sample_count
        return self._sample_count

    def __iter__(self):
        class InputMultiplierIter(object):
            def __init__(self,
                         pipeline,
                         sample_count: int,
                         batch_size: int):
                self._dali_iterator = dali_pytorch.DALIGenericIterator(
                    pipeline,
                    ["data", "label"],
                    size=sample_count,
                    auto_reset=True)
                self._buffer = None
                self._batch_size = batch_size
                self._count = 0

            def __iter__(self):
                return self
            def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
                if self._buffer == None or self._count >= self._batch_size:
                    self._count = 0
                    self._buffer = next(self._dali_iterator)
                data, label = self._buffer[0]["data"], self._buffer[0]["label"]
                data = data[self._count:self._count+1]
                label = label[self._count:self._count+1]
                self._count += 1
                return data, label

        return InputMultiplierIter(self._pipeline,
                                   self._sample_count,
                                   self._batch_size)

    def _build_pipeline(self, **kwargs):
        pipeline = Pipeline(batch_size=self._batch_size,
                            num_threads=self._threads,
                            device_id=self._device,
                            exec_async=True, 
                            py_num_workers=self._threads, 
                            prefetch_queue_depth={"cpu_size": 64, "gpu_size": 2})
        with pipeline:
            input_data, input_label = self.get_inputs(pipeline=pipeline,
                                                      **kwargs)
            pipeline.set_outputs(input_data, input_label)
        return pipeline

    def get_inputs(self, pipelin, **kwargs):
        pass

    def stage_data(self, executor=None, profile: bool = False):
        def no_wait():
            pass
        return no_wait