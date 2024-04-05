from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os
import glob
from tqdm import tqdm
from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile
from dali_core import InputPipelineCore
from typing import Literal, List, Optional, Tuple

dlp = Profile("TEST_DALI")
BATCH_SIZE=64
# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
data_root_dir = '/home/huihuo.zheng/gpfs/datascience/data_generator/datasets/ILSVRC/synthetic/images/'
images_dir = data_root_dir
dlp_logger = PerfTrace.initialize_log(logfile="trace.pfw",
                        data_dir=data_root_dir,
                        process_id=0)
sample_count = len(glob.glob(f"{data_root_dir}/*/*.png"))
def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass

class ImageDataPipeline(InputPipelineCore):
    def __init__(self,
                 batch_size: int, 
                 data_root_dir: str, 
                 sample_count: int,
                 device: int,
                 shuffle: Optional[bool] = False, 
                 dont_use_mmap: Optional[bool] = False, 
                 seed: Optional[int] = None,
                 threads: Optional[int] = None):
        self._data_root_dir = data_root_dir
        self._shuffle = shuffle
        self._dont_use_mmap = dont_use_mmap                 
        super().__init__(batch_size, sample_count, device, 
                         build_now=False, threads=threads)
        self._build()

    def get_inputs(self, pipeline, **kwargs):
        images, labels = fn.readers.file(
            file_root=self._data_root_dir, random_shuffle=self._shuffle, 
            name="Reader", dont_use_mmap=self._dont_use_mmap, device='cpu', 
            prefetch_queue_depth=64)
        images = fn.decoders.image_random_crop(
            images, device=self._device, output_type=types.RGB)
        images = fn.resize(images, resize_x=256, resize_y=256)
        images = fn.crop_mirror_normalize(
            images,
            crop_h=224,
            crop_w=224,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip())
        return images, labels

train_data = ImageDataPipeline(data_root_dir = data_root_dir, batch_size = 8, device = None, sample_count = sample_count, dont_use_mmap = True, threads=8)


for data in tqdm(dlp.iter(train_data)):
    x = data
    pred = model(x)
dlp_logger.finalize()