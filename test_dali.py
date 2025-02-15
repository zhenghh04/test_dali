from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os
import glob
from tqdm import tqdm
from dftracer.logger import dftracer as PerfTrace, fn_interceptor as Profile
from dali_core import InputPipelineCore
from typing import Literal, List, Optional, Tuple
import argparse
import nvidia.dali.tfrecord as tfrec

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument("--data-folder", type=str, default='./data')
parser.add_argument("--format", type=str, default="png")   
parser.add_argument('--batch-size', default=16, type=int)    
parser.add_argument('--computation-time', default=0.1, type=float)    
parser.add_argument('--device', default='cpu', type=str)      
parser.add_argument('--dali-threads', default=8, type=int)  
args = parser.parse_args()

dlp = Profile("TEST_DALI")
BATCH_SIZE=args.batch_size
# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
data_root_dir = args.data_folder
images_dir = data_root_dir
dlp_logger = PerfTrace.initialize_log(logfile=f"trace-{args.format}.pfw",
                        data_dir=data_root_dir,
                        process_id=0)
import time
def loss_func(pred, y):
    pass


def model(x):
    time.sleep(args.computation_time)


def backward(loss, model):
    pass


class TFRecordDataPipeline(InputPipelineCore):
    def __init__(self,
                 batch_size: int, 
                 data_root_dir: str, 
                 index_root_dir: str, 
                 sample_count: int,
                 device: int,
                 shuffle: Optional[bool] = False, 
                 dont_use_mmap: Optional[bool] = False, 
                 seed: Optional[int] = None,
                 threads: Optional[int] = None):
        self._data_root_dir = data_root_dir
        self._index_root_dir = index_root_dir
        self._shuffle = shuffle
        self._dont_use_mmap = dont_use_mmap          
        self._file_list = glob.glob(f"{self._data_root_dir}/*.tfrecord")   
        self._index_files = []
        for file in self._file_list:
            filename = os.path.basename(file)
            self._index_files.append(f"{self._index_root_dir}/{filename}.idx")    
        super().__init__(batch_size, sample_count, device, 
                         build_now=False, threads=threads)
        if device is None:
            self._hardware = 'cpu' 
        else: 
            self._hardware = 'gpu'
        self._build()

    def get_inputs(self, pipeline, **kwargs):
        dataset = fn.readers.tfrecord(path=self._file_list,
                                      index_path=self._index_files,
                                      features={
                                          'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                          'size': tfrec.FixedLenFeature([1], tfrec.int64, 0)
                                      }, num_shards=1,
                                      prefetch_queue_depth=8,
                                      initial_fill=1024,
                                      random_shuffle=True, seed=123,
                                      stick_to_shard=True, pad_last_batch=True, 
                                      dont_use_mmap=self._dont_use_mmap, device=self._hardware)
        return dataset['image'], 0

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
        if device is None:
            self._hardware = 'cpu' 
        else: 
            self._hardware = 'gpu'
        self._build()

    def get_inputs(self, pipeline, **kwargs):
        images, labels = fn.readers.file(
            file_root=self._data_root_dir, random_shuffle=self._shuffle, 
            name="Reader", dont_use_mmap=self._dont_use_mmap, device=self._hardware, 
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

if args.device.find('gpu')!=-1:
    device_id = 0
else:
    device_id = None
if args.format=='png':
    sample_count = len(glob.glob(f"{data_root_dir}/*/*.png"))
    print(sample_count)
    train_data = ImageDataPipeline(data_root_dir = data_root_dir, batch_size = args.batch_size, 
        device = None, sample_count = sample_count, dont_use_mmap = True, threads=args.dali_threads)
elif args.format=='tfrecord':
    sample_count = len(glob.glob(f"{data_root_dir}/data/*.tfrecord"))
    train_data = TFRecordDataPipeline(data_root_dir = f"{args.data_folder}/data/", index_root_dir = f"{args.data_folder}/index/", batch_size = args.batch_size, 
        device = None, sample_count = sample_count, dont_use_mmap = True, threads=args.dali_threads)

for data in tqdm(dlp.iter(train_data)):
    x = data
    pred = model(x)
dlp_logger.finalize()
