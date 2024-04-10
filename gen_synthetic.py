#!/usr/bin/env python
# This is to generate fake imagenet dataset
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from subprocess import call
import argparse
import cv2
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument("--data-folder", type=str, default='./data')
parser.add_argument("--num-samples", type=int, default=256) 
parser.add_argument("--num-samples-per-file", type=int, default=1)
parser.add_argument("--num-sub-folders", type=int, default=0)
parser.add_argument("--format", type=str, default="png")                   
args = parser.parse_args()

sz = 224
nimages = args.num_samples
if args.format=='png':
    if (args.num_sub_folders>0):
        num_samples_per_folder = args.num_samples//args.num_sub_folders
    else:
        num_samples_per_folder = args.num_samples
    for i in tqdm(range(nimages)):
        os.makedirs(f"{args.data_folder}/{i//num_samples_per_folder}", exist_ok=True)
        arr = np.random.uniform(size=(3, sz, sz))*255 # It's a r,g,b array
        img = cv2.merge((arr[2], arr[1], arr[0]))  # Use opencv to merge as b,g,r
        cv2.imwrite(f'{args.data_folder}/{i//num_samples_per_folder}/image%04d.png'%i, img)
elif args.format=='tfrecord':
    num_files = args.num_samples // args.num_samples_per_file
    os.makedirs(f"{args.data_folder}/data", exist_ok=True)
    os.makedirs(f"{args.data_folder}/index", exist_ok=True)
    for f in tqdm(range(num_files)):
        tf_out_path_spec = f"{args.data_folder}/data/file-{f}-of-{num_files}.tfrecord"
        id_out_path_spec = f"{args.data_folder}/index/file-{f}-of-{num_files}.tfrecord.idx"
        with tf.io.TFRecordWriter(tf_out_path_spec) as writer:
            for i in range(args.num_samples_per_file):
                record = np.random.randint(255, size=(3, 224, 224), dtype=np.uint8)
                img_bytes = record.tobytes()
                data = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        'size': tf.train.Feature(int64_list=tf.train.Int64List(value=[3*224*224]))
                }
                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)
                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)
                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
                tfrecord2idx_script = "tfrecord2idx"
                if not os.path.isfile(id_out_path_spec):
                    call([tfrecord2idx_script, tf_out_path_spec, id_out_path_spec ])
                np.random.seed()
