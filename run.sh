#!/bin/bash
# generate the file

FORMAT=png
python gen_synthetic.py --format=png --data-folder=data/${FORMAT}/
DLIO_PROFILER_ENABLE=1 DLIO_PROFILER_INC_METADATA=1 python test_dali.py --data-folder=data/png --format=${FORMAT}

FORMAT=tfrecord
python gen_synthetic.py --format=png --data-folder=data/${FORMAT}/
DLIO_PROFILER_ENABLE=1 DLIO_PROFILER_INC_METADATA=1 python test_dali.py --data-folder=data/png --format=${FORMAT}
