# Test DALI data loader

Run with png format
```bash
FORMAT=png
python gen_synthetic.py --format=png --data-folder=data/${FORMAT}/
DLIO_PROFILER_ENABLE=1 DLIO_PROFILER_INC_METADATA=1 python test_dali.py --data-folder=data/png --format\
=${FORMAT}
```

Run with tfrecord format
```bash
FORMAT=tfrecord
python gen_synthetic.py --format=png --data-folder=data/${FORMAT}/
DLIO_PROFILER_ENABLE=1 DLIO_PROFILER_INC_METADATA=1 python test_dali.py --data-folder=data/png --format\
=${FORMAT}
```

This will generate trace.pfw, which can be visualized by perfetto
