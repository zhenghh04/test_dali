#!/usr/bin/env python
# This is to generate fake imagenet dataset
import numpy as np
sz = 224
nimages = 8192
from tqdm import tqdm
import cv2
import numpy as np
for i in tqdm(range(nimages)):
    arr = np.random.uniform(size=(3, sz, sz))*255 # It's a r,g,b array
    d[i] = arr.reshape((sz, sz, 3))
    img = cv2.merge((arr[2], arr[1], arr[0]))  # Use opencv to merge as b,g,r
    cv2.imwrite('synthetic/images/image%04d.png'%i, img)
f.close()


