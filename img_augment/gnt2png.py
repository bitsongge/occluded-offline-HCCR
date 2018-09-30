# !/usr/bin/python
# The dataset address : http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
# you should use python 2.x because python 3.x will appear error.

import struct
import PIL.Image
import os
count = 0
path = '/home/alyssa/PythonProjects/occluded/dataset/gnt2png_train/'


for z in range(1241, 1301):
    ff = '/home/alyssa/PythonProjects/occluded/dataset/HWDB1.1tst_gnt/' + str(z) + '-c.gnt'
    f = open(ff, 'rb')
    # ifend = f.read(1)
    while f.read(1) != "":
        f.seek(-1, 1)
        global count
        count += 1
        length_bytes = struct.unpack('<I', f.read(4))[0]
        tag_code = f.read(2).decode('gbk')
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]

        im = PIL.Image.new('RGB', (width, height))
        img_array = im.load()
        for x in range(0, height):
            for y in range(0, width):
                pixel = struct.unpack('<B', f.read(1))[0]
                img_array[y, x] = (pixel, pixel, pixel)

        im = im.resize((64, 64), PIL.Image.ANTIALIAS)
        filename = str(count) + '.png'
        print(filename)
        if(os.path.exists(path + tag_code)):
            filename = path + tag_code + '/' + filename
            im.save(filename)
        else:
            os.makedirs(path + tag_code)
            filename = path + tag_code + '/' + filename
            im.save(filename)

    f.close()
