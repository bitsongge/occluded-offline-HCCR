from __future__ import print_function, division
from imgaug import augmenters as iaa
from scipy import ndimage
import PIL
import os
from glob import glob

# The path of original images
PATH = glob('/home/alyssa/PythonProjects/occluded/dataset/gnt2png_test/*')
# The path of target images
TARGET_PATH = "/home/alyssa/PythonProjects/occluded/dataset/test_aug/"


def draw_per_augmenter_images():
    for path in PATH:
        num = 0
        tag_code = path.split('/')[7]

        name = glob(path+'/*')
        for img_name in name:
            count = 0
            file_name = img_name.split('/')[8]
            image = ndimage.imread(img_name)

            rows_augmenters = [
                ("Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0]]),
                ("GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [2.0]]),
                ("ElasticTransformation\n(sigma=0.2)", [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=0.2)) for alpha in [3.0]])
            ]

            count = 0
            for (_, augmenters) in rows_augmenters:
                for _, augmenter in augmenters:
                    aug_det = augmenter.to_deterministic()
                    im = PIL.Image.fromarray(aug_det.augment_image(image))
                    im = im.resize((64, 64), PIL.Image.ANTIALIAS)

                    if(os.path.exists(TARGET_PATH + tag_code)):
                        filename = TARGET_PATH + tag_code + '/' + str(count) + "_" + file_name
                        im.save(filename)
                    else:
                        os.makedirs(TARGET_PATH + tag_code)
                        filename = TARGET_PATH + tag_code + '/' + str(count) + "_" + file_name
                        im.save(filename)
                    count += 1

    print(num)


if __name__ == "__main__":
    draw_per_augmenter_images()
