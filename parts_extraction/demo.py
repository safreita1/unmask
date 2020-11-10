import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
from VOClabelcolormap import color_map
from anno import ImageAnnotation

# bike images: 2009_000985, 2009_000756, 2009_002438, 2010_000898

if __name__ == "__main__":
    file = '2010_000898'
    fdir = 'examples'
    fname_anno = '/VOC2010/Annotations_Part/2009_000985.mat'
    fname_im = '/VOC2010/JPEGImages/2009_000985.jpg'

    an = ImageAnnotation(
        os.path.join(fdir, fname_im),
        os.path.join(fdir, fname_anno))

    f, ax = plt.subplots(1, 1)

    ax.imshow(an.part_mask, cmap=color_map(N=np.max(an.part_mask) + 1))
    ax.set_title('Part mask')
    ax.axis('off')
    plt.show()
