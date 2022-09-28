import pydicom
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
# import filters
import frangi
from skimage import filters
from skimage import color


# load the DICOM files
dicom_files = []
# print('glob: {}', glob.glob("/sample1/*", recursive=True))
file_list = glob.glob("sample1/*", recursive=False)
# print(file_list)
for fname in file_list:
    # print("loading: {}".format(fname))
    dicom_files.append(pydicom.dcmread(fname))

print("file count: {}".format(len(dicom_files)))

slices = []
skipcount = 0
for f in dicom_files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))
print("slices", len(slices))

slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
# print("image shape", img_shape)
img3d = np.zeros(img_shape)

for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# gray_img3d = color.rgb2gray(img3d[:, :, :, 3])

frangi_filter = frangi.frangi(img3d)

# a1 = plt.subplot(2, 2, 1)
# plt.imshow(gray_img3d[:, :, img_shape[2]//2])
# a1.set_aspect(ax_aspect)

# a2 = plt.subplot(2, 2, 2)
# plt.imshow(gray_img3d[:, img_shape[1]//2, :])
# a2.set_aspect(sag_aspect)

# a3 = plt.subplot(2, 2, 3)
# plt.imshow(gray_img3d[img_shape[0]//2, :, :].T)
# a3.set_aspect(cor_aspect)

# plt.show()
