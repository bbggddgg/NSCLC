#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 17:02
# @Author  : YangChenghan
# @File    : transImageFormat.py
# @Description : 这个函数是用来balabalabala自己写

import processing
from pma_python import *
import numpy as np
from osgeo import gdal
import math
import tqdm

_pmaCoreUrl = "http://localhost:54001/"

def mds2tiff(slidePath, target_quality=100, downscale_factor=1):
    """
    :param slidePath: what slide do you want to convert?
    :type slidePath:
    :param target_quality: set the target TIFF quality 0-100
    :type target_quality:
    :param downscale_factor: set the target scale factor to download. One of [1, 2, 4, 8, 16, 32, 64, 128]
    :type downscale_factor:
    :return:
    :rtype:
    """
    # Get the slide information and information about each zoomlevel available
    print("Fetching image info for {0}".format(slidePath))
    slideInfo = core.get_slide_info(slidePath)
    print(slideInfo)
    zoomLevelsInfo = core.get_zoomlevels_dict(slidePath)
    maxLevel = max(zoomLevelsInfo)
    tileSize = slideInfo["TileSize"]
    print("Horizontal Tiles | Vertical Tiles | Total Tiles")
    for level in zoomLevelsInfo:
        tilesX, tilesY, totalTiles = zoomLevelsInfo[level]
        print("{:>16} |{:>15} |{:>12}".format(tilesX, tilesY, totalTiles))

    filename = slidePath.rpartition("/")[-1]
    xresolution = 10000 / slideInfo["MicrometresPerPixelX"]
    yresolution = 10000 / slideInfo["MicrometresPerPixelY"]

    # Create new TIFF file using the GDAL TIFF driver
    # The width and height of the final tiff is based on number of tiles horizontally and vertically.

    # Validate the parameters
    if target_quality is None or target_quality < 0 or target_quality > 90:
        target_quality = 80
    if downscale_factor not in [1, 2, 4, 8, 16, 32, 64, 128]:
        downscale_factor = 1

    maxLevel = max(zoomLevelsInfo)
    powerof2 = int(math.log2(downscale_factor))

    level = maxLevel - powerof2
    level = min(max(level, 0), maxLevel)
    tilesX, tilesY, totalTiles = zoomLevelsInfo[level]

    # We set the region of the image we want to read to set the final tif size accordingly
    tileRegionX = (0, tilesX)
    tileRegionY = (0, tilesY)

    tileSize = 512
    tiff_drv = gdal.GetDriverByName("GTiff")
    # Set the final size
    ds = tiff_drv.Create(
        filename.split('.')[0] + '.tif',
        int((tileRegionX[1] - tileRegionX[0]) * 512),
        int((tileRegionY[1] - tileRegionY[0]) * 512),
        3,
        options=['BIGTIFF=YES',
                 'COMPRESS=JPEG', 'TILED=YES', 'BLOCKXSIZE=' + str(tileSize), 'BLOCKYSIZE=' + str(tileSize),
                 'JPEG_QUALITY=90', 'PHOTOMETRIC=RGB'
                 ])
    descr = "ImageJ=\nhyperstack=true\nimages=1\nchannels=1\nslices=1\nframes=1"
    ds.SetMetadata({'TIFFTAG_RESOLUTIONUNIT': '3', 'TIFFTAG_XRESOLUTION': str(int(xresolution / downscale_factor)),
                    'TIFFTAG_YRESOLUTION': str(int(yresolution / downscale_factor)), 'TIFFTAG_IMAGEDESCRIPTION': descr})

    print("Maximum level = ", maxLevel, ", level = ", level, ", power of 2 = ", powerof2)
    filename.split('.')[0] + '.tif'

    # We read each tile of the final zoomlevel (1:1 resolution) from the server and write it to the resulting TIFF file
    # Then we create the pyramid of the file using BuildOverviews function of GDAL
    tilesX, tilesY, totalTiles = zoomLevelsInfo[level]
    print("Requesting level {}".format(level))

    pbar = tqdm.tqdm(total=int((tileRegionX[1] - tileRegionX[0]) * (tileRegionY[1] - tileRegionY[0])))
    for x in range(tileRegionX[0], tileRegionX[1]):
        for y in range(tileRegionY[0], tileRegionY[1], 1):  # range of y-axis in which we are interested for this slide
            pbar.update()
            tile = core.get_tile(slidePath, x, y, level, quality=target_quality)
            arr = np.array(tile, np.uint8)

            # calculate startx starty pixel coordinates based on tile indexes (x,y)
            # for the final tif we want the first tile, i.e. (tileRegionX[0], tileRegionY[0]) ,to be at (0,0) so we need to transform the coordinates
            sx = (x - tileRegionX[0]) * tileSize
            sy = (y - tileRegionY[0]) * tileSize

            ds.GetRasterBand(1).WriteArray(arr[..., 0], sx, sy)
            ds.GetRasterBand(2).WriteArray(arr[..., 1], sx, sy)
            ds.GetRasterBand(3).WriteArray(arr[..., 2], sx, sy)

    pbar.close()
    print("Please wait while building the pyramid")
    ds.BuildOverviews('average', [pow(2, l) for l in range(1, level)])
    ds = None
    print("Done")

def files2tiff(root):
    """

    Parameters
    ----------
    root : 数据存放根路径

    Returns
    -------

    """
    for slide_name in os.listdir(root):
        slide_path = os.path.join(root, slide_name)
        files_name = os.listdir(slide_path)
        mds_file = [x for x in files_name if x.endswith('.mds')]
        tif_file = [x for x in files_name if x.endswith('.tif')]

        if len(tif_file) == 0:
            slide = os.path.join(slide_path, mds_file[0])
            # processing.mds2tiff(slide)
            try:
                print("start: {}".format(slide))
                mds2tiff(slide)
                print("finish: {}".format(slide))
            except Exception as e:
                print("error: {}".format(slide))

if __name__ == '__main__':

    slide = r"/path/to/dataset/sample_case/1.mds"
    processing.mds2tiff(slide)
