#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from shapely.geometry import Point, Polygon
from geopandas import GeoSeries, GeoDataFrame

import keras_spatial.grid as grid
from keras_spatial.samples import regular_grid, random_grid, point_grid
from keras_spatial.samples import mask_samples



__author__ = "Jeff Terstriep"
__copyright__ = "University of Illinois Board of Trustees"
__license__ = "ncsa"


def test_raster_meta():
    bounds, size, crs = grid.raster_meta('data/small.tif')
    assert len(bounds) == 4
    assert len(size) == 2
    assert crs

def test_regular_grid():
    bounds, _, _ = grid.raster_meta('data/small.tif')

    df = regular_grid(*bounds, 5, 5)
    assert len(df) == 40000
    assert df.total_bounds[0] >= bounds[0]
    assert df.total_bounds[1] >= bounds[1]
    assert df.total_bounds[2] <= bounds[2]
    assert df.total_bounds[3] <= bounds[3]

    df = regular_grid(*bounds, 200, 200)
    assert len(df) == 25
    assert df.total_bounds[0] >= bounds[0]
    assert df.total_bounds[1] >= bounds[1]
    assert df.total_bounds[2] <= bounds[2]
    assert df.total_bounds[3] <= bounds[3]

def test_regular_grid_oversize():
    bounds, _, _ = grid.raster_meta('data/small.tif')

    df = regular_grid(*bounds, 5000, 5000)
    assert len(df) == 1
    assert df.total_bounds[0] == bounds[0]
    assert df.total_bounds[1] == bounds[1]
    assert df.total_bounds[2] >= bounds[2]
    assert df.total_bounds[3] >= bounds[3]


def test_regular_grid_overlap():
    bounds, _, _ = grid.raster_meta('data/small.tif')
    size = (bounds[2]-bounds[0], bounds[3]-bounds[1])
    size = [i/10 for i in size]

    df = regular_grid(*bounds, *size, overlap=.5)
    assert len(df) == 400
    assert df.total_bounds[0] >= bounds[0]
    assert df.total_bounds[1] >= bounds[1]
    assert df.total_bounds[2] <= bounds[2]
    assert df.total_bounds[3] <= bounds[3]

def test_random_grid():
    bounds, _, _ = grid.raster_meta('data/small.tif')
    size = (bounds[2]-bounds[0], bounds[3]-bounds[1])
    size = [i/10 for i in size]

    df = random_grid(*bounds, *size, count=100)
    assert len(df) == 100
    assert df.total_bounds[0] >= bounds[0]
    assert df.total_bounds[1] >= bounds[1]
    assert df.total_bounds[2] <= bounds[2]
    assert df.total_bounds[3] <= bounds[3]

def test_mask_samples():
    bounds, shape, crs = grid.raster_meta('data/small.tif')
    minx, miny, maxx, maxy = bounds
    size = ((maxx - minx)/10, (maxy - miny)/10)
    samples = regular_grid(*bounds, *size, crs=crs)

    geom = Polygon(((minx,miny), (maxx,maxy), (maxx,miny), (minx,miny)))

    mask = GeoDataFrame(index=[0], crs=crs, geometry=[geom])

    masked_strict = mask_samples(samples, mask)
    assert len(samples) > len(masked_strict)

    masked_intersect = mask_samples(samples, mask, strict=False)
    assert len(masked_intersect) > len(masked_strict)

def test_point_grid():
    bounds, _, _ = grid.raster_meta('data/small.tif')
    xmin, ymin, xmax, ymax = bounds
    xsize, ysize = xmax - xmin, ymax - ymin
    xc = xsize * np.random.random(200) + xmin
    yc = ysize * np.random.random(200) + ymin
    pts = GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
    df = GeoDataFrame(geometry=pts)

    dfout = point_grid(df, xsize/50.0, ysize/50.0)
    assert len(df) == 200
    

def test_sample_size():
    bounds, _, _ = grid.raster_meta('data/small.tif')
