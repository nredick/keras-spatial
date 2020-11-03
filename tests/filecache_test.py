#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

from keras_spatial.datagen import SpatialDataGenerator
import keras_spatial.filecache as fc


__author__ = "Jeff Terstriep"
__copyright__ = "Jeff Terstriep"
__license__ = "ncsa"


def generate(path, count=5, size=(64,64), indexes=1):

    sdg = SpatialDataGenerator('data/small.tif', indexes=indexes)
    sdg.source = 'data/small.tif'
    df = sdg.random_grid(*size, count=count, units='pixels')

    return fc.flow_to_numpy(sdg, df, *size, path=path)


def test_flow_to_numpy():

    size = (64,64)
    sdg = SpatialDataGenerator('data/small.tif', indexes=1)
    sdg.source = 'data/small.tif'
    df = sdg.random_grid(*size, count=5, units='pixels')

    with TemporaryDirectory() as td:
        tmpdir = Path(td)
        filenames = fc.flow_to_numpy(sdg, df, *size, path=tmpdir)
        assert isinstance(filenames, pd.Series)

        for fname in filenames:
            filepath = tmpdir / fname
            assert filepath.exists()
            arr = np.load(filepath)
            assert len(arr.shape) == 4
            assert arr.shape[0] == 1
            assert arr.shape[1] == 1
            assert arr.shape[2] == size[1] and arr.shape[3] == size[0]


def test_filecache():

    with TemporaryDirectory() as tmpdir:
        generate(tmpdir, count=5)

        cache = fc.FileCache(path=tmpdir)
        assert isinstance(cache.path, Path)

        cache.find_files()
        assert isinstance(cache.filenames, pd.Series)
        assert len(cache.filenames) == 5

        for arr in cache.flow_from_files(batch_size=1):
            assert len(arr.shape) == 4
            assert arr.shape[0] == 1
            assert arr.shape[1] == 1

        cache.remove()
        assert len(list(cache.path.glob('*'))) == 0
