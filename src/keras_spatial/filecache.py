#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic utilities of stacking (merging) multiple SDGs into a single
numpy array.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def flow_to_numpy(dg, dataframe, width=0, height=0,
        basename='arr', path='.', identifier=None):
    """Write samples arrays of fixed size  as numpy files.

    Uses flow_from_dataframe to extract samples and same them as
    numpy files. File names are generated from and saved in the dataframe.

    Args:
      dg (DataGenerator): any instance with flow_from_dataframe method
      dataframe (GeoDataFrame): sample metadata
      width (int): array width
      height (int): array height
      basename (str): basename of numpy file (default='arr')
      path (Path | str): directory where files will be created (default='.')
      identifier (str): column whose content is appended to basename

    Return:
      (pandss.Series): series with filename sans path

    The returned Series has the same index as the dataframe and named 
    'filenames'. It can be joined to dataframe by the caller using 
    datafrmae.join. Example code:
    >>> filenames = to_files(...)
    >>> filenames.name = <column name>
    >>> dataframe.join(filenames)
    """

    if not isinstance(path, Path):
        path = Path(path)

    subnames = dataframe[identifier] if identifier else dataframe.index
    filenames = ['{}_{}.npy'.format(basename, _id) for _id in subnames]

    for fname, arr in zip(filenames, 
            dg.flow_from_dataframe(dataframe, width, height, batch_size=1)):
        np.save(path / fname, arr)

    return pd.Series(data=filenames, name='filename', index=dataframe.index)


class FileCache:

    def __init__(self, path='.', filenames=None):
        self.path = path
        self.filenames = filenames

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p if isinstance(p, Path) else Path(p)

    @property
    def filenames(self):
        return self._filenames

    @filenames.setter
    def filenames(self, f):
        self._filenames = f.copy() if isinstance(f, pd.Series) \
                else pd.Series(f)

    # TODO make safe for repeated filenames (ie subdirectories or from stem)
    def find_files(self, pattern='*.npy', stem=False):
        """Create filenames based on path matching.

        Args:
          pattern (str): pattern passed to Path glob method
          stem (bool): remove extension from filename (default=False)
        """

        if stem:
            self.filenames = sorted([f.stem for f in self.path.glob(pattern)])
        else:
            self.filenames = sorted([f.name for f in self.path.glob(pattern)])

        return len(self.filenames)

    def flow_from_files(self, filenames=None, batch_size=32):
        """Return generator that reads files into desired batch sizes.

        Args:
          filenames (pandas.Series): filesnames to be read 
          batch_size (int): number of files to be batched together
        """

        if filenames:
            self.filenames = filenames

        for i in range(0, len(self.filenames), batch_size):
            yield np.concatenate([np.load(self.path / f) \
                    for f in self.filenames.iloc[i:i+batch_size]])

    def pathgen(self, filenames=None):
        """Returns a genereator joining filenames to path.

        Args:
          filenames (Series | list): filenames to join
        """

        if filenames:
            self.filenames = filenames

        for f in self.filenames:
            yield self.path / f

    def to_files(self, gen, filenames=None):
        """Save numpy arrays as files.

        Args:
          gen generator(ndarray): generator returning numpy arrays
          filenames (Series | list): filenames to be created
        """

        if filenames:
            self.filenames = filenames

        for f, arr in zip(self.pathgen, gen):
            np.save(f, arr)

    def remove(self, filenames=None, missing_ok=False):
        """Remove files from cache directory.

        Args:
          filenames (Series | list): filenames to be removed

        Notes:
          Unlike other methods, remove does not save the filenames argument
          as an attribute of the FileCache instance. 
        """
        if not filenames:
            filenames = self.filenames

        for f in filenames:
            try:
                self.path.joinpath(f).unlink()
            except FileNotFoundError as e:
                if missing_ok == False:
                    raise


def to_archive(dg, dataframe, width=0, height=0,
        archive='./filecache', basename='arr', identifier=None):

    if not isinstance(archive, Path):
        archive = Path(archive)

    subnames = dataframe[identifier] if identifier else dataframe.index
    filenames = ['{}_{}'.format(basename, _id) for _id in subnames]

    # TODO what format -- numpy.savez? zipfile? tar?

    return pd.Series(data=filenames, name='filename', index=dataframe.index)

