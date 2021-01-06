#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A CLI used to produce the sample space.
"""

import argparse
import sys
import logging
import numpy as np
from shapely.geometry import box
import rasterio as rio
import geopandas as gpd

from keras_spatial import __version__
from keras_spatial.samples import regular_grid, random_grid

__author__ = "Jeff Terstriep"
__copyright__ = "University of Illinois Board of Trustees"
__license__ = "ncsa"

_logger = logging.getLogger(__name__)


def raster_meta(fname):
    """Return some metadata from a raster file.
    
    Args:
      fname (str): file path

    Returns:
      :tuple(bounds, size, crs)
    """

    with rio.open(fname) as src:
        return (src.bounds, (src.width, src.height), src.crs)

def get_parser():
    """Configure command line arguments

    Returns:
      :obj:`argparse.ArgumentParser`: 
    """
    parser = argparse.ArgumentParser(
        description="Generate geodataframe containing spatial patches")
    parser.add_argument(
        'output',
        metavar='FILE',
        help='output vector file')
    parser.add_argument(
        'size',
        metavar='SIZE',
        type=float,
        nargs=2,
        help='patch size in projection units')
    parser.add_argument(
        '--size-in-pixels',
        action='store_true',
        default=False,
        help='if size is specified in pixels (raster mode only)')
    parser.add_argument(
        '-f', '--format',
        metavar='FORMAT',
        default='GPKG',
        help='output file format (default=GPKG)')
    parser.add_argument(
        '--random-count',
        metavar='COUNT',
        type=int,
        default=0,
        help='number of randomly placed patches (default=0)')
    parser.add_argument(
        '--overlap',
        metavar='PERCENT',
        type=float,
        default=0.0,
        help='percent of overlap (default=0.0)')
    parser.add_argument(
        '-e', '--extent',
        type=float,
        nargs=4,
        metavar='FLOAT',
        help='spatial extent (minx, miny, maxx, maxy)')
    parser.add_argument(
        '--extent-crs',
        metavar='PROJ',
        default='EPSG:4326',
        help='projection of extents (default=EPSG:4326)')
    parser.add_argument(
        '-r', '--raster',
        metavar='FILE',
        help='raster file used to define extents and projection')
    parser.add_argument(
        '-m', '--mask',
        metavar='FILE',
        help='vector file used to define irregular study area')
    parser.add_argument(
        '-t', '--target-crs',
        metavar='PROJ',
        help='target projection if different from source projection')
    parser.add_argument(
        '-V', '--version',
        action='version',
        version='keras-spatial {ver}'.format(ver=__version__))
    parser.add_argument(
        '-v', '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv', '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    parser = get_parser()
    args = parser.parse_args(args)
    setup_logging(args.loglevel)

    # TODO
    if args.size_in_pixels:
        _logger.error('--size-in-pixels not implemented')
        sys.exit(-1)

    if args.overlap < 0 or args.overlap >= 100:
        _logger.error('overlap must be between 0-100')
        sys.exit(-1)
    else:
        args.overlap /= 100.0

    if args.raster:
        args.extent, _, args.extent_crs = raster_meta(args.raster)

    if not args.extent:
        _logger.error('raster file or extent must be provided')
        sys.exit(-1)

    if args.random_count > 0:
        df = random_grid(*args.extent, *args.size, args.random_count)
    else:
        df = regular_grid(*args.extent, *args.size, args.overlap)

    df.crs = args.extent_crs
    
    if args.target_crs:
        df.to_crs(args.target_crs)

    df.to_file(args.output, driver=args.format)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
