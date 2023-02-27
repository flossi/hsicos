#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:34:28 2022

@author: hermanns
"""

__version__ = '0.1'
__author__ = 'Floris Hermanns'
__license__ = 'BSD 3-Clause'
__status__ = 'Prototype'

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import fiona.errors as ferr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from sklearn import decomposition

import pyeto
import pyproj as proj
from shapely.geometry import Polygon, Point
import rasterio as rio
import rasterio.plot as riop
import rasterio.mask as riom
from rasterio.errors import RasterioIOError
import h5py
import spectral.io.envi as envi
from shapely import geometry
from shapely.ops import transform as stransform
from functools import reduce
from fmch import ffp
from fmch.HSI2RGB import HSI2RGB
from pymf.sivm import SIVM

from pathlib import Path
import datetime as dt
import cdsapi
from hda import Client

import logging
import sys
from zipfile import ZipFile

logger = logging.getLogger('hsicos')
logger.handlers = []
logger.propagate = False
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
#if not logger.hasHandlers():
f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler('/home/hermanns/Work/fluxes/hsicos.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(f)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(f)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
logger.info('Running HSICOS module')

def _build_icos_meta():
    '''
    Function to build a geodataframe containing metadata about a number of ICOS
    sites in different European countries. Includes infos about ecosystem type,
    coordinates, start of records, elevation, etc.
    '''
    flx = pd.DataFrame(
        {'name': ['NL-Loo', 'IT-Tor', 'IT-SR2', 'IT-Lsn', 'IT-Cp2', 'IT-BCi', 'FR-Hes', #7
                  'FR-Fon', 'FR-EM2', 'FR-Bil', 'ES-LM2', 'ES-LM1', 'ES-Abr', 'DE-Tha', #14
                  'DE-RuW', 'DE-RuS', 'DE-RuR', 'DE-Obe', 'DE-Kli', 'DE-Hzd', 'DE-Hte', #21
                  'DE-GsB', 'DE-Hai', 'DE-Hod', 'DE-HoH', 'DE-Gri', 'DE-Geb', 'DE-Akm', #28
                  'CZ-wet', 'CZ-Stn', 'CZ-RAJ', 'CZ-Lnz', 'CZ-BK1', 'CH-Oe2', 'CH-Lae', #35
                  'CH-Fru', 'CH-Dav', 'CH-Cha', 'CH-Aws', 'BE-Vie', 'BE-Maa', 'BE-Lon', #42
                  'BE-Bra', 'FI-Sii', 'FI-Hyy', 'SE-Deg', 'SE-Htm', 'SE-Nor', 'SE-Svb', #49
                  'FR-LGt', 'FR-Mej', 'DK-Vng', 'FR-Tou', 'DK-Skj', 'DK-Gds', 'FR-Lam'], #56
         'ecosystem': ['ENF', 'GRA', 'ENF', 'OSH', 'EBF', 'CRO', 'DBF', 'DBF', 'CRO', 'ENF', #10
                       'SAV', 'SAV', 'SAV', 'ENF', 'ENF', 'CRO', 'GRA', 'ENF', 'CRO', #19
                       'DBF', 'WET', 'GRA', 'DBF', 'CRO', 'DBF', 'GRA', 'CRO', 'WET', #28
                       'WET', 'DBF', 'ENF', 'MF', 'ENF', 'CRO', 'MF', 'GRA', 'ENF', #37
                       'GRA', 'GRA', 'MF', 'CSH', 'CRO', 'MF', 'WET', 'ENF', 'WET', #46
                       'ENF', 'ENF', 'ENF', 'WET', 'GRA', 'CRO', 'GRA', 'WET', 'ENF', #55
                       'CRO'],
         'lat': [52.16658, 45.84444, 43.73202, 45.740482, 41.7043, 40.5237, 48.6741,
                 48.4764, 49.8721, 44.4937, 39.9346, 39.9427, 38.7018, 50.9626, #14
                 50.50493, 50.86591, 50.6219, 50.7867, 50.89306, 50.96381, 54.2103, #21
                 52.02965, 51.07941, 52.00130, 52.08656, 50.95004, 51.0997, 53.8662, #28
                 49.0247, 49.03598, 49.44372, 48.68155, 49.5021, 47.2864, 47.4783, #35
                 47.1158, 46.8153, 47.2102, 46.5832, 50.3049, 50.9799, 50.5516, 51.3076, #43
                 61.8327, 61.8474, 64.1820, 56.0976, 60.0865, 64.2561, 47.3229, 48.1184, #51
                 56.0375, 43.5729, 55.9127, 56.0737, 43.4964],
         'lon': [5.74356, 7.57806, 10.29091, 12.7503, 12.3573, 14.9574, 7.06465,
                 2.7801, 3.02065, -0.95609, -5.77588, -5.77868, -6.78588, 13.5651, #14
                 6.33096, 6.44715, 6.3041, 13.7213, 13.52238, 13.48978, 12.1761, #21
                 11.1048, 10.4521, 11.1777, 11.22235, 13.5126, 10.9146, 13.6834, #28
                 14.7704, 17.9699, 16.69651, 16.94633, 18.5369, 7.7337, 8.3644, #35
                 8.5378, 9.85591, 8.4104, 9.79042, 5.9981, 5.6319, 4.7462, 4.5198, #43
                 24.1929, 24.2948, 19.5565, 13.4190, 17.4795, 19.7745, 2.2841, -1.79635, #51
                 9.1607, 1.3747, 8.4048, 9.3341, 1.2379],
         'start_year': [1996, 2008, 2013, 2016, 2012, 2004, 2014, 2019, 2017, 2014, 2014, #11
                        2014, 2015, 1996, 2010, 2011, 2011, 2008, 2004, 2010, 2009, #21
                        2012, 2000, 2015, 2015, 2004, 2001, 2009, 2006, 2010, 2012, #31
                        2015, 2004, 2004, 2004, 2005, 1997, 2005, 2010, 1996, 2020, #41
                        2004, 1996, 2016, 1996, 2001, 2015, 2014, 2014, 2018, 2019, #51
                        2019, 2018, 2020, 2020, 2020],
         'elev_(m)': [25, 2168, 4, 1, 19, 20, 310, 103, 85, 39, np.nan, np.nan, np.nan, #13
                      385, 610, 103, 515, 734, 478, 395, 0.2, 81, 438, 126, 193, 385, #26
                      162, -1, 426, 540, 625, 150, 875, 452, 689, 982, 1639, #37
                      393, 1978, 493, 87, 167, 16, 160, 181, 270, 115, 45, 267, 153, #50
                      40, 67, 158, 2, 86, 181],
         'MAT_(C)': [9.8, 2.9, 14.2, 13, 15.2, 18, 10, 11.4, 10.8, 12.9, 16, 16, 18.3, #13
                     8.2, 7.5, 10, 7.7, 5.5, 7.6, 7.8, 9.2, np.nan, 8.34, np.nan, 9.1, #25
                     7.8, 8.5, 8.7, 7.7, 8.7, 7.1, 9.3, 6.7, 9.8, 8.3, 7.2, 2.8, 9.5, #38
                     2.3, 7.8, 10.3, 10, 9.8, 3.5, 3.5, 1.2, 7.4, 5.7, 1.8, 12.3, 11.7, #52
                     8.1, 14.1, 8.7, 8.3, 13.4],
         'MAP_(mm)': [786, 920, 920, 1100, 805, 600, 889, 679, 680, 960.1, 700, 700, #12
                      400, 843, 1250, 700, 1033, 996, 842, 901, 645, np.nan, 744, #23
                      np.nan, 563, 901, 470, 558, 604, 685, 681, 550, 1316, 1155, #34
                      1100, 1651, 1062, 1136, 918, 1062, 839, 800, 750, 711, 711, #45
                      523, 707, 544, 614, 700, 722, 961, 544, 809, 883, 677],
         'slope': ['flat', 'gentle', 'flat', 'medium', 'flat', 'medium', 'medium', 'gentle', 'flat', #9
                   'flat', 'flat', 'undulated', 'flat', 'gentle', 'medium', 'flat', #16
                   'medium', 'significant', 'flat', 'medium', 'flat', 'flat', 'gentle', #23
                   'flat', 'flat', 'flat', 'flat', 'flat', 'flat', 'significant', 'medium', #31
                   'flat', 'strong', 'flat', 'strong', 'hilltop', 'significant', 'flat', #38
                   'gentle', 'medium', 'gentle', 'gentle', 'flat', 'flat', 'gentle', #45
                   'flat', 'gentle', 'gentle', 'significant', 'gentle', 'flat', 'flat', #52
                   'flat', 'flat', 'flat', 'flat'],
         #slope: gentle (<2%), medium (2-5%), significant (5-10%), strong (>10%)
         'long_name': ['Loobos', 'Torgnon', 'San Rossore 2', 'Lison', 'Castelporziano2', #5
                       'Borgo Cioffi', 'Hesse', 'Fontainebleau-Barbeau', 'Estrees-Mons A28', 'Bilos', #10
                       'Majadas del Tietar South', 'Majadas del Tietar North', #12
                       'Albuera', 'Tharandt', 'Wüstebach', 'Selhausen Juelich', #16
                       'Rollesbroich', 'Oberbärenburg', 'Klingenberg', 'Hetzdorf', #20
                       'Huetelmoor', 'Grosses Bruch', 'Hainich', 'Hordorf', 'Hohes Holz', #25
                       'Grillenburg', 'Gebesee', 'Anklam', 'Trebon', 'Stitna', 'Rajec', #31
                       'Lanzhot', 'Bily Kriz forest', 'Oensingen 2', 'Lägeren', #35
                       'Früebüel', 'Davos', 'Chamau', 'Alp Weissenstein', 'Vielsalm', #40
                       'Maasmechelen', 'Lonzee', 'Brasschaat', 'Siikaneva', 'Hyytiälä', #45
                       'Degerö', 'Hyltemossa', 'Norunda', 'Svartberget', 'La Guette', #50
                       'Mejusseaume', 'Voulundgaard', 'Toulouse', 'Skjern', 'Gludsted Plantage', #55
                       'Lamasquere'],
         'network': ['ICOS_2', 'ICOS_a.', 'ICOS_2', 'ICOS_a.', 'ICOS_1', 'ICOS_1', #6
                     'ICOS_1', 'ICOS_1', 'ICOS_a.', 'ICOS_2', 'none', 'none', 'none', 'ICOS_1', #14
                     'ICOS_a.', 'ICOS_1', 'ICOS_a.', 'CarboEurope', 'ICOS_a.', 'ICOS_a.', #20
                     'TERENO', 'ICOS_a.', 'ICOS_a.', 'TERENO', 'ICOS_1', 'ICOS_a.', 'ICOS_1', #27
                     'none', 'ICOS_a.', 'CzeCOS', 'CzeCOS', 'ICOS_1', 'ICOS_2', 'CarboEurope', #34
                     'CarboEurope', 'CarboExtreme', 'ICOS_1', 'CarboExtreme', #38
                     'CarboExtreme', 'ICOS_2', 'ICOS_2', 'ICOS_2', 'ICOS_1', 'ICOS_2', #44
                     'ICOS_1', 'ICOS_2', 'ICOS_2', 'ICOS_2', 'ICOS_2', 'ICOS_a.', 'ICOS_a.', #51
                     'ICOS_1', 'ICOS_a.', 'ICOS_a.', 'ICOS_a.', 'ICOS_1'],
         'icos_l2_avail': [0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, #20
                           0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, #39
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1],
                     #0: no L2 meteo. parameters; 1: all L2 meteo. params.; 2: all but PBLH
         'comment': ['mostly scots pine', 'mostly Nardus stricta', 'mostly stone pine, dense', #3
                     'vineyard', 'only holm oak, nature reserve', 'agro-ecosystem (buffalos & corn)', #6
                     'beech forest, natural regeneration', 'dominated by Quercus petraea Liebl. and Quercus robur L.', #8
                     'annual food & perennial bioenergy crops', #9
                     'managed pine forest plantation, root depth limited by soil', #10
                     'extensive mgmt (grazing), N(itrogen) added', 'extensive mgmt, N & P added', #12
                     '', 'mostly norway spruce', 'mostly norway spruce, partly deforested since 2013', #15
                     '', 'mostly Ranunculus repens-Alopecurus pratensis community',
                     '', 'intesive farmland, 5-year crop rotation', '', 'coastal fen, restoration since 1992', #21
                     'extensive mgmt (grazing)', 'protected old-growth forest, mostly beech & ash', #23
                     'seasonal crop rotation', 'alluvial forest, mostly beech & oak', #25
                     'extensive mgmt', 'crop rotation', 'Peene valley fen', 'inundation area, sedges and wetland grasses', #29
                     'mature beech forest', 'old-growth spruce monoculture (being transformed to near-natural forest)', #31
                     'hard-wood floodplain forest, species-rich', '99% norway spruce', #33
                     'intensive crop rotation', 'diverse, managed forest, dominated by beech & fir', #35
                     'moderately intensive mgmt (2 cuts/y, grazing in fall)',
                     'mostly norway spruce (LAI ~3.9, little thinning)', 'intesive mgmt (6 cuts/y)', #38
                     'extensive mgmt (grazed)', 'mainly beech & douglas fir, thinned every 6y', #40
                     '400ha heathland dominated by Calluna vulgaris L.',
                     '4y rotation (sugarbeet, winter wheat, potato, winter wheat)', #42
                     'experimental scots pine forest surrounded by broadleaf species',
                     'Sedges, continuous Sphagnum moss cover, dwarf shrubs', 'homogenous Scots pine stand',
                     'Sphagnum moss', 'dominated by Picea abies, some Betula spp.', #47
                     'dominated by Norway spruce & Scots pine, some birch trees',
                     'boreal forest', 'degradated peatland', 'grazed grassland, occasional maize crops (fert.)', #51
                     'hedges to prevent erosion', 'grassland cut at least once a year',
                     'recently rewetted meadow in alluvial valley, occ. grazing', #54
                     'largest spruce plantation in DK, high yield mgmt and young stands',
                     'rotation: maize - winter wheat'],
         'soil': ['orthidystric rubic arenosol (sandy)', '', '', 'endogleyic calcisol', #4
                  '', '', '(stagnic) luvisol', 'gleyic luvisol', 'deep loamy soil', #9
                  'sandy podzol', '', '', '', '', #14
                  'silty clay loam', '', 'cambisol, stagnosol', '', '', '', 'sapric histosol', '', #22
                  'cambisol, 40% clay', 'chernozem', 'luvisol', 'silty loam (A), clayey loam (B)', #26
                  'chernozem', '', 'histosol', 'stagnic cambisol', 'modal oligotrophic cambisol', #31
                  '', 'haplic podzol, loamy-sand', 'eutri-stagnic cambisol', 'rendzic leptosol, haplic cambisol', #35
                  'gleysol', '', 'cambisol', 'on calcareous bedrock, (humous) sandy loam', #39
                  'silty & stony', '', 'loamy', 'albic hypoluvic arenosol', 'histosol', #44
                  '', 'histosol', '', 'sandy-loamy tills', '', 'histosol', '', #51
                  'sandy soil', '', '', '', 'alluvial soil']
         })
    
    flx_gdf = gpd.GeoDataFrame(flx, geometry=gpd.points_from_xy(flx.lon, flx.lat))
    flx_gdf.crs = 'EPSG:4326'
    flx_gdf = flx_gdf.sort_values(by='name')
    
    return flx_gdf

def nearest(items, target):
    return min(items, key=lambda x: abs(x - target))

def _fnv(values, target):
    '''
    A convenience function that finds the index of a value in a list closest
    to a target value. Can be used to select certain wavelengths of remote
    sensing sensors.
    '''
    if target > max(values) + 3:
        logger.warning('Max wavelength is {} and target is {}.'.format(max(values), target) +
                       'Will proceed with max WL.')
    if target < min(values) - 3:
        logger.warning('Min wavelength is {} and target is {}.'.format(min(values), target) +
                       'Will proceed with min WL.')
    if type(values) == list:
        idx = min(range(len(values)), key=lambda i:abs(values[i]-target))
    elif type(values) == np.ndarray:
        idx = np.abs(values-target).argmin()
    else:
        raise ValueError('wavelength values should be provided as list or np.ndarray')
    return idx

def _local_mask(raster, transform, shapes, **kwargs):
    '''
    Wrapper for rasterio.mask.mask to allow for in-memory processing.
    From: https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
    Args:
        raster (numpy.ndarray): raster to be masked with dim: [H, W]
        count (int): number of bands
        transform (affine.Affine): the transform of the raster
        shapes, **kwargs: passed to rasterio.mask.mask
    '''
    with rio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=raster.shape[2],
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            for i in range(raster.shape[2]):
                dataset.write(raster[:,:,i], i+1)
        with memfile.open() as dataset:
            output, out_trans = riom.mask(dataset, shapes, **kwargs)
        if output.ndim == 3:
            output = np.einsum('kli->lik', output)
            
    return output, out_trans

def desis_crop(path, mask, indexes = None):
    '''
    Reads DESIS GeoTiffs and crops using rasterio.mask methods (argument
    all_touched is always set to True).
        
    Args:
        path (string): The file path of the zipped PRISMA image to be cropped.
        mask (GeoJSON-like dict): passed to rasterio.mask.mask. Should have
            ETRS89-extended / LAEA Europe coordinates (epsg:3035).
        indexes (int / list of ints, optional): Indexes of the bands to be
            included. Use Python indexing starting at 0.
    '''
    if indexes == None:
        indexes = list(range(0, 235)) # all DESIS bands by default
    if isinstance(indexes, int):
        indexes_rio = indexes + 1
    elif isinstance(indexes, list):
        indexes_rio = [x + 1 for x in indexes]
    hdr = envi.read_envi_header(path.parent / (path.stem + '.hdr'))
    wls = [float(i) for i in hdr['wavelength']]
    with rio.open(path, driver='GTiff') as src:
        epsg = src.crs.to_epsg()
    # (re-)project mask derived from ICOS coordinates to HSI CRS
    crs_lam = proj.CRS.from_epsg('3035')
    crs_utm = proj.CRS.from_epsg(epsg)
    transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
    utm_mask = stransform(transf.transform, mask)
    with rio.open(path, driver='GTiff') as src:
        raw_cube, out_trans = riom.mask(src, shapes=[utm_mask], all_touched=True,
                                        crop=True, indexes=indexes_rio)
    if raw_cube.ndim == 2:
        out_ext = riop.plotting_extent(raw_cube, out_trans)
        temp = raw_cube.T
        gain = float(hdr['data gain values'][indexes])
        offset = float(hdr['data offset values'][indexes])
        na_val = float(hdr['data ignore value']) * gain
    else:
        out_ext = riop.plotting_extent(raw_cube[0], out_trans)
        temp = np.einsum('kli->lik', raw_cube)
        gain = np.array([float(x) for x in hdr['data gain values']])\
            [np.newaxis, np.newaxis, indexes]
        offset = np.array([float(x) for x in hdr['data offset values']])\
            [np.newaxis, np.newaxis, indexes]
        na_val = (float(hdr['data ignore value']) * gain[:,:,0]).item()
    fp_cube = offset + gain * temp # L=OffsetOfBand+GainOfBand*DN
    fp_cube = fp_cube.astype('float32')
    
    return fp_cube, wls, out_ext, na_val, out_trans, epsg

def prisma_crop(path, mask, indexes = None, swir = False):
    '''
    Reads zipped PRISMA imagery and crops using rasterio.mask methods (argument
    all_touched is always set to True).
    
    Args:
        path (string): The file path of the zipped PRISMA image to be cropped.
        mask (GeoJSON-like dict): passed to rasterio.mask.mask. Should have
            ETRS89-extended / LAEA Europe coordinates (epsg:3035).
        indexes (int / list of ints, optional): Indexes of the bands to be
            included. Use Python indexing starting at 0.
        swir (bool, optional): If true, bands in the 1000-2500 nm range are
            imported and concatenated with the VNIR cube.
    '''
    with ZipFile(path) as zf:
        for file in zf.namelist():
            if not file.endswith('.he5'): # optional filtering by filetype
                logger.error('{}: Zip doesnt contain .he5 file.'.format(path))
                continue
            with zf.open(file) as f:
                with h5py.File(f, mode='r') as h5f:
                    vnir_wls = pd.Series(
                        np.flip(h5f.attrs['List_Cw_Vnir'])).drop(range(63, 66))
                    vnir_raw = h5f['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'][:]
                    #err_mat = h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX'][:]
                    
                    geo = {'xmin': min(h5f.attrs['Product_ULcorner_easting'], h5f.attrs['Product_LLcorner_easting']),
                           'xmax': max(h5f.attrs['Product_LRcorner_easting'], h5f.attrs['Product_URcorner_easting']),
                           'ymin': min(h5f.attrs['Product_LLcorner_northing'], h5f.attrs['Product_LRcorner_northing']),
                           'ymax': max(h5f.attrs['Product_ULcorner_northing'], h5f.attrs['Product_URcorner_northing']),
                           'proj_code': h5f.attrs['Projection_Id'],
                           'proj_name': h5f.attrs['Projection_Name'],
                           'proj_epsg': h5f.attrs['Epsg_Code']
                           }
                    smax = h5f.attrs['L2ScaleVnirMax']
                    smin = h5f.attrs['L2ScaleVnirMin']
                    if swir == True:
                        swir_wls = pd.Series(
                            np.flip(h5f.attrs['List_Cw_Swir'])).drop(range(0, 6))
                        swir_raw = h5f['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'][:]
                        smax_sw = h5f.attrs['L2ScaleSwirMax']
                        smin_sw = h5f.attrs['L2ScaleSwirMin']
                    
    rows0,_,cols0 = np.shape(vnir_raw)
    geo['xsize'] = (geo['xmax']-geo['xmin'])/cols0
    geo['ysize'] = (geo['ymax']-geo['ymin'])/rows0
    
    bbox = [geo['xmin'], geo['ymin'], geo['xmax'], geo['ymax']]
    
    vnir_cube = np.einsum('kli->kil', np.flip(vnir_raw, 1))
    vnir_refl = (smin + vnir_cube * (smax - smin)) / 65535
    
    if swir == True:
        swir_cube = np.einsum('kli->kil', np.flip(swir_raw, 1))
        swir_refl = (smin_sw + swir_cube * (smax_sw - smin_sw)) / 65535
        
        full_cube = np.concatenate([vnir_refl[:, :, vnir_wls.index],
                                    swir_refl[:, :, swir_wls.index]], axis=2)
        wls = pd.concat([vnir_wls, swir_wls]).reset_index(drop=True)
    else:
        full_cube = vnir_refl[:, :, vnir_wls.index]
        wls = vnir_wls
    
    if indexes == None:
        indexes_rio = [x + 1 for x in wls.index.tolist()]
    elif isinstance(indexes, int):
        indexes_rio = indexes + 1
    elif isinstance(indexes, list):
        indexes_rio = [x + 1 for x in indexes]

    rows, cols = np.shape(full_cube)[0:2]

    src_trans = rio.transform.from_bounds(*bbox, width=cols, height=rows)
    
    # (re-)project mask derived from ICOS coordinates to HSI CRS
    epsg = geo['proj_epsg']
    crs_lam = proj.CRS.from_epsg('3035')
    crs_utm = proj.CRS.from_epsg(epsg)
    transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
    utm_mask = stransform(transf.transform, mask)
    # In rio.Affine: last value is ymax, not ymin!
    fp_cube, out_trans = _local_mask(full_cube, src_trans, [utm_mask], crop=True,
                                    all_touched=True, indexes=indexes_rio)
    out_ext = riop.plotting_extent(fp_cube, out_trans)
    na_val = 0
    
    return fp_cube, wls, out_ext, na_val, out_trans, epsg


class HSICOS():
    
    def __init__(self, wdir, img_csv, do_mkdir = False, out_dir = False, era_blh = []):
        '''
        Args:
            wdir (string): The directory where BSQ files and the GeoPackage
                with ROI bounding box are located.
            img_csv (string): Name of the metadata CSV file that contains key
                info about the imagery to be processed. Required columns for
                cropping are:
                'name': Name or abbr. of the corresp. ICOS site
                'startdate': Timestamp of image acquisition (YYYY-MM-DD hh:mm:ss)
                'dataTakeID': datatake ID of the scene
                'clouds': commentary field to mark missing data, etc.
                'era_pblh': planetary boundary layer height value from ERA5 re-
                    analysis. Only used when PBLH is missing from ICOS L2 data.
                Colums 'date' and 'icostime' are created when the class is ini-
                tiated to match ICOS standards (e.g. conversion of TZ from UTC
                to CET)
            do_mkdir (bool, optional): If true, the output directory is created.
            out_dir (string, optional): Set a custom name for the working
                folder, e.g. to continue previously produced results. If
                unused, a generic output directory name will be used.
        '''
        self._wdir = Path(wdir)
        self.icos_dir = self._wdir / 'fluxes/level2'
        self._dp_icos = lambda x: dt.datetime.strptime(x, '%Y%m%d%H%M')
        
        self._dp = lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.img_db = pd.read_csv(
            self.icos_dir.parent / img_csv, parse_dates=['startdate'],
            date_parser=self._dp, dtype={'dataTakeID': str}).reset_index(drop=True)
        
        # check if metadata table is suitable
        if all(x in self.img_db.columns for x in \
               ['name', 'startdate', 'dataTakeID', 'clouds']):
            pass
        else:
            raise ValueError('Missing columns in "img_db". Please prepare the '\
                             'data frame according to documentation.')
        # check datetime format
        checkdate = self.img_db[['name', 'startdate']].select_dtypes(include=[np.datetime64])
        if checkdate.shape[1] == 0:
            raise ValueError('Incorrect datetime format, should be np.datetime64.')
        
        self.img_db['date'] = [dt.datetime.date(x).isoformat() for x in self.img_db['startdate']]
        self.img_db['icostime'] = (self.img_db.startdate +
                                   pd.Timedelta(hours=1)).dt.round('30min').dt.time
        self.img_db['wls'] = 0
        self.img_db['wls'] = self.img_db['wls'].astype('object')

        try: # read or build icos sites metadata table
            self.icos_db = gpd.read_file(self.icos_dir.parent / 'flx_sites.gpkg', driver='GPKG')
        except ferr.CPLE_OpenFailedError:
            self.icos_db = _build_icos_meta(save=False)
        
        # get sensor type from ID formatting
        if '_' in self.img_db.dataTakeID.iloc[0]:
            self.sensor = 'DESIS'
        else:
            self.sensor = 'PRISMA'
        self.img_dir = self._wdir / 'Data' / self.sensor
        
        # create output folder
        ssm = round((dt.datetime.now() - dt.datetime.now().replace(hour=0, minute=0,second=0, microsecond=0)).total_seconds())
        if out_dir == False:
            fnamestr = 'hsicos_out_' + dt.date.today().isoformat() + str(ssm)
        elif isinstance(out_dir, str):
            fnamestr = out_dir
        else:
            raise ValueError('Fname must be a string.')
        self.out_dir = self.icos_dir.parent / fnamestr
        if do_mkdir == True:
            try:
                self.out_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print('Output directory already exists')
        
        # Aggregated geom info for ICOS sites is loaded if available
        flx_loc_path = self.img_dir / '{}_hsicos_crs_loc.csv'.format(self.sensor)
        if flx_loc_path.exists():
            flx_df = pd.read_csv(flx_loc_path, on_bad_lines='skip')
            self.flx_loc = gpd.GeoDataFrame(flx_df, geometry=gpd.points_from_xy(
                flx_df.lon, flx_df.lat), crs='EPSG:4326')
            logger.info('CRS and location overview of ICOS sites loaded to self.flx_loc.')
        
        # check existing cropped PPI files
        self.img_db['ppi_file'] = ''
        ppi_dir = self.img_dir.parent / 'Copernicus/S2PPI'
        ppif_list = [x.name for x in list(ppi_dir.glob('*VI_PPI*.tif'))]
        
        for fname in ppif_list:
            self.img_db.loc[(self.img_db.name == fname[-25:-19]) &
                            (self.img_db.dataTakeID == fname[-18:-4]), 'ppi_file'] = fname
        logger.info('S2 PPI images locally available for {} data takes.'\
                    .format(len(self.img_db.loc[self.img_db.ppi_file != '', :])))
        

        
    
### QUALITY CHECKS ############################################################

    def crs_and_cropping(self, icos_list, zip_path = None, date = None,
                        overwrite = False, save_csv = False):
        '''
        Imports geospatial information for ICOS sites into the class and crops
        HSI to a 3km buffer around the site. Location is imported from ICOS
        data in WGS84. Projected CRS is imported from the hyperspectral data
        inside the PRISMA_crop function. It is assumed that all imagery for a
        single ICOS site has the same CRS (which might not be the case in very
        rare cases). Crops are saved as GeoTIFF in 'img_dir'.
        
        Args:
            icos_list (string / list of strings): Abbreviation of the ICOS
                site(s) for which geometry information will be collected.
            zip_path (string, optional): Location of zipped PRISMA imagery,
                e.g. if located on external drive instead of self.img_dir.
            date (string, optional): Date of image to be cropped. If no date is
                supplied, all available dates for the site will be processed.
            overwrite (bool, optional): If true, imagery for which a cropped
                file already exists will not be skipped.
            save_csv (bool, optional): If true, the collected CRS & geographic
                information will be saved inside the imagery folder.

        '''
        if isinstance(icos_list, str):
            icos_list = [icos_list]
        
        if zip_path:
            img_dir = Path(zip_path)
        else:
            img_dir = self.img_dir
        
        crslist = [0]*len(icos_list)
        lats = [0]*len(icos_list)
        lons = [0]*len(icos_list)
        for j, site in enumerate(icos_list):
            if date:
                datelist = self.img_db.loc[(self.img_db.name == site) &
                                           (self.img_db.date == date), 'startdate'].dt.date
            else:
                datelist = self.img_db.loc[self.img_db.name == site, 'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            
            if self.sensor == 'DESIS':
                img_paths = [list((self.img_dir).glob('*' + dt + '*SPECTRAL*.tif')) for dt in dtakes]
                crop_paths = [self.img_dir / 'DESIS-HSI-L2A-DT{}_{}_6km_crop.tif'.format(
                    dt.zfill(14), site) for dt in dtakes]
            elif self.sensor == 'PRISMA':
                img_paths = [list((img_dir).glob('*' + dt + '*.zip')) for dt in dtakes]
                crop_paths = [self.img_dir / 'PRS_L2D_STD_{}_{}_6km_crop.tif'.format(
                    dt, site) for dt in dtakes]
                
            crop_exists = [p.is_file() for p in crop_paths]
            if all(crop_exists):
                logger.warning('{}: All HSI are already cropped.'.format(site))
                return
            
            no_crop = [i for (i, v) in zip(dtakes, crop_exists) if not v]
            logger.info('{}: No cropped image found for data takes {}.'.format(site, no_crop))
            
            # 1) Load flux locations
            cols = ['VARIABLE', 'DATAVALUE']
            icos_exp = self.icos_dir / ('ICOSETC_{}_FLUXES_INTERIM_L2.csv'.format(site))
            if icos_exp.exists(): # Create file name template for single ICOS site
                site_info = pd.read_csv(
                    self.icos_dir / 'ICOSETC_{}_SITEINFO_INTERIM_L2.csv'\
                        .format(site), on_bad_lines='skip', usecols=cols)
            else:
                site_info = pd.read_csv(
                    self.icos_dir / 'ICOSETC_{}_SITEINFO_L2.csv'.format(site),
                    on_bad_lines='skip', usecols=cols)
            
            lats[j] = site_info.loc[site_info.VARIABLE == 'LOCATION_LAT',
                               'DATAVALUE'].astype(float).item()
            lons[j] = site_info.loc[site_info.VARIABLE == 'LOCATION_LONG',
                               'DATAVALUE'].astype(float).item()            
            
            flx_loc = gpd.GeoSeries(data=[Point(lons[j], lats[j])], crs='EPSG:4326').to_crs(3035)
            flx_roi_crop = flx_loc.buffer(3000) # lambert -> buffer in meters
            box_geom_crop = geometry.box(*flx_roi_crop.total_bounds)
            # 2) Load CRS & crop
            cubes = [0]*len(img_paths)
            wlss = [0]*len(img_paths)
            blength = []
            img_crss = [0]*len(img_paths)
            itransforms = [0]*len(img_paths)
            for i, path in enumerate(img_paths):
                if crop_exists[i] & (overwrite == False): # only process paths for which cropped imagery is missing:
                    continue
                if len(path) == 0:
                    logger.info('{}: The {} image with dataTakeID {} could not be found.'\
                                .format(site, self.sensor, dtakes.iloc[i]))
                    continue
                elif len(path) > 1:
                    raise ValueError('dataTakeID {} not unique. Please investigate.'\
                                     .format(dtakes.iloc[i]))
                if self.sensor == 'DESIS': # cubes = 6km crops for saving subset as GeoTIFF
                        cubes[i], wlss[i], _, na_val, itransforms[i], img_crss[i] =\
                            desis_crop(path[0], box_geom_crop)
                elif self.sensor == 'PRISMA':
                    cubes[i], wlss[i], _, na_val, itransforms[i], img_crss[i] =\
                        prisma_crop(path[0], box_geom_crop, swir=True) # SWIR should always be saved in cropped TIFF
                cubes[i] = np.round(cubes[i], 6)
                blength.append(len(wlss[i])) #usually for DESIS 235, for PRISMA 63 (VNIR) / 230 (VSWIR).
            if len(set(blength)) > 1:
                raise ValueError('{}: Wavelengths are not identical for all images. '.format(site) +
                                 'Wavelengths by order in img_db: {}'.format(blength))
            print(img_crss)
            img_crss = [x for x in img_crss if x != 0] # remove zeros from skipped images
            if len(img_crss) == 0:
                logger.info('{}: No HSI to be processed! Skipping site.'.format(site))
                continue
            elif len(set(img_crss)) > 1:
                logger.error('{}: CRS is not identical for all images!'.format(site))
            # unique CRS for all images (usually CRS will be unique across images from a single ICOS site)
            epsg = pd.Series(img_crss).value_counts().idxmax()
            crs_utm = proj.CRS.from_epsg(epsg)
            for i, cube in enumerate(cubes):
                if not isinstance(cube, np.ndarray): # only process cubes that could be loaded in first inner loop
                    continue
                rows, cols, b = np.shape(cube) # dimension order will be changed back to [b, y, x] when reading with rasterio
                ometa = {'driver': 'GTiff',
                         'dtype': 'float32',
                         'interleave': 'band',
                         'nodata': na_val,
                         'width': cols,
                         'height': rows,
                         'count': b,
                         'crs': crs_utm,
                         'transform': itransforms[i]}
                logger.info('saving crop of data take {} with CRS - EPSG:{}.'\
                            .format(dtakes.iloc[i], epsg))
                wls_string = [str(w) for w in wlss[i]]
                with rio.open(crop_paths[i], 'w', **ometa) as dst:
                    for k in range(0, b):
                        dst.write_band(k+1, cube[:,:,k])
                        dst.set_band_description(k+1, wls_string[k])

            crslist[j] = crs_utm.to_epsg()
        
        flx_df = pd.DataFrame(list(zip(icos_list, crslist, lons, lats)),
                              columns=['name', 'sensorcrs', 'lon', 'lat'])
        if save_csv == True:
            flx_loc_path = self.img_dir / '{}_hsicos_crs_loc.csv'.format(self.sensor)
            flx_df.to_csv(flx_loc_path, index=False)
        self.flx_loc = gpd.GeoDataFrame(flx_df, geometry=gpd.points_from_xy(
            flx_df.lon, flx_df.lat), crs='EPSG:4326')
        
        return
        

    def hsi_qc(self, icos_site, clip_dist = [1.0, 3.0], date = None, save = False):
        '''
        TODO: adjust to 6km TIFFs and zipped .he5 files!!!
        Quality checking function for DESIS & PRISMA imagery. Checks if imagery
        covers flux site and is not located on no data edge of an image. Gen-
        erates quicklook graphics to examine potential atmospheric disturbances.
        As assessment of image quality, clouds, etc. need visual examination,
        this function does not remove entries from 'img_db' automatically. Un-
        suitable entries have to be removed manually from the table!
        
        Args:
            icos_site (string): Abbreviation of the ICOS site for which DESIS
                imagery will be evaluated.
            icos_dir (string): The directory where ICOS location KMLs are stored.
            clip_dist (list of floats, optional): 2 buffer values in kilometers.
                The second value is used for cropping the scene while the first is
                only used for plotting.
            date (string, optional): A specific date (YYYY-MM-DD) can be passed to
                process a single image.
            save (bool, optional): If true, quicklooks are generated and saved
                to wdir.
        '''
        
        if isinstance(icos_site, str):
            icos_site = [icos_site]
        filestatus_l = [0]*len(icos_site)

        for j,site in enumerate(icos_site):
            if date == None:
                datelist = self.img_db.loc[
                    self.img_db.name == site, 'startdate'].dt.date
            else:
                try:
                    dt.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError('Incorrect date format, should be YYYY-MM-DD')
                datelist = self.img_db.loc[(self.img_db.name == site) & \
                                           (self.img_db.date == date),
                                           'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            filestatus = [0]*len(datelist)
            filemissing = [-99]*len(datelist)
            
            if self.sensor == 'DESIS':
                for i, dtake in enumerate(dtakes): # little loop to check imagery existence
                    filemissing[i] = len([x for x in self.img_dir.glob('*' + dtake + '*.tif')])
            elif self.sensor == 'PRISMA':
                for i, dtake in enumerate(dtakes):
                    filemissing[i] = len([x for x in self.img_dir.glob('*' + dtake + '*.he5')])
            
            nm_ix = np.where(np.array(filemissing) > 0)[0]
            if len(nm_ix) == 0:
                filestatus_l[j] = ['missing'] * len(datelist)
                continue # early end of iteration if no data found 
            
            flx_loc = self.flx_loc.loc[self.flx_loc.name == site, 'geometry']
            flx_roi = flx_loc.buffer((max(clip_dist)+1)*1000)
            box_geom = geometry.box(*flx_roi.total_bounds)
            flx_roi_na = flx_loc.buffer(1000) # 1000 m buffer for checking NaN ratio
            box_geom_na = geometry.box(*flx_roi_na.total_bounds)
            
            for i, dtake in enumerate(dtakes):
                # Adjust UTM zone to imagery location
                if self.sensor == 'DESIS': # after sorting: file 0 = 10-band QC image,
                # file 1 = spectral image, file 2 = RGB quicklook, file 4 = unused
                    fnames = sorted([x for x in self.img_dir.glob('*' + dtake + '*.tif')],
                                    key=lambda path: path.name[64:65]) # strange key index to get reliable sorting. don't ask...
                elif self.sensor == 'PRISMA':  # PRISMA offers no quality overviews -> 1 file per dtake
                    fnames = [x for x in self.img_dir.glob('*' + dtake + '*.he5')]
                if len(fnames) == 0: # check if file is present
                    print('The image with dataTakeID {} could not be found.'.format(dtake))
                    filestatus[i] = 'missing'
                    continue
                # Load & crop or create RGB quicklook
                try: # error handling if research site is not covered by input raster
                    if self.sensor == 'DESIS':
                        # fmch.desis_crop can't be used as it corrects hyperspectral digital numbers
                        # and uses a .hdr file. Both is not usable for the RGB quicklook
                        with rio.open(fnames[2], driver='GTiff') as src:
                            desis_ql, _ = riom.mask(src, shapes=[box_geom], crop=True, indexes=[3,2,1])
                        #out_ext = riop.plotting_extent(desis_ql[0], _)
                        quality_ql = np.einsum('kli->lik', desis_ql)
                        desis_cube, wls, out_ext, na_val, itrans, _ = desis_crop(
                            fnames[1], box_geom)
                        
                        na_rate_plot = (desis_cube == na_val).sum(axis=2) / desis_cube.shape[2]
                        
                        # DESIS only: Load quality information layer
                        with rio.open(fnames[0], driver='GTiff') as src:
                            desis_qi, _ = riom.mask(src, shapes=[box_geom], crop=True)
                            # 0=shadow, 1=clear land, 3=haze over land, 5=cloud over land
                        
                        # plotting params
                        pp = [2, 3, 27, 18, True, True]
                        titles = ['RGB', 'NA rate', 'Shadow', 'Haze', 'Clouds']
                    elif self.sensor == 'PRISMA':
                        prisma_cube, wls, out_ext, na_val, itrans, _ = \
                            prisma_crop(fnames[0], box_geom)
                        rows, cols, b = np.shape(prisma_cube)
                        prisma_cubeT = prisma_cube.reshape(-1,b)
                        prisma_rgb = HSI2RGB(wls, prisma_cubeT, rows, cols, 50, 0.0002)
                        # transform into 8-bit RGB to match DESIS quality quicklook format
                        quality_ql = (prisma_rgb*255).round(0).astype(int)
                        
                        na_rate_plot = (prisma_cube == na_val).sum(axis=2) / prisma_cube.shape[2]
                        
                        pp = [1, 2, 18, 9, False, True]
                        titles = ['RGB', 'NA rate']
                except ValueError as e:
                    if str(e) == 'Input shapes do not overlap raster.':
                        filestatus[i] = 'geom_error'
                        continue
                    else:
                        raise
                # Check NA ratio
                na_rate_crop, _ = _local_mask(na_rate_plot[:,:,np.newaxis],
                                              itrans, [box_geom_na], crop=True)
                na_ratio = round(na_rate_crop.mean(), 2)
    
                if save == False:
                    continue
                else:
                    if na_ratio >= 0.1:
                        logger.warning('Averaged per-pixel NaN rate over all bands is {} for DataTake {} within {}km around the ICOS location!'\
                                       .format(na_ratio, dtake, min(clip_dist)))  
                        filestatus[i] = 'nodata_warn'
                    else:
                        filestatus[i] = 'good'
                        
                    fig, axes = plt.subplots(pp[0], pp[1], figsize=(pp[2], pp[3]),
                                             sharex=pp[4], sharey=pp[5])
                    divider = make_axes_locatable(axes.flat[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    if self.sensor == 'DESIS':                
                        axes[0,0].imshow(quality_ql, extent=out_ext)
                        axes[0,1].imshow(na_rate_plot, extent=out_ext)
                        axes[0,2].imshow(desis_qi[0,:,:], extent=out_ext)
                        axes[1,0].imshow(desis_qi[3,:,:], extent=out_ext)
                        axes[1,1].imshow(desis_qi[5,:,:], extent=out_ext)
                        for i,ax in enumerate(axes.flat[:-1]):
                            flx_loc.plot(ax=ax, marker='x', markersize=100, c='crimson')
                            flx_loc.buffer(clip_dist[0]*1000)\
                                .plot(ax=ax, fc='none', ec='crimson', linewidth=2)
                            flx_loc.buffer(clip_dist[1]*1000)\
                                .plot(ax=ax, fc='none', ec='purple', linewidth=2)
                            ax.title.set_text(titles[i])
                        qc_fn = 'DESIS-HSI-L2A-DT{}_{}_qc.png'.format(dtake.zfill(14), site)
                    elif self.sensor == 'PRISMA':                
                        axes[0].imshow(quality_ql, extent=out_ext)
                        im = axes[1].imshow(na_rate_plot, extent=out_ext)
                        fig.colorbar(im, cax=cax, orientation='vertical')
                        for i,ax in enumerate(axes.flat):
                            flx_loc.plot(ax=ax, marker='x', markersize=100, c='crimson')
                            [flx_loc.buffer(d*1000)\
                             .plot(ax=ax, fc='none', ec='crimson', lw=2) for d in clip_dist]
                            ax.title.set_text(titles[i])
                        qc_fn = 'PRS_L2D_STD_{}_{}_qc.png'.format(dtake, site)
                    fig.savefig(self.img_dir / qc_fn, pad_inches=0, bbox_inches='tight', dpi=150)
                    plt.close(fig)
            filestatus_l[j] = filestatus
        filestatus_all = sum(filestatus_l, [])
        filestatus_df = pd.concat([self.img_db[['name', 'dataTakeID']],
                                   pd.Series(filestatus_all).rename('status')], axis=1)
        return filestatus_df


    def var_check(self, site_list):
        '''
        A function to check which ICOS station data does not include certain
        (micrometeorological) variables that are needed for flux footprint 
        modeling. If only planetary boundary layer height (PBLH) is missing,
        it can be substituted from ERA5 reanalysis data. If wind speed & di-
        rection (WS, WD) are missing in the "...FLUXES.csv" they will be auto-
        matically imported from the "...METEO.csv" instead.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) to be checked.
        '''
        cols = ['TIMESTAMP_START', 'TIMESTAMP_END', 'USTAR', 'V_SIGMA', 'MO_LENGTH', 'PBLH', 'WS', 'WD']
        l2_var_count = [0] * len(site_list) 
        for i, site in enumerate(site_list):
            icos_exp = self.icos_dir / ('ICOSETC_' + site + '_FLUXES_INTERIM_L2.csv')
            if icos_exp.exists(): # Create file name template for single ICOS site
                icos_v = 'ICOSETC_' + site + '_{}_INTERIM_L2.csv'
            else:
                icos_v = 'ICOSETC_' + site + '_{}_L2.csv'
                
            flx_hh = pd.read_csv(self.icos_dir / icos_v.format('FLUXES'),
                                 date_parser=self._dp_icos, nrows=0,
                                 parse_dates=cols[:2],
                                 usecols=lambda x: x in cols)
            tvar = cols[2:]
            if pd.Series(tvar).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 6
                print('{}: Full L2 vars'.format(site))
            elif pd.Series(tvar[0:4]).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 4
                print('{}: Only missing WS, WD'.format(site))
            elif pd.Series(tvar[0:3]).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 3
                print('{}: Missing PBLH and/or WS, WD -> era_blh site!'.format(site))
            else:
                print('{}: Missing more than 50% of required FFP model variables.'\
                      .format(site))
        result = pd.DataFrame({'name': site_list,
                               'l2_var_count': l2_var_count})
        miss = result.loc[result.l2_var_count == 3, 'name'].tolist()
        prob = result.loc[result.l2_var_count < 3, 'name'].tolist()
        # Remove era_blh entries if no usable imagery is available at all
        no_usable = self.img_db.groupby('name').usable.sum()
        miss = [x for x in miss if x not in no_usable[no_usable == 0].index.tolist()]
        
        logger.warning('ICOS sites {} dont have planetary '.format(miss) +
              'boundary layer height (PBLH) data. Use the "fmch.icos_era5_get" ' +
              'method to download the data from ERA5 and use the mentioned ' +
              'stations as "era_blh" variable when using the "update_img_db" ' +
              'method.')
        if len(prob) != 0:# result any less 3: throw error:
            logger.error('ICOS sites {} have missing FFP '.format(prob) +
                         'variables that will cause problems later!')
        
        return miss

#### COPERNICUS CDS/WEKEO FUNCTIONS ###########################################

    def icos_cds_get(self, icos_list, eobs = False, version = '25.0e',
                     ts = True, suffix = None):
        '''
        Requests and downloads single ERA5 PBLH pixels (~30x30 km, specific
        datetimes) or multiple E-OBS temperature & precipitation pixels (~10x10
        km, time series from 1950-01-01 to 2021-06-30) located at ICOS site(s).
        Downloaded variables are saved as .zip (E-OBS) or .nc (ERA5).
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which ERA 5 data will be requested.
            eobs (bool): If true, E-OBS temperature and precipitation data are
                downloaded. Otherwise, ERA5 PBLH data is downloaded.
            version (string): Version of the dataset. Check your request online
                (https://cds.climate.copernicus.eu/) to get version info.
            ts (bool, optional): If true, datetimes are provided to the ERA5
                API as timestamp values.
            suffix (str, optional): A string to be appended to the .nc file.
                Not used for E-OBS data as these cover decades.
        '''
        if suffix is None:
            suffix = self.sensor
        if isinstance(icos_list, str):
            icos_list = [icos_list]

        for i,site in enumerate(icos_list):
            c = cdsapi.Client()
            if (self.flx_loc.name == site).any():
                flx_loc = self.flx_loc.loc[self.flx_loc.name == site, 'geometry']
            else: # backup for ICOS locations which are not in img_db
                flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry']
            if (not 30 < flx_loc.y.item() < 70) or not (-15 < flx_loc.x.item() < 45):
                raise ValueError('Coordinates for {} are out of '.format(site) +
                                 'the usual range for ICOS sites in Europe.' +
                                 ' Please investigate.')
            if eobs:
                ext = flx_loc.buffer(0.2, cap_style=3).bounds.squeeze(axis=0) # 0.2 deg!
                dataset = 'insitu-gridded-observations-europe'
                params = {
                    'format': 'zip',
                    'grid_resolution': '0.1deg',
                    'product_type': 'ensemble_mean',
                    'variable': ['maximum_temperature', 'mean_temperature',
                                 'minimum_temperature', 'precipitation_amount'],
                    'period': 'full_period',
                    'version': version,
                    'area': [ext.maxy, ext.minx, ext.miny, ext.maxx],
                    }
            else:
                epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
                crs_utm = proj.CRS.from_epsg(epsg)
                flx_loc = flx_loc.to_crs(crs_utm)
                ext = flx_loc.buffer(100, cap_style=3).bounds.squeeze(axis=0)
                if ts == True:
                    datelist = self.img_db.loc[self.img_db.name == site,
                                               'startdate'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                else:
                    datelist = [dt.datetime.date(x).isoformat() for x in self.img_db['startdate']]
                    # round to 1h to match corresp. ERA5 time slices
                    times = self.img_db.loc[self.img_db.name == site,
                                       'startdate'].dt.round('1h').dt.time.unique()
                    timelist = [x.strftime('%H:%M') for x in times]
                    timelist.sort()
                
                dataset = 'reanalysis-era5-complete'
                params = {
                    'format': 'netcdf',
                    'class': 'ea',
                    'date': datelist,
                    'expver': '1',
                    'levtype': 'sfc',
                    'param': '159.128', # PBLH
                    'stream': 'oper',
                    'type': 'an',
                    'grid': [0.25, 0.25],
                    'area': [ext.maxy, ext.minx, ext.miny, ext.maxx],
                    }
                if ts != True:
                    params['time'] = timelist # times are UTC! 
            
            request = c.retrieve(dataset, params)
            cdspath = self.icos_dir / site
            cdspath.mkdir(parents=True, exist_ok=True)
            if eobs:
                request.download(cdspath / 'e-obs_meteo_large.zip')
            else:
                request.download(cdspath / ('era5_pblh_' + suffix + '.nc'))
        return

    def icos_eobs_pet(self, icos_list):
        '''
        Extracts E-OBS precipitation and temperature data for ICOS site(s).
        Resulting precip. & PET time series are saved as NetCDF.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which ERA 5 data will be requested.
        '''
        if isinstance(icos_list, str):
            icos_list = [icos_list]

        for i,site in enumerate(icos_list):
            cdspath = self.icos_dir / site
            variables = ['tn', 'tx', 'tg', 'rr'] # tmean, tmin, tmax, precipitation
            # caution: variable order matters: hargreaves function needs mean>min>max order
            
            datasets0 = [xr.open_dataset(list(cdspath.expanduser(
                ).glob('*'+ v +'_ens_mean_0_area_subset.nc'))[0]) for v in variables]
            ndays = len(datasets0[0].time)
            
            pet = np.zeros(ndays)
            datasets = [0]*len(datasets0)
            js = [0]*len(datasets0)
        
            eobs_tg_t = pd.to_datetime(datasets0[0].time.values).dayofyear
            if eobs_tg_t[-1] != 365:
                logger.warning('E-OBS time series for {}: '.format(site) +
                               'Last DOY is not 365 but {}'.format(eobs_tg_t[-1]))
            eobs_tg_times = pd.to_datetime(datasets0[0].time.values)
            
            if (self.flx_loc.name == site).any():
                flx_loc = self.flx_loc.loc[self.flx_loc.name == site, 'geometry']
            else: # backup for ICOS locations which are not in img_db
                flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry']
            lat_flx = flx_loc.y.item()
            lon_flx = flx_loc.x.item()
            lat_r = pyeto.deg2rad(lat_flx)  # Convert latitude to radians
            
            # NA handling loop
            for i, (ds0, v) in enumerate(zip(datasets0, variables)):
                lats = ds0.latitude.values
                lons = ds0.longitude.values
                lat_diff = abs(lats - lat_flx) # differences list to find min. distance...
                lon_diff = abs(lons - lon_flx) # from flux tower to E-OBS grid cell
                lat = lats[np.argmin(lat_diff)]
                lon = lons[np.argmin(lon_diff)]
                
                # Slice 1 test data set for NA check. Increase indices using sorted differences
                # list to try neighboring grid cells instead of the closest cell
                ds = ds0[v].sel(longitude=lon, latitude=lat).values
                na_days = len(ds[np.isnan(ds)])
                if (na_days / ndays) > .5:
                    lat_diff_s = sorted(abs(lats - lat_flx))
                    lon_diff_s = sorted(abs(lons - lon_flx))
                    j = 0
                    while (na_days / ndays) > .5:
                        j = j + 1
                        if j == len(lats):
                            raise ValueError('No grid cell with <50% NA values' +
                                             ' found for ICOS site {}'.format(site))
                        lat = lats[lat_diff == lat_diff_s[j]]
                        ds = ds0[v].sel(longitude=lon, latitude=lat).values
                        na_days = len(ds[np.isnan(ds)])
                        if (na_days / ndays) > .5: # second check allows to change lat
                        # and lon value in turns
                            lon = lons[lon_diff == lon_diff_s[j]]
                            ds = ds0[v].sel(longitude=lon, latitude=lat).values
                            na_days = len(ds[np.isnan(ds)])
                    js[i] = j
                datasets[i] = ds
            datasets = [np.squeeze(ds).astype('float64') for ds in datasets]
                
            if (np.array(js) == 0).all():
                print('Lat/lon of ICOS site {} '.format(site) +
                      'is suitable for all E-OBS datasets.')
            else:
                print('''Lat/lon has been shifted by {} cells due to
                      NA values for ICOS site {}'''.format(js, site))
    
            # using either list comprehensions ...
            sol_dec = np.array([pyeto.sol_dec(xi) for xi in eobs_tg_t])
            sha = np.array([pyeto.sunset_hour_angle(lat_r, xi) for xi in sol_dec])
            ird = np.array([pyeto.inv_rel_dist_earth_sun(xi) for xi in eobs_tg_t])
            # or lambda functions inside np.apply_along_axis to process numpy arrays row-wise
            et_rad_args = np.stack((sol_dec, sha, ird), axis=1)
            et_rad = np.apply_along_axis(lambda x: pyeto.et_rad(lat_r, x[0], x[1], x[2]), 1, et_rad_args)
            
            pet_args = np.stack(datasets[0:3] + [et_rad], axis=1)
            pet = np.apply_along_axis(lambda x: pyeto.hargreaves(x[0], x[1], x[2], x[3]), 1, pet_args)
            
            # Build xarrays for .nc saving
            pet_ds = xr.Dataset(
                {'pet': (('time'), pet)},
                coords={'time': eobs_tg_times})
            pre_ds = xr.Dataset(
                {'pre': (('time'), datasets[3])},
                coords={'time': eobs_tg_times})
            pet_ds.to_netcdf(cdspath / 'eobs_pet.nc')
            pre_ds.to_netcdf(cdspath / 'eobs_pre.nc')
        return

    
    def _icos_era5_pblh(self, icos_site, nc_name = None):
        '''
        Extracts ERA5 data matching the acquisition timestamps of available
        hyperspectral imagery for a single ICOS site.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site for which ERA 5
                data will be extracted.
            nc_name (string, optional): Name of the NetCDF file that stores
                ERA5 PBLH time series. The file must only contain one pixel.
        '''
        if nc_name is None:
            nc_name = 'era5_pblh_{}.nc'.format(self.sensor)
        era5 = xr.open_dataset(self.icos_dir / icos_site / nc_name)
        
        era5_times = pd.to_datetime(era5.blh.time.values)
        img_db_site = self.img_db[self.img_db.name == icos_site]
        pblh = [0]*len(img_db_site)
        for i,idate in enumerate(img_db_site.startdate.dt.round('1h')):
            ts = nearest(era5_times, idate)
            # in case of nan values, cdsapi will download additional expver version
            if 'expver' in era5.dims:
                v = era5.coords['expver'].values # use last version (!= 1)
                val = era5.blh.sel(time=ts, expver=v[0]).item()
                if np.isnan(val):
                    val = era5.blh.sel(time=ts, expver=v[1]).item()
            else:
                val = era5.blh.sel(time=ts).item()
            pblh[i] = val
        if np.isnan(pblh).any():
            raise ValueError('NaN value detected in ERA5 derived PBLH values.')
        return pblh

    def update_img_db(self, img_csv, era_blh = [], nc_name = None, save = False):
        '''
        Update 'img_db' data frame after manual modifications have been made,
        e.g. to remove unsuitable imagery after using 'hsi_qc' method.
        
        Args:
            img_csv (string): Name of the metadata CSV that contains key info
                about the imagery to be processed. Required columns are:
                'name': Name or abbr. of the corresp. ICOS site
                'startdate': Timestamp of image acquisition (YYYY-MM-DD hh:mm:ss)
                'dataTakeID': datatake ID of the scene
                'clouds': commentary field to mark missing data, etc.
                'era_pblh': planetary boundary layer height value from ERA5 re-
                    analysis. Only used when PBLH is missing from ICOS L2 data.
                'usable': integer values for results of imagery inspection.
                    0 = unusable, 1 = usable, 2 = maybe usable (requires FFP
                    inspection), 3 = missing flux data
            era_blh (list of strings, optional): ICOS abbreviations of sites
                which do not provide boundary layer height (BLH) values for FP
                estimation. BLH values will be imported from ERA5 if available.
            nc_name (string, optional): Name of the NetCDF file that stores
                ERA5 PBLH time series. The file must only contain one pixel.
                If None, default ICOS file name structure is used.
            save (bool, optional): If true, the cleaned imagery data frame is
                saved as CSV.
        '''
        old_img_db = pd.read_csv(
            self.icos_dir.parent / img_csv, parse_dates=['startdate'],
            date_parser=self._dp, dtype={'dataTakeID': str}).reset_index(drop=True)
        
        if all(x in old_img_db.columns for x in \
               ['name', 'startdate', 'dataTakeID', 'clouds', 'usable']):
            pass
        else:
            raise ValueError('Missing columns in "img_db". Please prepare the '\
                             'data frame according to documentation.')
        ppi_df = self.img_db[['name', 'dataTakeID', 'ppi_file']]
        checkdate = old_img_db[['name', 'startdate']].select_dtypes(include=[np.datetime64])
        if checkdate.shape[1] == 0:
            raise ValueError('Incorrect datetime format, should be np.datetime64.')
        
        old_img_db['date'] = [dt.datetime.date(x).isoformat() for x in old_img_db['startdate']]
        old_img_db['icostime'] = (old_img_db.startdate +
                                   pd.Timedelta(hours=1)).dt.round('30min').dt.time
        # only keep usable & maybe usable imagery entries
        self.img_db = old_img_db[old_img_db.usable.between(1,2)].reset_index(drop=True)
        
        self._era_blh = era_blh
        if (isinstance(era_blh, list) and all(isinstance(x, str) for x in era_blh)) or era_blh is None:
            pass
        else:
            raise ValueError('Incorrect era_blh specification. Variable must be None or list of strings.')
        
        # add PBLH values from ERA5 data
        self.img_db['era_pblh'] = 0
        for site in self._era_blh:
            pblh_values = self._icos_era5_pblh(site, nc_name = nc_name)
            self.img_db.loc[self.img_db.name == site, 'era_pblh'] = pblh_values
            
        self.img_db = pd.merge(self.img_db, ppi_df, how='left',
                               on=['name', 'dataTakeID'], suffixes=('', '_y'))
        self.img_db.drop(self.img_db.filter(regex='_y$').columns, axis=1, inplace=True)
        
        if save is True:
            self.img_db.to_csv(self.icos_dir.parent / (img_csv[:-4] + '_f.csv'), index=False)
        icos_list = self.img_db.name.unique().tolist()
        
        return icos_list

### ICOS FUNCTIONS ############################################################

    def _ppfd_check(self, site_list):
        '''
        A helper function to find out which ICOS station data does not include
        photosynthetic photon flux density (PPFD_IN), or neither PPFD or incom-
        ing shortwave radiation (SW_IN_F). If PPFD is not available, it is es-
        timated from SW. If both are missing, the site is removed from further
        calculations.
        '''
        no_ppfd = []
        no_rad = []
        cols = ['TIMESTAMP_START', 'TIMESTAMP_END', 'PPFD_IN', 'SW_IN_F']
        for i, site in enumerate(site_list):
            icos_exp = self.icos_dir / ('ICOSETC_' + site + '_FLUXNET_HH_INTERIM_L2.csv')
            if icos_exp.exists(): # Create file name template for single ICOS site
                icos_v = 'ICOSETC_' + site + '_{}_INTERIM_L2.csv'
            else:
                icos_v = 'ICOSETC_' + site + '_{}_L2.csv'
                
            flx_gpp = pd.read_csv(self.icos_dir / icos_v.format('FLUXNET_HH'),
                                 date_parser=self._dp_icos, nrows=0,
                                 parse_dates=cols[:2],
                                 usecols=lambda x: x in cols)
            tvar = ['PPFD_IN', 'SW_IN_F']
            if ~pd.Series(tvar[0]).isin(flx_gpp.columns).all(): # if PPFD not in columns
                no_ppfd.append(site)
            elif ~pd.Series(tvar).isin(flx_gpp.columns).all(): # if neither variable is in columns
                no_rad.append(site)
        if len(no_ppfd) > 0:
            logger.info('ICOS site(s) {} dont have PPFD '.format(no_ppfd) +
                        'measurements in their FLUXNET dataset. PAR will ' +
                        'be calculated from SW_IN.')
        if len(no_rad) > 0:
            logger.warning('ICOS site(s) {} have neither '.format(no_ppfd) +
                           'PPFD nor SW_IN data and will be removed.')
        return no_ppfd, no_rad


    def _icos_height_aux(self, flx_anc, icos_site, D_avg = False):
        '''
        Helper function that cycles through the different variants of obtaining
        vegetation height information from ICOS L2 data: By default, all obser-
        vations of the 75th percentile will be used. If not available, mean
        values or single observations will be used or averaged if observations
        are only available per species (HEIGHTC_SPP) and not for the whole
        canopy.
        '''
        flx_anc_w = flx_anc[flx_anc.VARIABLE_GROUP == 'GRP_HEIGHTC'] \
            .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
        if flx_anc_w.HEIGHTC_STATISTIC.isin(['75th Percentile']).any() \
            & ~flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            D_gid0 = flx_anc_w.loc[
                (flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile'), :].index
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['75th Percentile']).any() \
            & flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            if flx_anc_w.HEIGHTC_SPP.isnull().any():
                D_gid0 = flx_anc_w.loc[
                    (flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile') \
                    & flx_anc_w.HEIGHTC_SPP.isnull(), :].index
        # DE-Hai only has SPP entries from 1 date -> shortcut to D
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']).any() \
            & ~flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            D_gid0 = flx_anc_w.loc[flx_anc_w.HEIGHTC_STATISTIC.isin(
                ['Mean', 'Single observation']), :].index
            logger.info('Using mean instead of P75 values for site {}'.format(icos_site))
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']).any() \
            & flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            if flx_anc_w.HEIGHTC_SPP.isnull().any():
                D_gid0 = flx_anc_w.loc[
                    flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']) \
                    & flx_anc_w.HEIGHTC_SPP.isnull() , :].index
                logger.info('Using mean instead of P75 values for site {}'.format(icos_site))
            else:
                D_gid0 = flx_anc_w.loc[(flx_anc_w.HEIGHTC_STATISTIC == 'Mean') \
                                       & flx_anc_w.HEIGHTC_SPP.notnull() , :].index
                D_avg = True
                logger.info('Using SPP means of canopy height for site {}'\
                            .format(icos_site))
        else:
            D_gid0 = []
        if len(D_gid0) == 0:
            raise KeyError('No usable mean or P75 values for D found in ICOS data.')
        else:
            D_hdates = pd.to_datetime(
                flx_anc_w.loc[flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE']).dt.date
            
        return flx_anc_w, D_hdates, D_avg
    

    def _load_icos_subset(self, icos_site, datelist, fluxvars, aggregate = 'na',
                          ppfd_missing = [], zonal = False):
        '''
        Reads a number of ICOS level 2 CSV files (FLUXES, FLUXNET_HH, SITEINFO,
        ANCILLARY, INST, if applicable METEO) to extract information on meteo-
        rological variables required for flux footprint modeling (WS, WD, PBLH,
        MO_LENGTH, V_SIGMA, USTAR) and ecosystem productivity estimates. Data
        are reduced to the time of interest, i.e. the days for which hyperspec-
        tral imagery are available.
        If zonal statistics were selected, only productivity estimates will be
        exported from L2 data.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            datelist (pandas.Series): Dates of available imagery for the
                selected ICOS site.
            fluxvars (list of strings): ICOS variable names of ecosystem fluxes
                (CO2, radiation, ...) to be imported.
            aggregate (string): Type of aggregation of daily ICOS values, either
                "na", "mean" or "sum".
            ppfd_missing (list of strings, optional): ICOS sites for which PPFD
                variable is missing and will be derived from SW_IN instead.
            zonal (bool, optional): If true, no footprints will be calculated
                and the import of ICOS data will be restricted to fluxes only
                (no meteorological variables required).
        '''
        timelist = self.img_db.loc[datelist.index, 'icostime']
        datetimes = pd.to_datetime(datelist.astype(str) + " " + timelist.astype(str))
        D_avg = False
        D_default = False
        ZM = [-9999]*len(datelist)
        missd = [pd.Timestamp('19000101 00:00:00')] # return dummy
        
        dc = ['TIMESTAMP_START', 'TIMESTAMP_END']
        cols_gpp = dc + fluxvars
        icos_exp = self.icos_dir / ('ICOSETC_' + icos_site + '_FLUXES_INTERIM_L2.csv')
        if icos_exp.exists(): # Create file name template for single ICOS site
            icos_v = 'ICOSETC_' + icos_site + '_{}_INTERIM_L2.csv'
        else:
            icos_v = 'ICOSETC_' + icos_site + '_{}_L2.csv'
            
        # Import GPP percentiles from ensemble (daytime & nighttime)
        flx_gpp = pd.read_csv(self.icos_dir / icos_v.format('FLUXNET_HH'),
                              date_parser=self._dp_icos,
                              parse_dates=dc, usecols=lambda x: x in cols_gpp)
    
        # Reference date to deal with missing flux data
        # not so elegant... maybe sort unsuitable imagery out before?
        #dl_dummy = datelist.reset_index(drop=True)
        #beforeicosstart = dl_dummy[dl_dummy < flx_gpp['TIMESTAMP_START'].dt.date.iloc[0]].index
        #aftericosend = dl_dummy[flx_gpp['TIMESTAMP_START'].dt.date.iloc[-1] < dl_dummy].index
        #self._no_flx_data_ix = beforeicosstart.union(aftericosend)
        
        # Import site metadata for coordinate & IGBP ecosystem type info
        site_info = pd.read_csv(self.icos_dir / icos_v.format('SITEINFO'),
                                on_bad_lines='skip',
                                usecols=['VARIABLE', 'DATAVALUE'])
        
        # SW_IN is always added to the output data frame. PPFD_IN is converted to
        # PAR if available.
        if icos_site not in ppfd_missing:
            flx_gpp_temp0 = flx_gpp.loc[flx_gpp.TIMESTAMP_END.dt.date.isin(datelist),
                                       cols_gpp[1:]]
            flx_gpp_temp0['PAR'] = flx_gpp_temp0.PPFD_IN / 4.57
            flx_gpp_temp0.drop('PPFD_IN', axis=1, inplace=True)
            fluxvars = [x for x in fluxvars if x != 'PPFD_IN'] + ['PAR']
        else: # If PPFD not available, it's estimated from SW_IN instead.
            fluxvars = [x for x in fluxvars if x != 'PPFD_IN']
            flx_gpp_temp0 = flx_gpp.loc[flx_gpp.TIMESTAMP_END.dt.date.isin(datelist),
                                       ['TIMESTAMP_END'] + fluxvars]
            flx_gpp_temp0['PAR'] = flx_gpp_temp0.SW_IN_F * 0.47 # average PAR/SI ratio
            fluxvars = fluxvars + ['PAR']
        # In any case, output DF will contain 'SW_IN_F' & 'PAR' but not 'PPFD_IN'
        flx_gpp_temp0.loc[flx_gpp_temp0.PAR < -100, 'PAR'] = -9999
        # to exclude invalid values from potential aggregation
        exclude_rows = flx_gpp_temp0[(flx_gpp_temp0.loc[:,fluxvars[:9]] == -9999).any(axis=1)].index
        # 'time's in timelist are already tz:CET! -> reduce to 1 row per HSI
        flx_gpp_temp = flx_gpp_temp0.loc[flx_gpp_temp0.TIMESTAMP_END.isin(datetimes), :]
        
        # only certain flux columns (fluxvars[:9]) are aggregated, QC columns and radiation fluxes are excluded (unaggregated PAR required for UPW transformation)
        if aggregate == 'mean': # for mean, night values have to be dropped for GPP
            flx_gpp_aggr = flx_gpp_temp0.drop(exclude_rows, axis=0)\
                .loc[(flx_gpp_temp0.PAR > 0) | (flx_gpp_temp0.PAR == -9999), # keep PAR NA values while filtering out night time values
                     fluxvars[:9]].groupby(
                         flx_gpp_temp0.TIMESTAMP_END.dt.date, sort=False).agg(
                             {var: 'mean' for var in fluxvars[:9]})
            fluxvars[:9] = ['{}_mean'.format(x) for x in fluxvars[:9]]
        elif aggregate == 'sum':
            flx_gpp_aggr = flx_gpp_temp0.drop(exclude_rows, axis=0)\
                .loc[:, fluxvars[:9]].groupby(
                    flx_gpp_temp0.TIMESTAMP_END.dt.date, sort=False).agg(
                        {var: 'sum' for var in fluxvars[:9]})
            fluxvars[:9] = ['{}_sum'.format(x) for x in fluxvars[:9]]

        if aggregate != 'na':
            flx_gpp_aggr.columns = fluxvars[:9]
            # join aggregated fluxes with unaggregated columns and add timestamp
            flx_gpp_aggr = pd.merge(flx_gpp_aggr, flx_gpp_temp[fluxvars[9:]], how='left',
                                    left_index=True, right_on=flx_gpp_temp.TIMESTAMP_END.dt.date)
            flx_gpp_aggr.insert(0, 'TIMESTAMP_END', flx_gpp_temp.TIMESTAMP_END.values)
            flx_gpp_temp = flx_gpp_aggr.drop('key_0', axis=1)
        # add missing dates back in
        if len(datelist) > flx_gpp_temp.shape[0]:
            ds1 = set(datetimes.tolist())
            ds2 = set(flx_gpp_temp.TIMESTAMP_END)
            missd = list(ds1.difference(ds2))
            add = pd.DataFrame(np.full((len(missd), len(fluxvars)+1), -9999))
            add.columns = [dc[1]] + fluxvars
            add.iloc[:, 0] = missd
            flx_gpp_temp = pd.concat([flx_gpp_temp, add], axis=0)
            flx_gpp_temp.sort_values(by=dc[1], inplace=True)
            logger.info('One or more datelist entries are excluded from the aggregated GPP data frame.')   
            print('One or more datelist entries are excluded from the aggregated GPP data frame.')
            
        if zonal: # Only insert TS and return before FFP calc.
            icos_subset = flx_gpp_temp.reset_index(drop=True)
            return icos_subset, ZM, fluxvars, missd
        
        cols_flx = ['USTAR', 'V_SIGMA', 'MO_LENGTH', 'PBLH', 'WS', 'WD']
        cols = dc + cols_flx
        # Import micrometeo. data and NEE
        flx = pd.read_csv(self.icos_dir / icos_v.format('FLUXES'),
                          date_parser=self._dp_icos,
                          parse_dates=dc, usecols=lambda x: x in cols)
        
        # Import 75th percentile of canopy height
        flx_anc = pd.read_csv(self.icos_dir / icos_v.format('ANCILLARY'),
                              encoding = 'unicode_escape')
        if site_info.loc[site_info.VARIABLE == 'IGBP', 'DATAVALUE'] \
            .item() in ['DBF', 'ENF', 'EBF', 'MF']:
            if flx_anc.VARIABLE_GROUP.isin(['GRP_TREE_HEIGHT']).any():
                flx_anc_w = flx_anc[flx_anc.VARIABLE_GROUP == 'GRP_TREE_HEIGHT'] \
                    .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
                # Serach for 75th perc. where HEIGHTC_SPP is NaN (to get value for whole stand)
                D_gid0 = flx_anc_w.loc[(flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile')
                                       & flx_anc_w.HEIGHTC_SPP.isnull() , :].index
                try:
                    D_hdates = pd.to_datetime(flx_anc_w.loc[
                        flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE_START']).dt.date
                except KeyError:
                    D_hdates = pd.to_datetime(flx_anc_w.loc[
                        flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE']).dt.date

            elif flx_anc.VARIABLE_GROUP.isin(['GRP_HEIGHTC']).any(): # Sonderfall DE-Hai
                flx_anc_w, D_hdates, D_avg = self._icos_height_aux(flx_anc, icos_site)
            else:
                raise KeyError('Ancillary file for ICOS site {} '.format(icos_site) +
                               'holds no data about vegetation height.')
        
        elif site_info.loc[site_info.VARIABLE == 'IGBP', 'DATAVALUE'] \
            .item() in ['CRO', 'GRA', 'CSH', 'SAV']:
            flx_anc_w, D_hdates, D_avg = self._icos_height_aux(flx_anc, icos_site)
            
        elif site_info.loc[site_info.VARIABLE == 'IGBP', 'DATAVALUE'].item() == 'WET':
            D_default = True
            logger.info('Using default value for canopy height for wetland' +
                        ' site {} (D = 0.1m)'.format(icos_site))
            D = 0.1 # default low vegetation height for bogs as no height is reported
        # TODO: Throw out IT-LSN? Single OSH site = vineyard, special case, no info about veg. height.
        elif site_info.loc[site_info.VARIABLE == 'IGBP', 'DATAVALUE'].item() == 'OSH':
            D_default = True
            logger.info('Using default value for canopy height for open' +
                        ' shrubland site {} (D = 1m)'.format(icos_site))
            D = 1 # default low vegetation height for bogs as no height is reported
        if D_avg == True:
            D = flx_anc_w.loc[D_hdates.index, 'HEIGHTC'].astype(float).mean()
            
        # Import sensor height
        flx_ins = pd.read_csv(self.icos_dir / icos_v.format('INST'), on_bad_lines='skip')
            
        flx_ins_w = flx_ins[flx_ins.VARIABLE_GROUP == 'GRP_INSTOM'] \
            .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
    
        # Filter for EC sensor installation IDs and extract dates
        if len(flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains('LI-COR'), 'INSTOM_HEIGHT']) > 0:
            senstr = 'LI-COR'
        elif len(flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains('Campbell EC'), 'INSTOM_HEIGHT']) > 0:
            senstr = 'Campbell EC'
        else:
            raise KeyError('Unknown EC sensor model. Please check your ICOS ' +
                           'instrument metadata.')
        Z_gid0 = flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains(senstr) & \
                               (flx_ins_w.INSTOM_TYPE == 'Installation') , :].index        
        Z_hdates0 = pd.to_datetime(flx_ins_w.loc[flx_ins_w.index.isin(Z_gid0), 'INSTOM_DATE']).dt.date
        # Swap index and gid values and truncate installation dates after image acq. date.
        Z_hdates = pd.Series(Z_hdates0.index.values, index=Z_hdates0).sort_index() # truncate requires sorted index
        # now I want to compare the entries to get height meas. from a date closest
        # to image acq. date (for canopy height) or the last entry before acq.
        # date (for sensor height).    
        for i, idate in enumerate(datelist):
            if idate in [x.date() for x in missd]:
                continue
            else:
                if (D_avg == True) | (D_default == True):
                    pass
                else:
                    D_gid = D_hdates[D_hdates == nearest(D_hdates.values, idate)].index[-1] # Find nearest date and in case of multiple entries, choose last
                    D = flx_anc_w.loc[flx_anc_w.index == D_gid, 'HEIGHTC'].astype(float).item()
                try:
                    Z_gid = Z_hdates.truncate(after=idate)[-1] # Use last valid installation date.
                except IndexError: # if image date is before official installation date, use first installation date
                    Z_gid = Z_hdates[0]
                Z = flx_ins_w.loc[flx_ins_w.index == Z_gid, 'INSTOM_HEIGHT'].astype(float).item()
                if D > Z:
                    raise ValueError('Canopy height D > sensor height Z. Please check your ICOS ancillary metadata.')
                ZM[i] = Z-D
        
        # Reduce DF to dates of interest (all records for days of interest) and concatenate
        if icos_site in self._era_blh:
            flx0 = flx.loc[flx.TIMESTAMP_END.dt.date.isin(datelist),
                           ['TIMESTAMP_END'] + cols_flx[:3]]
            flx_met0 = pd.read_csv(self.icos_dir / icos_v.format('METEO'),
                                   date_parser=self._dp_icos, parse_dates=dc)
            flx_met = flx_met0.loc[flx_met0.TIMESTAMP_END.dt.date.isin(datelist),
                                   cols_flx[4:]]
            flx_temp = pd.concat([flx0.reset_index(drop=True),
                                  flx_met.reset_index(drop=True)], axis=1)
        else:
            flx_temp = flx.loc[flx.TIMESTAMP_END.dt.date.isin(datelist),
                               cols[1:]]
        # flx_temp is not aggregated as micromet. data is only required for FFP modeling
        icos_subset = pd.merge(flx_gpp_temp.reset_index(drop=True),
                               flx_temp.reset_index(drop=True),
                               how='left', on='TIMESTAMP_END')
        icos_subset.replace(np.nan, -9999, inplace=True)
        # Final ICOS subset only contains 1 row per HSI
        return icos_subset, ZM, fluxvars, missd


### FFP & CROPPING FUNCTIONS ##################################################
    
    def _model_geoms(self, icos_site, datelist, icos_subset, ZM, fluxvars, missd,
                     zonal = False):
        '''
        Calculates flux footprint geometries matching the timestamps of hyper-
        spectral imagery in 'img_db'. Uses the external 'FFP' function from
        the flux footprint modeling module from Kljun et al.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            datelist (pandas.Series): Dates of available imagery for the
                selected ICOS site.
            icos_subset (pandas.DataFrame): Subset of ICOS L2 data cut to the
                dates of interest. Contains fluxes and if necessary micromete-
                orological variables for FFP modeling.
            ZM (list): Differences between observation & vegetation height.
            fluxvars (list of strings): ICOS variable names of ecosystem fluxes
                (CO2, radiation, ...) to be imported.
            missd (list of pd.Timestamps): dates from datelist for which ICOS
                data is completely missing.
            zonal (bool, optional): If true, zonal statistics (buffer value
                depending on ecosystem) will be calculated instead of FFPs.
        '''
        timelist = self.img_db.loc[datelist.index, 'icostime']
        dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
        ffp_params = [0]*len(datelist)
        
        # Project to Lambert Azimuthal Equal Area (LAEA) for collection of european geometries in a single gdf
        crs_lam = proj.CRS.from_epsg('3035')
        flx_loc = self.flx_loc.loc[self.flx_loc.name == icos_site,
                                   'geometry'].to_crs(crs_lam)
        
        if zonal:
            ecosystem = self.icos_db.loc[self.icos_db.name == icos_site, 'ecosystem'].item()
            if ecosystem in ['MF', 'DBF', 'EBF', 'ENF']:
                zr = 80
            elif ecosystem in ['GRA', 'OSH']:
                zr = 50
            elif ecosystem == 'WET':
                zr = 40
            elif ecosystem == 'CRO':
                zr = 30
            else:
                raise ValueError('Ecosystem {} not supported '.format(ecosystem) +
                                 'Please update codebase.')
            geoms = [flx_loc.buffer(zr).item()] * len(timelist)
        else:
            geoms = [0]*len(datelist)
        
        #transformer = proj.Transformer.from_crs(4326, crs_utm, always_xy=True)
        #x, y = transformer.transform(lon, lat)
        for i,date in enumerate(datelist):
            #print('Modeling FFP for {}: {}'.format(icos_site, dtakes.iloc[i]))
            ffp_params[i] = icos_subset.loc[
                icos_subset.TIMESTAMP_END.dt.date == date, :].squeeze(axis=0) # squeeze creates series instead of 1D DF
            if zonal:
                continue
            if icos_site in self._era_blh:
                pblh = pd.Series([self.img_db.loc[datelist.index[i],
                                              'era_pblh']], index=['PBLH'])
                ffp_params[i] = ffp_params[i].append(pblh)
            # In case of Nodata values in ICOS data, no FP can be calculated
            if ffp_params[i][['WS', 'PBLH', 'MO_LENGTH', 'V_SIGMA',
                             'USTAR', 'WD']].isin([-9999]).any() | (date in [x.date() for x in missd]):
                geoms[i] = None
                self.img_db.loc[self.img_db.dataTakeID == dtakes.iloc[i],
                                'clouds'] = 'icos_na'
                logger.warning('{}: No ICOS data available for {} image {} from {}.'\
                               .format(icos_site, self.sensor, dtakes.iloc[i],
                                       datelist.iloc[i]))
                ffp_params[i] = pd.Series(data=[0]*len(fluxvars), index=fluxvars)
            # Estimate 80th (and 50th) percentile of flux footprint with Kljun model
            else:
                flx_fp = ffp.FFP(zm=ZM[i], umean=ffp_params[i].WS, h=ffp_params[i].PBLH,
                                 ol=ffp_params[i].MO_LENGTH, sigmav=ffp_params[i].V_SIGMA,
                                 ustar=ffp_params[i].USTAR, wind_dir=ffp_params[i].WD,
                                 rs=[50., 80.], fig=0)
            
                # Compute FP coordinates & create geometry. Uses 80% FP coordinates (index 1)
                xs = [round(xoff + flx_loc.x.item(), 2) for xoff in flx_fp['xr'][1]]
                ys = [round(yoff + flx_loc.y.item(), 2) for yoff in flx_fp['yr'][1]]
                geoms[i] = Polygon(zip(xs, ys))
        
        # Prepare DF for merging with geoinformation.
        img_df = self.img_db.loc[datelist.index,
                                 ['name', 'startdate', 'date', 'dataTakeID',
                                  'clouds', 'ppi_file']].reset_index(drop=True)
        # Create single geopandas object holding information about geometry and observations
        geom_gdf = gpd.GeoDataFrame(data=img_df, geometry=gpd.GeoSeries(
            geoms, crs=crs_lam))
        # Add data about GPP, NEE
        flx_list = [0]*len(timelist)
        for i,pset in enumerate(ffp_params):
            flx_list[i] = pset.loc[fluxvars]

        flx_df = pd.concat(flx_list, axis=1).T.reset_index(drop=True)
        flx_geom_gdf = pd.concat([geom_gdf, flx_df], axis=1)
        
        ix = flx_geom_gdf.columns.get_loc(fluxvars[0]) # get index of first prod variable
        flx_geom_gdf.iloc[:, ix:ix+len(fluxvars)] = flx_geom_gdf.iloc[
            :, ix:ix+len(fluxvars)].apply(pd.to_numeric, errors='ignore')
        
        # Notify in case of lower quality values in important variables
        for var in ['NEE_VUT_50_QC', 'SW_IN_F_QC']:
            quality = 'medium'
            for flag in [2, 3]:
                if (flx_geom_gdf[var] == flag).any():
                    if flag == 3:
                        quality = 'poor'
                    ix = flx_geom_gdf[flx_geom_gdf[var] == flag].index
                    if len(ix) == 1:
                        ix = ix.item()
                    else:
                        ix = ix.tolist()
                    logger.warning('{}: {} data quality for {} dataTakeID(s) {} '\
                                   .format(icos_site, var, self.sensor,
                                           dtakes.iloc[ix]) + 'is {} ({})'\
                                       .format(quality, flag))
        
        return flx_geom_gdf
    

    def _crop_hsi_2_geoms(self, icos_site, flx_geom_gdf, sr = 'vnir',
                          upw = False, save_plot = False):
        '''
        Crops hyperspectral imagery to FFP geometries and averages resulting
        pixels per band. Cropped imagery can be saved as GeoTIFF for later use.
        Additionally, RGB representations for overviews of the footprint geo-
        metries and surrounding areas can be plotted.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            ffp_gdf (geopandas.GeoDataFrame): Flux measurements that match the
                observation times of the hyperspectral imagery including geo-
                metries for cropping (zonal or FFP).
            ffp_params (list of pandas.Series): Micrometeorological variables
                used in FFP modeling. Only used for NA value checks.
            sr (string): Abbreviation of the spectral range within which
                HSI will be used, either 'vis', 'vnir' or 'vswir' (the latter
                is only available for PRISMA data).
            upw (bool, optional): If true, reflectance values will be cropped
                to visible light range and multiplied with PAR from ICOS sensor
                data resulting in 400-700 nm upwelling radiation (UPW).
            save_plot (bool, optional): If true, ICOS site surroundings will be
                plotted with the zonal/FFP geometry superimposed and saved.
        '''
        # this datelist is updated after step 2 of the cropping procedure
        datelist = flx_geom_gdf.loc[flx_geom_gdf.name == icos_site, 'startdate'].dt.date
        dtakes = flx_geom_gdf.loc[datelist.index, 'dataTakeID']
        datetimes = flx_geom_gdf.loc[datelist.index, 'startdate'].dt.strftime('%Y-%m-%d %H:%M')
        
        img_paths = [list((self.img_dir).glob('*{}_{}_6km_crop.tif'.format(dt, icos_site))) for dt in dtakes]
        
        #with rio.open(crop_paths[0]) as src:
        #    test = src.read(np.linspace(1, 63, 63))
        
        nm_min = 400
        if sr == 'vis':
            nm_max = 700
        elif sr == 'vnir':
            nm_max = 1000
        elif sr == 'vswir':
            nm_max = 2500
        
        epsg = self.flx_loc.loc[self.flx_loc.name == icos_site, 'sensorcrs'].item()
        crs_utm = proj.CRS.from_epsg(epsg)
        # Local UTM coordinates used for cropping, modeled geoms are in LAEA
        flx_loc = self.flx_loc.loc[self.flx_loc.name == icos_site,
                                   'geometry'].to_crs(crs_utm)
        flx_roi_plot = flx_loc.buffer(500)
        box_geom_plot = geometry.box(*flx_roi_plot.total_bounds)
        
        crs_lam = proj.CRS.from_epsg('3035')
        transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
        
        cubes = [0]*len(datelist)
        wlss = [0]*len(datelist)
        itrans = [0]*len(datelist)
    
        for i,path in enumerate(img_paths):
            # imagery must exist and ID must be unique
            if len(path) == 0: # check if file is present
                raise ValueError('The cropped {} image with dataTakeID {} could not be found.'\
                                 .format(self.sensor, dtakes.iloc[i]))
            elif len(path) > 1:
                raise ValueError('dataTakeID {} not unique. Please investigate.'\
                                 .format(dtakes.iloc[i]))
            with rio.open(path[0]) as src:
                cubes[i] = src.read()
                wlss[i] = [float(w) for w in list(src.descriptions)]
                itrans[i] = src.meta['transform'] # for plotting & cropping
                na_val = src.nodata
        cubes = [np.einsum('kli->lik', c) for c in cubes]
        
        wl_min = _fnv(wlss[0], nm_min)
        wl_max = _fnv(wlss[0], nm_max) + 1
        nbands = len(wlss[0][wl_min:wl_max])
        logger.debug('initial band length: {}, sr_min + index: {}-{}, sr_max + index: {}-{}, reduced band length: {}'\
                     .format(len(wlss[0]), nm_min, wl_min, nm_max, wl_max, nbands))

        cube_list = [0]*len(datelist)
        geom_px_avgs = [0]*len(datelist)
        
        for i,path in enumerate(img_paths):
            # NA handling
            if flx_geom_gdf.loc[i, 'clouds'] == 'icos_na'\
                or ((upw is True) & (flx_geom_gdf.loc[i, 'PAR'] == -9999)):
                cube_list[i] = -9999
                geom_px_avgs[i] = -9999
                geom_cube = np.array([-9999])
            else:
                #ffp_poly = ffp_gdf[['geometry']].values[i].flatten()[0]
                lam_poly = flx_geom_gdf.loc[i, 'geometry']
                utm_poly = stransform(transf.transform, lam_poly)
                geom_cube, otrans = _local_mask(cubes[i], itrans[i], [utm_poly],
                                                crop=True, all_touched=True)
                '''imagery is cropped 2 times: a 6km diameter subset is used
                for RGB calculation and can be saved. A smaller subset is used
                for footprint value extraction'''

                # TODO: values>1 should also be removed (total amount of pixels is very small)
                #describe(temp, axis=None)

                if (len(geom_cube[geom_cube == na_val]) / geom_cube.size) > 0.05:
                    logger.info('{}_{}: {}% of regular pixels have NaN values.'\
                                .format(icos_site, dtakes.iloc[i], round(
                                    (len(geom_cube[geom_cube == na_val]) / geom_cube.size)*100, 2)))
                geom_cube[geom_cube == na_val] = np.nan
                if (len(geom_cube[geom_cube < 0]) / geom_cube.size) > 0.05:
                    logger.info('{}_{}: {}% of regular pixels have < 0 values. Negative reflectance values are converted to NaN.'\
                                .format(icos_site, dtakes.iloc[i], round(
                                    (len(geom_cube[geom_cube < 0]) / geom_cube.size)*100, 2)))
                geom_cube[geom_cube < 0] = np.nan
                
                # Crop dimensions according to chosen spectral range
                
                geom_cube = geom_cube[:, :, wl_min:wl_max]
                '''ICOS PPFD 400-700 nm: http://archive.sciendo.com/INTAG/intag.2017.32.issue-4/intag-2017-0049/intag-2017-0049.pdf
                1 W/m2 ~ 4.57 micromole/m2/s (in Foken 2008 "Micrometeorology" recherchieren)
                flx_geom_gdf contains the variables 'PPFD_IN', 'SW_IN_F' for UPW calculations
                SW_IN can be used to calculate PAR when missing: https://doi.org/10.2134/agronj1984.00021962007600060018x
                PAR/SW_IN = 0.47 -> mean from mid-latitude studies referenced in Walker (2005)'''
                if upw is True: # HSI are cropped to 400-700 nm and multiplied with PAR
                    par = flx_geom_gdf.loc[i, 'PAR']
                    geom_cube = geom_cube * par
                
                geom_px_avgs[i] = np.nanmean(geom_cube, axis=(1, 0))
                if np.isnan(geom_px_avgs[i]).all():
                    flx_geom_gdf.loc[i, 'clouds'] = 'hsi_na'
                    # also update in img_db for consistency
                    self.img_db.loc[self.img_db.dataTakeID == dtakes.iloc[i],
                                    'clouds'] = 'hsi_na'
                    logger.warning('{}: Only NA pixels in FP area of image {} from {}.'\
                                   .format(icos_site, dtakes.iloc[i], datelist.iloc[i]))

                if len(datelist) == 1:
                    cube_list = geom_cube
                else:
                    cube_list[i] = geom_cube
                
                if not save_plot:
                    continue # no plotting since zonal stats are set constant for each ICOS station
                rows, cols, b = np.shape(cubes[i])
                cubeT = cubes[i].reshape(-1,b)
                cube_rgb = HSI2RGB(wlss[i], cubeT, rows, cols, 50, 0.0002)
                # clip RGB to plotting extent
                cube_rgb_plot, otrans = _local_mask(
                    cube_rgb, itrans[i], [box_geom_plot], crop=True, indexes=[1,2,3])
                out_ext = riop.plotting_extent(cube_rgb_plot, otrans)
                fig, ax = plt.subplots(1,1, figsize=(12,12))
                ax.imshow(cube_rgb_plot, extent=out_ext)
                ax.title.set_text('{} : 80% footprint estimate, {}, DT {}'\
                                  .format(icos_site, datetimes.iloc[i], dtakes.iloc[i]))
                flx_geom_gdf[flx_geom_gdf.index == i].to_crs(crs_utm).plot(
                    ax=ax, ec='crimson', fc='none', zorder=10, linewidth=2)
                ax.scatter(flx_loc.x.item(), flx_loc.y.item(), marker='x',
                           s=80, c='crimson', label='EC tower')
                ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                ytx = ax.get_yticks()
                xtx = ax.get_xticks()
                ax.set_yticks([ytx[0], ytx[-1]], visible=True, rotation='vertical')
                ax.set_xticks([xtx[0], xtx[-1]], visible=True, rotation='horizontal')
                if (np.isnan(geom_cube).all()):
                    ax.text(0.4, 0.5, 'NA pixel', size='30', c='w', transform=ax.transAxes)
                fig.savefig(self.img_dir / '{}rgb_{}.png'.format(img_paths[i][0].stem[:-8], self.ptype['mask']),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
        # np arrays cannot be saved as a GPKG element. Therefore 1 column in GDF for each band.
        sr_band_ix = ['b' + str(x).zfill(3) for x in range(1, nbands + 1)]
        geom_avg_df = pd.DataFrame([
            pd.Series(vals, index=sr_band_ix) for vals in geom_px_avgs])
        logger.debug('single geom_px_avgs: {}, band_cols in geom_avg_df: {}'\
                     .format(geom_px_avgs[-1], sr_band_ix))
        logger.debug('cube shape after sr adjustment: {}'.format(np.shape(geom_cube)))
        flx_hsi_gdf = pd.concat([flx_geom_gdf, geom_avg_df], axis=1)
        
        return flx_hsi_gdf, cube_list

    def hsi_geom_crop(self, icos_list, date = None, sr = 'vnir', zonal = False,
                      upw = False, aggr = 'na', save = False, save_plot = False):
        '''
        Crops hyperspectral imagery to flux footprints derived from the 30-min
        interval of EC measurements at ICOS flux towers during or before the DESIS
        data take.
        Footprints are calculated using level 2 data of ICOS class 1 & 2 stations
        using the flux footprint model by Kljun et al. (2015) [1].
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which hyperspectral imagery will be evaluated.
            date (string, optional): A specific date (YYYY-MM-DD) can be passed to
                process a single image.
            sr (string): Abbreviation of the spectral range within which
                HSI will be used, either 'vis', 'vnir' or 'vswir' (the latter
                is only available for PRISMA data).
            zonal (bool, optional): If true, zonal statistics (buffer value
                depending on ecosystem) will be calculated instead of FFPs.
            upw (bool, optional): If true, reflectance values will be multi-
                plied with PAR from ICOS sensor data resulting in upwelling
                radiation (UPW). Only possible for 400-700nm.
            aggr (string): Type of aggregation of daily ICOS values, either
                "na", "mean" or "sum".
            save (bool, optional): If true, footprint geometries are saved as
                GeoPackage.
            save_plot (bool, optional): If true, ICOS site surroundings will be
                mapped with the zonal/FFP geometry superimposed and saved.
        
        [1] Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015: A simple two‐
        dimensional parameterisation for Flux Footprint Prediction (FFP). Geosci.
        Model Dev., 8, 3695‐3713. doi:10.5194/gmd‐8‐3695‐2015.
        '''
        if (upw is True) & (sr != 'vis'):
            logger.info('upw is true but sr is not "vis". Since UPW ' +
                        'calculation is only possible for 400-700nm, ' +
                        'sr will be set to "vis"')
            sr = 'vis'
        if isinstance(icos_list, str):
            icos_list = [icos_list]
                         
        t = 'upw' if upw is True else 'ref' # upwelling radiation or reflectance
        a = 'zst' if zonal is True else 'ffp' # flux footprint
        self.ptype = {'rad': t, 'mask': a, 'sr': sr, 'aggr': aggr} # processing info
        
        # seletion of productivity variables to be extracted (percentiles, partitioning)
        fluxvars0 = ['GPP_DT_VUT_05', 'GPP_DT_VUT_50', 'GPP_DT_VUT_95', 'GPP_NT_VUT_05',
                    'GPP_NT_VUT_50', 'GPP_NT_VUT_95', 'NEE_VUT_05', 'NEE_VUT_50',
                    'NEE_VUT_95', 'NEE_VUT_50_QC', 'PPFD_IN', 'SW_IN_F', 'SW_IN_F_QC']
        # additional data for incoming SW rad.
        no_ppfd, no_rad = self._ppfd_check(icos_list)
        # update list to remove stations without PPFD from UPW calculation
        if len(no_rad) > 0:
            icos_list = [x for x in icos_list if not x in no_rad]
        # TODO: Remove or separate check for NAs in SW_IN_F?
        
        if len(icos_list) > 1:
            flx_hsi_gdf_l = [0]*len(icos_list)
            flx_imgs_l = [0]*len(icos_list)
        
        for i,site in enumerate(pbar := tqdm(icos_list)):
            pbar.set_description('Processing %s' % site)
            if date == None:
                datelist = self.img_db.loc[self.img_db.name == site, 'startdate'].dt.date
            else:
                try:
                    dt.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError('Incorrect date format, should be YYYY-MM-DD')
                datelist = self.img_db.loc[(self.img_db.name == site) & \
                                                 (self.img_db.date == date), 'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            
            if self.sensor == 'DESIS':
                img_paths = [list((self.img_dir).glob('*'+ dt + '*SPECTRAL*.tif'))\
                             for i,dt in enumerate(dtakes)]
                # Only 1 image is used for getting CRS info as all images at one ICOS location share the same CRS!
            elif self.sensor == 'PRISMA':
                img_paths = [list((self.img_dir).glob('*'+ dt + '*.he5'))\
                             for i,dt in enumerate(dtakes)]
        
            ip_check = pd.Series([len(x) > 1 for x in img_paths])
            if ip_check.any():
                raise ValueError('More than 1 matching img for dataTakeID {}'\
                                 .format(dtakes.iloc[ip_check[ip_check == True].index]))
            
            # (1) Load FFP model parameters
            icos_subset, ZM, fluxvars, missd = self._load_icos_subset(
                site, datelist, fluxvars0, aggr, no_ppfd, zonal)
            
            # (2) Footprint modeling and geometry generation
            flx_geom_gdf = self._model_geoms(
                    site, datelist, icos_subset, ZM, fluxvars, missd, zonal)
            logger.info('{}: Footprints for all {} images have been estimated.' \
                        .format(site, self.sensor))
            logger.debug('CRS of gdf after geometry generation (step 2): {}'.format(flx_geom_gdf.crs))

            # (3) Crop of hyperspectral imagery to footprint
            # 'zonal' arg not needed as cropping is identical for FFP & zonal
            flx_hsi_gdf, flx_imgs = self._crop_hsi_2_geoms(
                site, flx_geom_gdf, sr, upw, save_plot)
            logger.debug('CRS of gdf after cropping (step 3): {}'.format(flx_hsi_gdf.crs))
            logger.info('{}: All {} images have been cropped to footprint size.'\
                        .format(site, self.sensor))
            n_na = len(flx_hsi_gdf[flx_hsi_gdf.clouds.isin(
                ['icos_na', 'hsi_na'])])
            if n_na == len(flx_hsi_gdf):
                logger.warning('{}: All imagery acquisition dates '.format(site) +
                               'have no ICOS or hyperspectral data available.' +
                               ' Station will be missing in output DF.')
            if len(icos_list) == 1:
                flx_hsi_gdf = flx_hsi_gdf[~flx_hsi_gdf.clouds.isin(
                    ['icos_na', 'hsi_na'])].reset_index(drop=True)
                return flx_hsi_gdf, flx_imgs
            else:
                flx_hsi_gdf_l[i] = flx_hsi_gdf
                flx_imgs_l[i] = flx_imgs
        flx_imgs_c = [y for x in flx_imgs_l for y in x]
        flx_hsi_gdf_c = pd.concat(flx_hsi_gdf_l).reset_index(drop=True)
        # Sort out imagery outside the ICOS record period and where flux or hyperspectral data have NAs
        n_na = len(flx_hsi_gdf_c[flx_hsi_gdf_c.clouds.isin(
            ['icos_na', 'hsi_na'])])
        logger.info('{} images with missing ICOS or HSI pixel '.format(n_na) +
                    'data were removed from the final geodataframe ' +
                    '("self.flx_hsi_gdf").')
        flx_hsi_gdf = flx_hsi_gdf_c[~flx_hsi_gdf_c.clouds.isin(
            ['icos_na', 'hsi_na'])].reset_index(drop=True)
        
        self.flx_hsi_gdf = flx_hsi_gdf
        self.flx_imgs = flx_imgs_c
        
        if save is True:
            fname = 'all_sites_{}_{}_{}_{}_{}.gpkg'\
                .format(self.sensor, self.ptype['sr'], self.ptype['mask'],
                        self.ptype['rad'], self.ptype['aggr'])
            logger.info('Saving file {}'.format(fname))
            print('Saving file: {}'.format(fname))
            print(flx_hsi_gdf.columns)
            flx_hsi_gdf.to_file(self.out_dir / fname, driver='GPKG')
        no_ppi = len(flx_hsi_gdf[flx_hsi_gdf.ppi_file == ''])
        if no_ppi > 0:
            logger.warning('No S2 PPI images available for {} data takes'.format(no_ppi))
            logger.debug('No PPI data takes:\n{}'.format(flx_hsi_gdf.loc[flx_hsi_gdf.ppi_file == '', 'dataTakeID']))
        return flx_hsi_gdf, flx_imgs_c
    
    
    def load_cropped_db(self, fdir = None, sr = 'vnir', zonal = False,
                        upw = False, aggr = 'na'):
        '''
        Load the processed, cleaned version of the imagery data base which com-
        bines ICOS flux estimates with spectrometer data cropped to flux foot-
        prints.
        
        Args:
            fdir (str, optional): Directory with saved HSICOS data.
            sr (string): The spectral range for which the GeoPackage
                includes HSI data.
            zonal (bool, optional): The GeoPackage is based on zonal statistics
                instead of footprint estimates.
            upw (bool, optional): The GeoPackage contains upwelling radiation
                instead of reflectance values.
            aggr (string, optional): Type of aggregation of daily ICOS values,
                either "na", "mean" or "sum".
        '''
        if (upw is True) & (sr != 'vis'):
            logger.info('upw is true but sr is not "vis". Since UPW ' +
                        'calculation is only possible for 400-700nm, ' +
                        'sr will be set to "vis"')
            sr = 'vis'
        if fdir is None:
            fdir = self.out_dir
        t = 'upw' if upw is True else 'ref' # upwelling radiation or reflectance
        a = 'zst' if zonal is True else 'ffp' # flux footprint
        self.ptype = {'rad': t, 'mask': a, 'sr': sr, 'aggr': aggr} # processing info
        
        fname = 'all_sites_{}_{}_{}_{}_{}.gpkg'.format(self.sensor, sr, a, t, aggr)
        logger.info('Loading file {}'.format(fname))
        print('Loading file: {}'.format(fname))
        self.flx_hsi_gdf = gpd.read_file(
            fdir / fname, driver='GPKG')
        self.flx_hsi_gdf['startdate'] = pd.to_datetime(
            self.flx_hsi_gdf.startdate)
        
        return self.flx_hsi_gdf

            
### INCLUDE COVARIATES ########################################################

    def icos_ppi_get(self, icos_list, dataset = 'ST', day_range = 12,
                     datelist = None, save = True):
        '''
        This function should only be used AFTER HSI have been cropped to GOI
        (i.e. flx_hsi_gdf exists) as a substantial amount of data will have to
        be downloaded to get phenology information (S2 PPI data can only be
        cropped to 10k x 10k raster tiles before downloading). This function
        downloads and matches PPI data with HSI acquisition dates, crops the
        tiles to a bounding box with 500m radius around ICOS sites and saves
        them for following computations.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which hyperspectral imagery will be evaluated.
            dataset (string): Either 'ST' (seasonal trajectories) or 'VI'
                (vegetation indices). Both contain PPI data.
            day_range (int): time window around the HSI dates within which PPI
                files will be searched for.
            datelist (pd.Series): When processing a single station, a series of
                specific timestamps can be supplied for processing.
            save (bool, optional): If true, info about PPI imagery filenames
                matched to HSI is saved in the flux geopackage.
        '''
        
        if isinstance(icos_list, str):
            icos_list = [icos_list]
        
        c = Client(debug=False, quiet=True, progress=False, sleep_max=120)
        if dataset == 'ST':
            ds = 'EO:HRVPP:DAT:SEASONAL-TRAJECTORIES'
        elif dataset == 'VI':
            ds = 'EO:HRVPP:DAT:VEGETATION-INDICES'
        #if 'ppi_file' not in self.flx_hsi_gdf.columns:
        #    self.flx_hsi_gdf['ppi_file'] = '' # empty string instead of NA (np.nan is float)
        
        for site in icos_list:
            if datelist is None:
                datetimes = self.flx_hsi_gdf.loc[(self.flx_hsi_gdf.name == site) &
                                                 (self.flx_hsi_gdf.ppi_file == ''), 'startdate']
                n_hsi = len(self.flx_hsi_gdf[self.flx_hsi_gdf.name == site])
                if len(datetimes) == 0:
                    logger.info('{}: PPI data for all HSI data takes '.format(site) +
                                 'have already been downloaded. Site will be skipped.')
                    continue
                elif len(datetimes) < n_hsi:
                    logger.info('{}: Downloading PPI data for '.format(site) +
                                '{}/{} HSI data takes.'.format(len(datetimes), n_hsi))
                elif len(datetimes) == n_hsi:
                    logger.info('{}: Downloading PPI data for all HSI data takes'.format(site))
            elif isinstance(datelist, pd.Series):
                datetimes = datelist
            else:
                raise ValueError('datelist has to be submitted as pd.Series.')
            ppi_files = [''] * len(datetimes)
            odir = self.img_dir.parent / 'Copernicus/S2PPI'
            odir.mkdir(parents=True, exist_ok=True)
            
            epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            flx_loc = self.flx_loc.loc[self.flx_loc.name == site, 'geometry']
            
            ext = flx_loc.buffer(0.0001).bounds.squeeze(axis=0) # 0.0001 deg to get 1 tile only
            flx_loc = flx_loc.to_crs(crs_utm)
            flx_loc_check = flx_loc.buffer(250)
            box_geom_check = geometry.box(*flx_loc_check.total_bounds)
                
            for i, ts in enumerate(pbar := tqdm(datetimes)):
                pbar.set_description('Processing %s - TS %s' % (site, ts))
                ts1 = (ts - pd.Timedelta(days=round(day_range/2))).isoformat()
                ts2 = (ts + pd.Timedelta(days=round(day_range/2))).isoformat()
                for sat in ['S2B', 'S2A']: # sat loop as backup against NA files
                
                    na_rate = 1
                    ct = 0 # while loop counter
                    query = {'datasetId': ds,
                             'boundingBoxValues': [{'name': 'bbox',
                                                    'bbox': [*ext]}],
                             'dateRangeSelectValues': [{'name': 'temporal_interval',
                                                        'start': ts1,
                                                        'end': ts2}],
                             'stringChoiceValues': [{'name': 'productType',
                                                     'value': 'PPI'}]
                             }
                    if dataset == 'VI':
                        query['stringChoiceValues'].append({
                            'name': 'platformSerialIdentifier',
                            'value': sat})
                    matches = c.search(query)
                    
                    if (len(matches.results) == 0) & (sat == 'S2B'):
                        # loop (i.e. download) is skipped, ppi_files entry remains empty
                        logger.error('site {}: no usable S2B PPI data for TS{} (day range: {}). Trying S2A data now.\n'\
                                     .format(site, ts, day_range))
                        continue # If no success with S2B, continue with S2A
                    elif (len(matches.results) == 0) & (sat == 'S2A'):
                        logger.error('site {}: no usable S2A & S2B PPI data for TS{} (day range: {}).\n'\
                                     .format(site, ts, day_range))
                        datetimes.iloc[i] = 0 # no suitable imagery available flag
                        break # No S2A+S2B -> break sat loop and start next TS iteration
                    
                    matches.download(download_dir=odir) # download all "candidates" to have alternatives for NA check
                        
                    matches_filenames = pd.Series([x['filename'] for x in matches.results])
                    matches_dates = pd.to_datetime(pd.Series([x[3:18] for x in matches_filenames]))
                    matches_datediffs = abs(matches_dates - ts).sort_values().drop_duplicates()
                    
                    while na_rate > .7: # while NA rate is too high, the loop will iterate
                    # through adjacent dates in ascending order until a usable PPI file is found
                        ft_ix = matches_datediffs.index[ct]
                        ft_date = matches_dates[ft_ix]
                        # date matching
                        match_ix = matches_dates[matches_dates == ft_date].index
                        if len(match_ix) == 1:
                            ppi_files[i] = matches_filenames[match_ix].item()
                        else: # crs matching
                            logger.warning('site {}: {} date matches, possibly due to CRS edge case. files: {}\n'\
                                           .format(site, len(match_ix), matches_filenames[match_ix]))
                            utm_zone = 'T' + crs_utm.utm_zone[:2] # target CRS
                            correct_crs_ix = matches_filenames[match_ix].str.contains(utm_zone)
                            if correct_crs_ix.sum() == 1: # only 1 true value
                                ppi_files[i] = matches_filenames[match_ix][correct_crs_ix].item()
                            else:
                                ppi_files[i] = matches_filenames[match_ix][correct_crs_ix.index[0]]
                                logger.warning('site {}: still more imagery remaining after CRS matching. First match is kept.\n'.format(site))
                        # Check matched file for NAs. When too many, loop will test next file (chronologically).
                        with rio.open(odir / ppi_files[i]) as src:
                            ppi, _ = riom.mask(src, shapes=[box_geom_check],
                                               crop=True, nodata=-32768, indexes=1)
                        na_rate = len(ppi[ppi == -32768]) / np.size(ppi)
                        ct += 1
                        # Last iteration:
                        if (ct == len(matches_datediffs)) & (sat == 'S2B'):
                            logger.warning('site {}: Matching S2B files for TS{} (day range: {}) are >70% NA. Trying S2A data now.\n'\
                                           .format(site, ts, day_range))
                            na_rate = -9999
                        elif (ct == len(matches_datediffs)) & (sat == 'S2A'):
                            logger.warning('site {}: All S2A+S2B files for TS{} (day range: {}) are >70% NA.\n'\
                                           .format(site, ts, day_range))
                            na_rate != -9999
                            ppi_files[i] = ''
                            datetimes.iloc[i] = 0
                    if na_rate != -9999:
                        logger.debug('site {} - TS{}: Found suitable file {}.\n'.format(site, ts, ppi_files[i]))
                        break # second sat loop will only start when all files are NA, else break sat loop      
                # END of satellite loop
            # END of timestamp loop
            logger.debug('NAr: {}, count: {}, sat: {}, ts-iter: {}\n'.format(na_rate, ct, sat, i))
            logger.debug('datetimes: {} \n'.format(datetimes))
            logger.debug('ppi_files: {} \n'.format(ppi_files))
        

            datetimes = datetimes[datetimes != 0] # remove entries without match
            dtakes = self.flx_hsi_gdf.loc[datetimes.index, 'dataTakeID'] # for UID of each PPI img
            ppi_files = [x for x in ppi_files if x] # removes empty strings
            if len(datetimes) != len(ppi_files):
                raise ValueError('length of datetimes and ppi_files for site {} does not align.'.format(site))
    
            ppi_files_new = ['{}_{}_PPI_{}_{}.tif'.format(f[3:18], dataset, site, dtakes.iloc[i]) for i,f in enumerate(ppi_files)]
            for i, ts in enumerate(datetimes):
                self.flx_hsi_gdf.loc[(self.flx_hsi_gdf.name == site) & \
                                     (self.flx_hsi_gdf.startdate == ts),
                                     'ppi_file'] = ppi_files_new[i]
            # crop & save PPI rasters
            flx_loc_pb = flx_loc.buffer(1000)
            box_geom = geometry.box(*flx_loc_pb.total_bounds)
            for i, f in enumerate(ppi_files):
                with rio.open(odir / f) as src:
                    ppi, itrans = riom.mask(src, shapes=[box_geom],
                                            crop=True, nodata=-32768)
                    ometa = src.meta
                    
                ometa['width'] =  ppi.shape[2]
                ometa['height'] =  ppi.shape[1]
                # from rio docs: "Note that the interpretation of the 3 axes is (bands, rows, columns)"
                ometa['transform'] =  itrans
                with rio.open(odir / ppi_files_new[i], 'w', **ometa) as dest:
                    dest.write(ppi)
            # After cropping, delete all original files (originals start with their dataset abbr.)
            rm_files = list(odir.glob(dataset + '*.tif'))
            logger.debug('deleting originals: {}'.format([f.name for f in rm_files]))
            for f in rm_files:
                f.unlink()
            
        if save is True:
            fname = 'all_sites_{}_{}_{}_{}_{}.gpkg'\
                .format(self.sensor, self.ptype['sr'], self.ptype['mask'],
                        self.ptype['rad'], self.ptype['aggr'])
            logger.info('Saving file {}'.format(fname))
            print('Saving file: {}'.format(fname))
            self.flx_hsi_gdf.to_file(self.out_dir / fname, driver='GPKG')
        return
    
    def hsi_add_spei_ppi(self, zonal = False, save = False):
        '''
        After SPEI/PPI data have been acquired and processed using the methods
        icos_ppi_get/icos_cds_get, icos_eobs_pet and the SPEI calculation in R,
        they can be matched with the final flux+HSI data frame generated by the
        hsi_fp_crop method. All remaining imagery will be matched with corresp.
        SPEI & PPI values.
        
        Args:
            zonal (bool, optional): If true, PPI files will be cropped with
                zonal geometries instead of FFPs.
            save (bool, optional): If true, the resulting data frame is saved
                as GeoPackage.
        '''
        icos_list = self.flx_hsi_gdf.name.unique().tolist()
        spei_list = [0]*len(icos_list)
        fdir = self.img_dir.parent / 'Copernicus/S2PPI'
        dparse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d')
        for i,site in enumerate(icos_list):
            spei_list[i] = pd.read_csv(self.icos_dir / site / 'eobs_spei.csv',
                               parse_dates=['Date'], date_parser=dparse)\
                [['Date', 'SPEI365_{}'.format(site)]]
        spei = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date'), spei_list)
        spei['date_str'] = spei.Date.dt.strftime('%Y-%m-%d')
    
        insert_ix = self.flx_hsi_gdf.columns.get_loc('b001')
        if 'SPEI_365' not in self.flx_hsi_gdf.columns:
            self.flx_hsi_gdf.insert(insert_ix, 'SPEI_365', -9999)
        if 'PPI' not in self.flx_hsi_gdf.columns:
            self.flx_hsi_gdf.insert(insert_ix, 'PPI', -9999)

        
        for site in icos_list:
            
            datelist = self.flx_hsi_gdf.loc[self.flx_hsi_gdf.name == site, 'date']
            dtakes = self.flx_hsi_gdf.loc[self.flx_hsi_gdf.name == site, 'dataTakeID']
            #site_spei = spei.loc[spei.date_str.isin(datelist), ['date_str', 'SPEI365_{}'.format(site)]]
            
            epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            flx_loc = self.flx_loc.loc[self.flx_loc.name == site, 'geometry'].to_crs(crs_utm)
            
            if zonal:
                ecosystem = self.icos_db.loc[self.icos_db.name == site, 'ecosystem'].item()
                if ecosystem in ['MF', 'DBF', 'EBF', 'ENF']:
                    r = 80
                elif ecosystem in ['GRA', 'OSH']:
                    r = 50
                elif ecosystem == 'WET':
                    r = 40
                elif ecosystem == 'CRO':
                    r = 30
                else:
                    raise ValueError('Ecosystem {} not supported '.format(ecosystem) +
                                     'Please update codebase.')
                ppi_geom = flx_loc.buffer(r).item()
            else:
                crs_lam = proj.CRS.from_epsg('3035')
                transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)

            for i, date in enumerate(datelist):
                if zonal:
                    pass
                else:
                    lam_poly = self.flx_hsi_gdf.loc[datelist.index[i], 'geometry']
                    ppi_geom = stransform(transf.transform, lam_poly)
                fname = self.flx_hsi_gdf.loc[
                    (self.flx_hsi_gdf.name == site) &
                    (self.flx_hsi_gdf.date == date), 'ppi_file'].item()
                if len(fname) == 0:
                    logger.error('{}: no ppi_file matching DT {}.'.format(date, dtakes.iloc[i]))
                    continue
                fpath = fdir / fname
                with rio.open(fpath) as src: # ppi rasters are cropped
                    ppi, otrans = riom.mask(src, shapes=[ppi_geom], crop=True,
                                            all_touched=True, nodata=-32768,
                                            indexes=1)
                ppi = ppi / 10000
                ppi[ppi < 0] = np.nan
                spei_val = spei.loc[spei.date_str == date, 'SPEI365_{}'.format(site)].item()
                ppi_val = np.nanmean(ppi, axis=(1, 0))
                if ppi_val == 0:
                    logger.warning('{} - TS{}: Calculated PPI = 0'.format(site, date))
                self.flx_hsi_gdf.at[(self.flx_hsi_gdf.name == site) & \
                                    (self.flx_hsi_gdf.date == date),
                                    'PPI'] = ppi_val
                # mean of cropped area for more robust ppi estimate
                self.flx_hsi_gdf.at[(self.flx_hsi_gdf.name == site) & \
                                    (self.flx_hsi_gdf.date == date),
                                    'SPEI_365'] = spei_val

            # Looping over single SPEI values to ensure correct date matching
            #for ix, row in site_spei.iterrows(): # Use '.at' for single values!
            #    self.flx_hsi_gdf.at[(self.flx_hsi_gdf.name == site) & \
            #                        (self.flx_hsi_gdf.date == row.iloc[0]), 'SPEI_365'] = row.iloc[1]

        if (self.flx_hsi_gdf.SPEI_365 == -9999).any():
            logger.error('No SPEI value was found for some acquisition dates!')
        if (self.flx_hsi_gdf.PPI == -9999).any():
            logger.error('No PPI value was found for some acquisition dates! Observations will be removed.')
            self.flx_hsi_gdf = self.flx_hsi_gdf[self.flx_hsi_gdf.ppi != -9999, :]
        if save is True:
            fname = 'all_sites_{}_{}_{}_{}_{}_covars.gpkg'\
                .format(self.sensor, self.ptype['sr'], self.ptype['mask'],
                        self.ptype['rad'], self.ptype['aggr'])
            logger.info('Saving file {}'.format(fname))
            print('Saving file: {}'.format(fname))
            self.flx_hsi_gdf.to_file(self.out_dir / fname, driver='GPKG')
        return

    def load_spei_ppi_db(self, fdir = None, sr = 'vnir', zonal = False,
                         upw = False, aggr = 'na'):
        '''
        Load the processed, cleaned version of the imagery data base encompas-
        sing ICOS flux estimates, spectrometer data cropped to ROIs, and co-
        variates (PPI, SPEI).
        
        Args:
            fdir (str, optional): Directory with saved HSICOS data.
            sr (string): The spectral range for which the GeoPackage
                includes HSI data.
            zonal (bool, optional): The GeoPackage is based on zonal statistics
                instead of footprint estimates.
            upw (bool, optional): The GeoPackage contains upwelling radiation
                instead of reflectance values.
            aggr (string, optional): Type of aggregation of daily ICOS values,
                either "na", "mean" or "sum".
        '''
        if (upw is True) & (sr != 'vis'):
            logger.info('upw is true but sr is not "vis". Since UPW ' +
                        'calculation is only possible for 400-700nm, ' +
                        'sr will be set to "vis"')
            sr = 'vis'
        if fdir is None:
            fdir = self.out_dir
        t = 'upw' if upw is True else 'ref' # upwelling radiation or reflectance
        a = 'zst' if zonal is True else 'ffp' # flux footprint
        self.ptype = {'rad': t, 'mask': a, 'sr': sr, 'aggr': aggr} # processing info
        
        fname = 'all_sites_{}_{}_{}_{}_{}_covars.gpkg'.format(self.sensor, sr, a, t, aggr)
        logger.info('Loading file {}'.format(fname))
        print('Loading file: {}'.format(fname))
        
        self.flx_hsi_gdf = gpd.read_file(fdir / fname, driver='GPKG')
        self.flx_hsi_gdf['startdate'] = pd.to_datetime(
            self.flx_hsi_gdf.startdate)
        
        return self.flx_hsi_gdf
    
        
### DIMENSION REDUCTION #######################################################

    def dimred(self, method, ncomp, save = False):
        cube = self.flx_hsi_gdf.filter(regex=('b\d\d\d')) # all columns starting with b followed by 3 digits (band columns)
        na_cols = np.where(np.isnan(cube))[1]
        
        for col in na_cols:
            na_ix = np.where(np.isnan(cube.iloc[:, col]))[0]
            if len(na_ix) > 1:
                for ix in na_ix:
                   cube.iloc[ix, col] = (cube.iloc[ix-1, col] + cube.iloc[ix+1, col]) / 2
            elif len(na_ix) == 1:
                ix = na_ix[0]
                cube.iloc[ix, col] = (cube.iloc[ix-1, col] +  cube.iloc[ix+1, col]) / 2
        if method == 'sivm':
            cubeT = np.asarray(cube).T
            lvm = SIVM(cubeT, num_bases=ncomp, dist_measure='l2')
            lvm.factorize()
            self.dr_score = lvm.H
            self.dr_base = lvm.W
        elif method == 'pca':
            pca_s = decomposition.PCA(n_components=ncomp)
            self.dr_score = pca_s.fit_transform(cube)
            self.dr_base = pca_s.components_.T * np.sqrt(pca_s.explained_variance_)
        flx_comp_gdf = pd.concat([self.flx_hsi_gdf.filter(regex=('^(?!b\d\d\d)')).iloc[:,:-1], # all columns not starting with bxxx
                                  pd.DataFrame(self.dr_score.T, columns=['comp{}'.format(str(x).zfill(2)) for x in range(1, ncomp+1)]),
                                  self.flx_hsi_gdf.loc[:, 'geometry']],
                                 axis=1).reset_index(drop=True)
        if save is True:
            flx_comp_gdf.to_file(
                self.out_dir /\
                ('all_sites_{}_{}_{}'.format(self.sensor, method, self.ptype['sr']) +
                 '_{}_{}.gpkg'.format(self.ptype['mask'], self.ptype['rad'])),
                driver='GPKG')
            pd.DataFrame(self.dr_base).to_csv(self.out_dir /\
                ('dimred_{}_base_{}_{}'.format(method, self.sensor, self.ptype['sr']) +
                 '_{}_{}.csv'.format(self.ptype['mask'], self.ptype['rad'])))
        
        return flx_comp_gdf