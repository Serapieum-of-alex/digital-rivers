from typing import Dict
import pytest
import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
import geopandas as gpd


@pytest.fixture(scope="module")
def coello_df_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/fd4000.tif")


@pytest.fixture(scope="module")
def coello_fdt() -> Dict:
    return {
        "1,5": [],
        "1,6": [],
        "1,7": [],
        "2,5": [(1, 5)],
        "2,6": [],
        "2,7": [(1, 6), (1, 7)],
        "3,3": [],
        "3,4": [],
        "3,5": [(2, 5)],
        "3,6": [],
        "3,7": [(2, 6), (2, 7)],
        "3,8": [],
        "3,9": [],
        "4,3": [],
        "4,4": [(3, 3), (3, 4), (4, 3), (5, 3)],
        "4,5": [(3, 5), (3, 6)],
        "4,6": [],
        "4,7": [(3, 7)],
        "4,8": [(3, 8), (3, 9)],
        "4,9": [],
        "5,3": [],
        "5,4": [],
        "5,5": [(4, 4), (4, 5), (4, 6)],
        "5,6": [],
        "5,7": [],
        "5,8": [(4, 7), (4, 8), (5, 7)],
        "5,9": [(4, 9)],
        "6,2": [],
        "6,3": [],
        "6,4": [],
        "6,5": [(5, 4), (5, 5), (5, 6), (6, 4)],
        "6,6": [],
        "6,7": [],
        "6,8": [],
        "6,9": [(5, 8), (5, 9)],
        "7,1": [],
        "7,2": [(7, 1)],
        "7,3": [(6, 2)],
        "7,4": [(6, 3), (7, 3), (8, 3)],
        "7,5": [(7, 4), (8, 4)],
        "7,6": [(6, 5), (6, 6), (7, 5), (8, 5), (8, 6)],
        "7,7": [(6, 7)],
        "7,8": [],
        "7,9": [(6, 8), (6, 9)],
        "8,1": [],
        "8,2": [(8, 1)],
        "8,3": [(7, 2), (8, 2), (9, 2)],
        "8,4": [(9, 3), (9, 4)],
        "8,5": [],
        "8,6": [],
        "8,7": [(7, 6)],
        "8,8": [(7, 7), (7, 8), (8, 7)],
        "8,9": [],
        "8,10": [(7, 9)],
        "8,11": [(8, 10)],
        "8,12": [],
        "9,1": [],
        "9,2": [(9, 1), (10, 1)],
        "9,3": [(10, 2), (10, 3)],
        "9,4": [],
        "9,5": [(10, 4)],
        "9,6": [(9, 5), (10, 5)],
        "9,7": [(9, 6)],
        "9,8": [(9, 7)],
        "9,9": [(8, 8), (9, 8)],
        "9,10": [(8, 9)],
        "9,11": [(9, 10)],
        "9,12": [(8, 11), (8, 12), (9, 11), (10, 11)],
        "10,0": [],
        "10,1": [(10, 0), (11, 0), (11, 1)],
        "10,2": [(11, 2)],
        "10,3": [],
        "10,4": [],
        "10,5": [],
        "10,7": [],
        "10,8": [(10, 7)],
        "10,9": [(10, 8)],
        "10,10": [(9, 9), (10, 9)],
        "10,11": [(10, 10), (11, 10), (11, 11)],
        "10,12": [],
        "10,13": [(9, 12), (10, 12)],
        "11,0": [],
        "11,1": [(12, 1)],
        "11,2": [(12, 2)],
        "11,9": [],
        "11,10": [(11, 9)],
        "11,11": [],
        "12,1": [],
        "12,2": [],
    }


@pytest.fixture(scope="module")
def coello_dem_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/coello-dem-4000.tif")


@pytest.fixture(scope="function")
def coello_slope() -> np.ndarray:
    return np.load("tests/data/coello/slope.npy")


@pytest.fixture(scope="function")
def coello_max_slope() -> np.ndarray:
    return np.load("tests/data/coello/coello-max-slope.npy")


@pytest.fixture(scope="module")
def coello_flow_direction_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/flow-direction-with-outfall.tif")


@pytest.fixture(scope="function")
def flow_direction_array_cells_indices() -> np.ndarray:
    return np.load("tests/data/coello/flow_direction_array.npy")


@pytest.fixture(scope="module")
def coello_flow_accumulation_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/flow-accumulation.tif")


@pytest.fixture(scope="module")
def coello_outfall() -> GeoDataFrame:
    """Point Geometry of the Coello river outfall"""
    return gpd.read_file("tests/data/coello/coello-outfall.geojson")


@pytest.fixture(scope="function")
def elev_sink_free() -> np.ndarray:
    return np.load("tests/data/coello/elevation-sink-free.npy")
