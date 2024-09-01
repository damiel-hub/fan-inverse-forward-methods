import sys
sys.path.append('src/')
from SPM import SPM
from EDR import EDR
import gis_utils
import reconstruct_fantopo

tiff_before = r"D:\YuanHungCHIU\c_Data\LaonongDEM_allTime\dem20231124_nobridge_copy.tif"
tiff_after = r"D:\YuanHungCHIU\c_Data\LaonongDEM_allTime\20240801@all\2024Aug02_1m_OH_all_align_merge_small.tif"
shape_boundary = r"D:\YuanHungCHIU\b_NTUinvest\v01_visibility_polygon\fanTopoPy\datasets\raw\shape\PT2024.shp"
volume_difference = gis_utils.calculate_volume_difference_within_polygon(tiff_before, tiff_after,shape_boundary,pltFlag=1)
print(volume_difference)