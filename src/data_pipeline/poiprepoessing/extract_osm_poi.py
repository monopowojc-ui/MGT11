"""
extract_osm_poi.py
----------------------------------------------------
Extract all POIs from Shanghai OSM PBF

Input:
    data/raw/shanghai-latest.osm.pbf

Output:
    data/poi/shanghai_poi.gpkg
    data/poi/shanghai_poi.csv
"""

# extract_osm_poi.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import osmium                    # ← 必须导入
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# 从新全局配置导入所需变量
from src.config import (
    OSM_PBF,
    POI_GPKG,
    POI_CSV,
    SOURCE_CRS,
    OSM_TAGS,
)

# 从同级目录导入分类器
from poi_category import POICategoryMapper


# ============================================================
# 自定义 Handler：提取所有符合条件的 POI（节点）
# ============================================================
class POIHandler(osmium.SimpleHandler):
    def __init__(self, category_mapper):
        super().__init__()
        self.mapper = category_mapper
        self.records = []

    def node(self, n):
        if not n.tags:
            return
        tags = dict(n.tags)
        # 快速过滤：是否包含 OSM_TAGS 中的任意键
        if not any(key in tags for key in OSM_TAGS):
            return
        category = self.mapper.classify(tags)
        if category == "Other":
            return
        record = {
            "osm_id": n.id,
            "lat": n.location.lat,
            "lon": n.location.lon,
            "category": category,
            "geometry": Point(n.location.lon, n.location.lat),
        }
        # 将所有标签字段加入记录
        for key, value in tags.items():
            if key not in record:
                record[key] = value
        self.records.append(record)


# ============================================================
# 提取POI（替代原来的 load_osm + extract_poi）
# ============================================================

def extract_poi():
    """直接读取 PBF，返回 GeoDataFrame"""
    print("=" * 60)
    print("Extracting POIs from OSM PBF...")
    print(OSM_PBF)

    mapper = POICategoryMapper()
    handler = POIHandler(mapper)

    # 带进度条读取（可选，但 tqdm 无法直接挂钩，可简单打印）
    print("Scanning OSM file...")
    handler.apply_file(OSM_PBF, locations=True)   # locations=True 确保获取坐标

    print(f"Extracted {len(handler.records)} POIs.")
    return handler.records


# ============================================================
# 构建 GeoDataFrame（清洗）
# ============================================================

def clean_poi(records):
    """将记录列表转为 GeoDataFrame 并执行清洗"""
    print("=" * 60)
    print("Building GeoDataFrame and cleaning...")

    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], crs=SOURCE_CRS)

    gdf = gpd.GeoDataFrame(records, crs=SOURCE_CRS)

    # 去重（基于 geometry 和 category 等，可根据需要调整）
    gdf = gdf.drop_duplicates(subset=["geometry", "category"])

    # 删除几何无效的行（Shapely 点总是有效的，但以防万一）
    gdf = gdf[gdf.geometry.notna()]

    # 只保留你想要的列（根据你原来的 clean_poi 逻辑）
    # 注意：原代码保留了一些标签列，我们保留所有标签列，但保证存在
    # 这里我们可以选择保留所有列，因为后面 category_mapping 不需要额外添加分类（已完成）
    # 但为了保持与原代码一致，我们仅保留需要的列（但原代码是基于 columns 筛选，这里我们已有了）
    # 实际上，由于我们已经在 handler 中添加了 tags 作为列，可以直接使用
    # 如果希望限制列，可按照原逻辑，但原逻辑中 “if col in gdf.columns” 会保留存在的列
    # 这里我们维持原样：保留所有标签列 + name + geometry
    # 但原代码还试图从 gdf.columns 中找出存在的标签列，我们不必多此一举，因为 handler 已包含这些列
    # 我们可以直接 reset_index
    gdf = gdf.reset_index(drop=True)

    print(f"After cleaning: {len(gdf)} POIs.")
    return gdf


# ============================================================
# 分类（但实际上已在提取时完成，此函数可改为统计或验证）
# ============================================================

def category_mapping(gdf):
    """如果之前未分类，则进行分类；此处仅做统计展示"""
    print("=" * 60)
    print("POI Category Distribution:")

    # 如果还没有 category 列（以防万一），则调用 mapper 添加
    if "category" not in gdf.columns:
        mapper = POICategoryMapper()
        gdf = mapper.batch_classify(gdf)

    print(gdf["category"].value_counts())
    return gdf


# ============================================================
# 保存
# ============================================================

def save(gdf):
    print("=" * 60)
    print("Saving...")
    # 保存 CSV（去掉几何列）
    gdf.drop(columns=["geometry"]).to_csv(POI_CSV, index=False, encoding="utf-8-sig")
    # 保存 GeoPackage
    gdf.to_file(POI_GPKG, driver="GPKG")
    print(f"Saved to {POI_CSV}")
    print(f"Saved to {POI_GPKG}")

# ============================================================
# 统计
# ============================================================

def statistics(gdf):
    print("=" * 60)
    print("Statistics")
    print()
    print("Total POIs:", len(gdf))
    print()
    print("Category distribution:")
    print(gdf["category"].value_counts())
    print()
    if "name" in gdf.columns:
        print("Named POIs:", gdf["name"].notna().sum())


# ============================================================
# Main
# ============================================================

def main():
    # 1. 提取（直接替代 load_osm + extract_poi）
    records = extract_poi()   # 返回 list of dict

    # 2. 转为 GeoDataFrame 并清洗
    poi_gdf = clean_poi(records)

    # 3. 分类统计（实际上已经分好，但为了展示）
    poi_gdf = category_mapping(poi_gdf)

    # 4. 统计信息
    statistics(poi_gdf)

    # 5. 保存
    save(poi_gdf)

    print("=" * 60)
    print("Finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()