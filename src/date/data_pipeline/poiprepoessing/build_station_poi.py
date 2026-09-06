# E:\2\MGT\src\data_pipeline\poiprepoessing\build_station_poi.py
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
# 将项目根目录加入 sys.path，确保能找到 src 模块
sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# ================== 从新全局配置导入 ==================
from src.config import (
    STATION_FILE,
    POI_GPKG,
    STATION_POI_CSV,
    SOURCE_CRS,
    TARGET_CRS,
    BUFFER_RADIUS,
    FEATURE_COLUMNS,
    CITY
)

# 从同级目录导入辅助函数
from poi_category import build_empty_feature


# ============================================================
# 核心函数：构建站点POI特征（含熵值计算）
# ============================================================
def build_station_poi_features():
    print("=" * 60)
    print(f"正在处理城市: {CITY}")
    print("构建站点POI特征...")

    # 1. 读取站点数据（列名：name, lon, lat）
    stations_df = pd.read_csv(STATION_FILE)
    stations_df['geometry'] = stations_df.apply(
        lambda row: Point(row['lon'], row['lat']), axis=1
    )
    stations_gdf = gpd.GeoDataFrame(stations_df, crs=SOURCE_CRS)

    # 2. 读取已分类的 POI 数据
    pois = gpd.read_file(POI_GPKG)
    if pois.empty:
        print("⚠️  POI 数据为空，请先运行 extract_osm_poi.py")
        return None

    # 3. 统一投影到米制（做缓冲区）
    stations_proj = stations_gdf.to_crs(TARGET_CRS)
    pois_proj = pois.to_crs(TARGET_CRS)

    # 4. 为每个站点做缓冲区，统计各类 POI 数量
    results = []
    for idx, station in tqdm(stations_proj.iterrows(), total=len(stations_proj), desc="处理站点"):
        buffer_geom = station['geometry'].buffer(BUFFER_RADIUS)
        pois_in_buffer = pois_proj[pois_proj.intersects(buffer_geom)]

        feature_dict = build_empty_feature()   # 返回 {cat:0}
        if not pois_in_buffer.empty:
            counts = pois_in_buffer['category'].value_counts()
            for cat, cnt in counts.items():
                if cat in feature_dict:
                    feature_dict[cat] = int(cnt)

        total_poi = sum(feature_dict.values())
        # 计算多样性熵（用于后续分析）
        entropy = 0.0
        if total_poi > 0:
            probs = np.array(list(feature_dict.values())) / total_poi
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))

        result_row = {
            'station': station['name'],   # 站名（修正列名）
            'lon': station['lon'],
            'lat': station['lat'],
            'total_poi': total_poi,
            'entropy': entropy,
            **feature_dict
        }
        results.append(result_row)

    # 5. 转为 DataFrame 并保存
    df_result = pd.DataFrame(results)
    df_result.to_csv(STATION_POI_CSV, index=False, encoding='utf-8-sig')

    print(f"✅ 站点POI特征已保存至: {STATION_POI_CSV}")
    print(f"   共处理 {len(df_result)} 个站点")
    print(f"   POI类别: {FEATURE_COLUMNS}")
    print("=" * 60)
    return df_result


if __name__ == "__main__":
    build_station_poi_features()