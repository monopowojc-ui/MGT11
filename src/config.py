# src/config.py
from pathlib import Path
import math

# ============================================================
# 1. 项目根目录定位（自动适应你的目录结构）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 定位到 E:\2\MGT

# ============================================================
# 2. 基础目录（固定不变）
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"    # 中间文件（如 POI 提取结果）
PROCESSED_DIR = DATA_DIR / "processed"  # 最终特征、图数据

# 自动创建目录
for d in [INTERIM_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. 城市配置字典（核心：所有城市参数集中管理）
# ============================================================
CITY_CONFIGS = {
    "shanghai": {
        "name": "上海",
        "crs": "EPSG:4326",
        "target_crs": "EPSG:3857",
        "buffer_radius": 800,
        # 输入文件
        "files": {
            "osm_pbf": RAW_DIR / "shanghai-260714.osm.pbf",
            "station": RAW_DIR / "stationInfo.csv",
            "od_flow": RAW_DIR / "metroData_ODFlow.csv",
            "inout_flow": RAW_DIR / "metroData_InOutFlow.csv",
            "weather": RAW_DIR / "shanghai_weatherHourly.csv",
            "workday": RAW_DIR / "workday_calendar.csv",
        },
        # 输出文件（自动带上前缀，互不干扰）
        "outputs": {
            "poi_gpkg": INTERIM_DIR / "shanghai_poi.gpkg",
            "poi_csv": INTERIM_DIR / "shanghai_poi.csv",
            "station_poi": PROCESSED_DIR / "shanghai_station_poi.csv",
            "station_entropy": PROCESSED_DIR / "shanghai_entropy.csv",
            "semantic_graph": PROCESSED_DIR / "shanghai_semantic.npy",
            "fused_graph": PROCESSED_DIR / "shanghai_fused.npy",
        },
        # 多图融合权重（语义、空间、出行）
        "fusion_weights": [0.4, 0.3, 0.3],
    },
    
    "beijing": {
        "name": "北京",
        "crs": "EPSG:4326",
        "target_crs": "EPSG:3857",
        "buffer_radius": 800,
        "files": {
            "osm_pbf": RAW_DIR / "beijing-latest.osm.pbf",     # 未来下载
            "station": RAW_DIR / "beijing_stations.csv",
            "od_flow": RAW_DIR / "beijing_od.csv",
            "inout_flow": RAW_DIR / "beijing_inout.csv",
            "weather": RAW_DIR / "beijing_weather.csv",
            "workday": RAW_DIR / "beijing_workday.csv",
        },
        "outputs": {
            "poi_gpkg": INTERIM_DIR / "beijing_poi.gpkg",
            "poi_csv": INTERIM_DIR / "beijing_poi.csv",
            "station_poi": PROCESSED_DIR / "beijing_station_poi.csv",
            "station_entropy": PROCESSED_DIR / "beijing_entropy.csv",
            "semantic_graph": PROCESSED_DIR / "beijing_semantic.npy",
            "fused_graph": PROCESSED_DIR / "beijing_fused.npy",
        },
        "fusion_weights": [0.3, 0.3, 0.4],
    }
}

# ============================================================
# 4. 城市切换开关（核心：只需改这一个变量）
# ============================================================
CITY = "shanghai"   # 迁移到北京时，改为 "beijing"

# 激活当前城市配置
ACTIVE = CITY_CONFIGS[CITY]

# ============================================================
# 5. 为了方便旧代码兼容，将常用字段扁平化导出（可选）
# ============================================================
OSM_PBF = ACTIVE["files"]["osm_pbf"]
STATION_FILE = ACTIVE["files"]["station"]
OD_FILE = ACTIVE["files"]["od_flow"]
INOUT_FILE = ACTIVE["files"]["inout_flow"]
WEATHER_FILE = ACTIVE["files"]["weather"]
WORKDAY_FILE = ACTIVE["files"]["workday"]

POI_GPKG = ACTIVE["outputs"]["poi_gpkg"]
POI_CSV = ACTIVE["outputs"]["poi_csv"]
STATION_POI_CSV = ACTIVE["outputs"]["station_poi"]
STATION_ENTROPY = ACTIVE["outputs"]["station_entropy"]
SEMANTIC_GRAPH = ACTIVE["outputs"]["semantic_graph"]
FUSED_GRAPH = ACTIVE["outputs"]["fused_graph"]

SOURCE_CRS = ACTIVE["crs"]
TARGET_CRS = ACTIVE["target_crs"]
BUFFER_RADIUS = ACTIVE["buffer_radius"]
FUSION_WEIGHTS = ACTIVE["fusion_weights"]

# ============================================================
# 6. 通用参数（不分城市）
# ============================================================
BUFFER_AREA = math.pi * BUFFER_RADIUS * BUFFER_RADIUS
BUFFER_AREA_KM2 = BUFFER_AREA / 1_000_000
TOP_K = 10
SIMILARITY_METHOD = "cosine"
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "Restaurant", "Shopping", "Office", "Education", "Medical",
    "Residential", "Transport", "Entertainment", "Hotel", "Finance"
]
# ============================================================
# OSM 过滤标签（用于 extract_osm_poi.py）
# ============================================================
OSM_TAGS = {
    "amenity": True,
    "shop": True,
    "office": True,
    "building": True,
    "tourism": True,
    "leisure": True,
    "railway": True,
    "public_transport": True,
}

# ============================================================
# POI 目录（保持与旧代码兼容，实际上可以和 INTERIM_DIR 相同）
# ============================================================
POI_DIR = INTERIM_DIR   # 或者如果你想保留单独的 poi 目录，可以改为 DATA_DIR / "poi"
POI_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在
# ============================================================
# 7. 打印当前配置（用于确认）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"当前城市: {ACTIVE['name']} (代码标识: {CITY})")
    print(f"OSM 文件: {OSM_PBF}")
    print(f"站点文件: {STATION_FILE}")
    print(f"POI 输出: {POI_GPKG}")
    print(f"融合权重: {FUSION_WEIGHTS}")
    print("=" * 60)