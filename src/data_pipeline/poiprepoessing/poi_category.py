"""
poi_category.py
----------------------------------------
OSM POI Category Mapping

Author: Your Name
"""

# poi_category.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from collections import defaultdict
from src.config import FEATURE_COLUMNS   # 原来的 from config import ... 改成这行

# 其余代码（POICategoryMapper 类等）原封不动


class POICategoryMapper:

    """
    将OSM标签映射到论文使用的10类POI
    """

    def __init__(self):

        self.mapping = {

            "Restaurant": {
                "amenity": {
                    "restaurant",
                    "fast_food",
                    "cafe",
                    "food_court",
                    "bar",
                    "pub"
                }
            },

            "Shopping": {
                "shop": "*"
            },

            "Office": {
                "office": "*"
            },

            "Education": {
                "amenity": {
                    "school",
                    "college",
                    "university",
                    "kindergarten"
                }
            },

            "Medical": {
                "amenity": {
                    "hospital",
                    "clinic",
                    "pharmacy",
                    "doctors"
                }
            },

            "Residential": {
                "building": {
                    "residential"
                }
            },

            "Transport": {

                "railway": {
                    "station",
                    "subway_entrance"
                },

                "public_transport": {
                    "station",
                    "platform",
                    "stop_position"
                },

                "amenity": {
                    "bus_station",
                    "parking",
                    "taxi"
                }

            },

            "Entertainment": {

                "amenity": {
                    "cinema",
                    "theatre",
                    "nightclub"
                },

                "leisure": "*"

            },

            "Hotel": {

                "tourism": {
                    "hotel",
                    "hostel",
                    "guest_house"
                }

            },

            "Finance": {

                "amenity": {
                    "bank",
                    "atm"
                }

            }

        }

    # --------------------------------------------------

    def classify(self, row):

        """
        输入一条POI

        返回所属类别

        """

        for category, rules in self.mapping.items():

            for field, value in rules.items():

                if field not in row:

                    continue

                item = row[field]

                if item is None:

                    continue

                if item != item:

                    continue

                # 通配符
                if value == "*":

                    return category

                if item in value:

                    return category

        return "Other"

    # --------------------------------------------------

    def batch_classify(self, gdf):

        """
        GeoDataFrame

        增加category列
        """

        category = []

        for _, row in gdf.iterrows():

            category.append(

                self.classify(row)

            )

        gdf["category"] = category

        return gdf

    # --------------------------------------------------

    def statistics(self, gdf):

        """

        全市POI统计

        """

        if "category" not in gdf.columns:

            gdf = self.batch_classify(gdf)

        return gdf["category"].value_counts()

    # --------------------------------------------------

    def feature_template(self):

        """

        返回论文特征模板

        """

        feature = {}

        for item in FEATURE_COLUMNS:

            feature[item] = 0

        return feature


# =====================================================

def build_empty_feature():

    """

    建立站点空特征

    """

    feature = defaultdict(int)

    for name in FEATURE_COLUMNS:

        feature[name] = 0

    return feature


# =====================================================

if __name__ == "__main__":

    mapper = POICategoryMapper()

    print("=" * 60)

    print("POI Categories")

    for key in mapper.mapping.keys():

        print(key)

    print("=" * 60)