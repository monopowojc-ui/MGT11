import pandas as pd

class StationBuilder:

    def __init__(self, input_file):
        self.input_file = input_file


    def load(self):

        df = pd.read_csv(
            self.input_file
        )

        return df


    def clean(self, df):

        # 删除无用索引
        if "Unnamed: 0" in df.columns:
            df = df.drop(
                columns=["Unnamed: 0"]
            )


        # 字段标准化
        df = df.rename(
            columns={
                "stationID": "station_id",
                "name": "station_name",
                "lon": "longitude",
                "lat": "latitude",
                "neighbour": "neighbors"
            }
        )


        # 删除重复站点
        df = df.drop_duplicates(
            subset=["station_id"]
        )


        return df


    def build(self):

        df = self.load()

        df = self.clean(df)

        return df