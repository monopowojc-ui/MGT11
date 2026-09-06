from src.data_pipeline.station.station_builder import StationBuilder


builder = StationBuilder(
    "data/raw/stationInfo.csv"
)


station = builder.build()


station.to_csv(
    "data/processed/station_master.csv",
    index=False
)


print(station.head())

print(
    "station number:",
    len(station)
)