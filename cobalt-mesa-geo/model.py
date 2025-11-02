# model.py

import geopandas as gpd
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa_geo import GeoSpace
from agents import MinerAgent, RefineryAgent
import random, os, json, math
from shapely.geometry import Point

class CobaltGeoModel(Model):
    def __init__(self, mines_gjson="data/cobalt_mines.geojson", ref_gjson="data/refineries.geojson", prov_gjson="data/provinces.geojson", seed=42):
        super().__init__()
        random.seed(seed)
        self.schedule = RandomActivation(self)
        self.space = GeoSpace()
        self.market_price = 12.0  # $/kg
        self.market_pool = 0.0
        self.regulation_level = "LOW"
        self.llm_model = "mistral"
        self.logs = []
        os.makedirs("logs", exist_ok=True)

        self.datacollector = DataCollector(
            model_reporters={},  # we can add metrics later
            agent_reporters={}
        )

        # load data
        self.mines_gdf = gpd.read_file(mines_gjson).to_crs("EPSG:4326")
        self.ref_gdf = gpd.read_file(ref_gjson).to_crs("EPSG:4326")
        self.prov_gdf = gpd.read_file(prov_gjson).to_crs("EPSG:4326")

        # create miner agents
        for idx, row in self.mines_gdf.iterrows():
            a = MinerAgent(unique_id=f"miner_{idx}", model=self, shape=row.geometry, mine_type=row.get("type","artisanal"), production=row.get("prod",10))
            self.space.add_agents(a)
            self.schedule.add(a)

        # create refineries
        for idx, row in self.ref_gdf.iterrows():
            r = RefineryAgent(unique_id=f"ref_{idx}", model=self, shape=row.geometry, capacity=500, name=row.get("name",f"ref_{idx}"))
            self.space.add_agents(r)
            self.schedule.add(r)

    # simple geoutils
    def find_nearest_refinery(self, geom):
        # returns refinery agent object nearest to geom
        agents = [a for a in self.schedule.agents if hasattr(a, "capacity")]
        min_a = min(agents, key=lambda a: geom.distance(a.geometry))
        return min_a

    def distance_km(self, g1, g2):
        # rough haversine (both in lon/lat)
        lon1, lat1 = g1.x, g1.y
        lon2, lat2 = g2.x, g2.y
        # haversine
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2-lat1)
        dlambda = math.radians(lon2-lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def step(self):
        # each tick: gather data, step agents, update price, save logs
        self.datacollector.collect(self)
        self.schedule.step()
        # simple price dynamics: price nudges by pool supply
        supply = self.market_pool
        if supply > 100:
            self.market_price *= 0.995
        else:
            self.market_price *= 1.001
        # persist logs incrementally
        if len(self.logs) >= 50:
            self._flush_logs()

    def log_event(self, obj):
        self.logs.append(obj)

    def _flush_logs(self):
        fn = f"logs/log_tick_{self.schedule.time}.json"
        with open(fn, "w") as f:
            json.dump(self.logs, f, indent=2)
        self.logs = []

    def save_snapshot(self, fn="logs/snapshot_agents.geojson"):
        # export agent states to geojson for mapping
        rows = []
        for a in self.schedule.agents:
            rows.append({
                "id": a.unique_id,
                "type": getattr(a, "mine_type", getattr(a, "name", "refinery")),
                "cash": getattr(a, "cash", None),
                "inventory": getattr(a, "inventory", None),
                "geometry": a.geometry
            })
        gdf = gpd.GeoDataFrame(rows, geometry=[r["geometry"] for r in rows], crs="EPSG:4326")
        gdf.to_file(fn, driver="GeoJSON")
