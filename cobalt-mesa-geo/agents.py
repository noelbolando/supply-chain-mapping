# agents.py

from mesa import Agent
from mesa_geo import GeoAgent
from shapely.geometry import Point
import random, json
from llm_wrapper import llm_decide

class MinerAgent(GeoAgent):
    def __init__(self, unique_id, model, shape, mine_type="artisanal", production=10, crs="EPSG:4326"):
        super().__init__(unique_id, model, shape, crs)
        self.mine_type = mine_type
        self.inventory = float(production)
        self.cash = 1000.0 if mine_type=="industrial" else 200.0
        self.quality = random.uniform(0.3,0.9) if mine_type=="industrial" else random.uniform(0.15,0.6)
        self.trace_cert = False
        self.reputation = 0.5

    def step(self):
        # build a compact prompt summarizing local context
        nearest_ref = self.model.find_nearest_refinery(self.geometry)
        dist_km = self.model.distance_km(self.geometry, nearest_ref.geometry)
        prompt = (
            f"You are a {self.mine_type} miner with inventory={self.inventory:.1f}kg, cash={self.cash:.2f}, "
            f"quality={self.quality:.2f}. Nearest refinery {nearest_ref.name} at {dist_km:.1f}km. "
            f"Market price={self.model.market_price:.2f}. Regulation={self.model.regulation_level}. "
            "Decide SELL_PERCENT (0-1) and INVEST_TRACE (YES/NO). Output JSON."
        )
        resp, elapsed = llm_decide(prompt, temperature=0.0)
        # parse
        try:
            if isinstance(resp, str):
                jstart = resp.find('{')
                j = json.loads(resp[jstart:]) if jstart!=-1 else json.loads(resp)
            else:
                j = resp
            sell_pct = float(j.get("SELL_PERCENT", 1.0))
            invest = str(j.get("INVEST_TRACE","NO")).upper().startswith("Y")
        except Exception:
            sell_pct, invest = (1.0 if self.cash < 500 else 0.5, False)

        sell_amt = max(0.0, min(self.inventory, self.inventory * sell_pct))
        revenue = sell_amt * self.model.market_price * (1.05 if self.trace_cert else 0.9)
        self.inventory -= sell_amt
        self.cash += revenue
        self.model.market_pool += sell_amt

        if invest and self.cash >= 150:
            self.trace_cert = True
            self.cash -= 150

        # log action
        self.model.log_event({
            "tick": self.model.schedule.time,
            "agent": self.unique_id,
            "type": "miner_step",
            "sell_amt": sell_amt,
            "cash": self.cash,
            "inventory": self.inventory,
            "llm_resp": resp,
            "elapsed": elapsed
        })

class RefineryAgent(GeoAgent):
    def __init__(self, unique_id, model, shape, capacity=500, name="ref", crs="EPSG:4326"):
        super().__init__(unique_id, model, shape, crs)
        self.capacity = capacity
        self.inventory = 0.0
        self.cash = 10000.0
        self.name = name

    def step(self):
        # buy from market pool up to capacity
        space = self.capacity - self.inventory
        buy_amt = min(space, self.model.market_pool)
        buy_price = self.model.market_price * 1.02
        cost = buy_amt * buy_price
        if buy_amt > 0 and self.cash >= cost:
            self.cash -= cost
            self.inventory += buy_amt
            self.model.market_pool -= buy_amt
            self.model.log_event({
                "tick": self.model.schedule.time,
                "agent": self.unique_id,
                "type": "refinery_buy",
                "buy_amt": buy_amt,
                "cash": self.cash,
                "inventory": self.inventory
            })
