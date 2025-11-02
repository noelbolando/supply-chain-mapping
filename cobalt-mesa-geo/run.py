# run.py

from model import CobaltGeoModel
from visualize import plot_snapshot
import os

def main(ticks=50):
    m = CobaltGeoModel()
    for t in range(ticks):
        m.step()
    m.save_snapshot()
    plot_snapshot()

if __name__ == "__main__":
    main(50)
