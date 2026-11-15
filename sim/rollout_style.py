from __future__ import annotations

from sim.rollout import Sim


class StyleSim(Sim):
    def cost(self, o: dict) -> float:
        base = super().cost(o)
        st = self.cfg["cost"].get("style")
        if not st or o["win"] is None:
            return base
        pen = 0.0
        for j in ("stifle", "hock", "hip"):
            rom = o["win"][j][1] - o["win"][j][0]
            pen += ((rom - st["ref"][j]) / st["scale"]) ** 2
        return float(base + st["w"] * pen / 3.0)
