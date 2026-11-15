from __future__ import annotations

import numpy as np

from sim.gait import GaitSpec
from sim.rollout import Sim

_G: dict = {}


def init_worker(cfg: dict, imp_cfg: dict):
    if cfg["cost"].get("style"):
        from sim.rollout_style import StyleSim
        _G["sim"] = StyleSim(cfg)
    else:
        _G["sim"] = Sim(cfg)
    _G["cfg"] = cfg
    _G["imp"] = imp_cfg


def eval_task(task: dict):
    sim: Sim = _G["sim"]
    sim.limiter.set_grade(_G["imp"], task["grade"], task["mode"], tuple(task["affected"]))
    spec = GaitSpec(_G["cfg"]["gait"], task["free_legs"])
    o = sim.run(spec, np.asarray(task["x"]))
    keys = ("fell", "t_fall", "v_mean", "progress", "lat_rate", "yaw_abs", "tilt", "energy", "duty", "hind_load", "z_mean", "win", "wobble", "shape_err", "meta_pen", "meta_mean", "toeoff_pen", "td_lag", "retract_pen", "peak_pen", "paw_pen", "paw_pitch", "ref_err", "carpus_rom", "yaw_rate", "clear_pen", "clearance", "spine_pen", "lumb_mean", "pdiff_mean", "track_pen", "track_excess", "sole_pen", "swing_pitch")
    return o["cost"], {k: o.get(k) for k in keys}
