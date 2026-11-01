"""Small, seeded EV model used by the scenario simulator and the demo-data seeder.

It is a *plausible* vehicle, not a validated one: longitudinal dynamics (rolling + aero + inertia), regenerative
braking, a lumped battery thermal model, and a rolling Wh/km estimate that drives the "range" signal. Every effect
the demo questions ask about is a visible parameter, so evaluators can reproduce and audit the ground truth:

    colder / hotter ambient  -> more HVAC power and traction losses
    lower tyre pressure      -> higher rolling resistance
    more highway driving     -> higher aerodynamic load
    coolant degradation      -> lower battery cooling conductance, so cell temperature rises (fault injection)

ALL DATA PRODUCED HERE IS SIMULATED and must be labelled as such in any report.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator

SIGNALS = [
    "vehicle_speed_kph", "motor_power_kw", "soc_pct", "pack_voltage_v", "pack_current_a",
    "battery_temp_max_c", "battery_temp_min_c", "coolant_temp_c", "ambient_temp_c", "tire_pressure_kpa",
    "est_range_km", "odometer_km", "adas_active", "battery_overheat_warning",
]

MASS, G, RHO, CDA = 1600.0, 9.81, 1.2, 0.62
CRR0, ETA_DRIVE, ETA_REGEN = 0.009, 0.90, 0.60
AUX_KW, PACK_KWH = 0.45, 40.0
THERMAL_C, UA0, R_HEAT = 60_000.0, 600.0, 0.35      # J/K, W/K, ohm-equivalent
WARN_ON_C, WARN_OFF_C = 55.0, 50.0


@dataclass
class Fault:
    """Coolant degradation: cooling conductance ramps down to ``ua_factor`` between start_s and start_s + ramp_s."""
    start_s: float = 300.0
    ramp_s: float = 300.0
    ua_factor: float = 0.12
    heat_mult: float = 1.0


@dataclass
class VehicleModel:
    seed: int = 42
    ambient_c: float = 22.0
    tire_kpa: float = 230.0
    soc: float = 90.0
    odometer_km: float = 12_000.0
    ema_wh_km: float = 160.0
    fault: Fault | None = None
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.v = 0.0                       # m/s
        self.mode = "city"
        self.target_kph = 0.0
        self._target_until = 0.0
        self.t_batt = self.ambient_c + 3.0
        self.warning = 0
        self.adas = 0
        self._adas_ready_at = 0.0
        self._t0: float | None = None

    # -- driving profile ----------------------------------------------------
    def set_mode(self, mode: str, t: float) -> None:
        if mode == self.mode:
            return
        if self.mode == "highway" and self.adas:      # leaving the highway hands control back to the driver
            self.adas = 0
        self.mode = mode
        self._target_until = t
        if mode == "highway":
            self._adas_ready_at = t + 15.0

    def _retarget(self, t: float) -> None:
        if t < self._target_until:
            return
        r = self.rng
        if self.mode == "highway":
            self.target_kph, hold = r.uniform(85, 105), r.uniform(30, 90)
        elif self.mode == "city":
            if r.random() < 0.18:
                self.target_kph, hold = 0.0, r.uniform(8, 25)         # traffic light
            else:
                self.target_kph, hold = r.uniform(20, 52), r.uniform(10, 30)
        else:
            self.target_kph, hold = 0.0, 5.0
        self._target_until = t + hold

    # -- one simulation step --------------------------------------------------
    def step(self, t: float, dt: float = 1.0) -> dict[str, float]:
        if self._t0 is None:
            self._t0 = t
        el = t - self._t0
        self._retarget(t)
        dv = max(-2.2 * dt, min(1.4 * dt, self.target_kph / 3.6 - self.v))
        a = dv / dt
        self.v = max(0.0, self.v + dv)
        v, kph = self.v, self.v * 3.6

        crr = CRR0 * (1.0 + 0.006 * (230.0 - self.tire_kpa))
        force = (crr * MASS * G if v > 0.1 else 0.0) + 0.5 * RHO * CDA * v * v + MASS * a
        p_wheel = force * v / 1000.0
        cold = 1.0 + 0.012 * max(0.0, 22.0 - self.ambient_c)
        if p_wheel >= 0:
            p_trac = p_wheel / ETA_DRIVE * cold
        else:
            p_trac = p_wheel * ETA_REGEN if v > 2.0 else 0.0
        hvac = 0.3 + 0.12 * abs(self.ambient_c - 22.0)
        p_kw = p_trac + AUX_KW + hvac

        self.soc = max(0.0, self.soc - p_kw * dt / 3600.0 / PACK_KWH * 100.0)
        volt = 320.0 + 0.8 * self.soc - 0.08 * (p_kw * 1000.0 / 360.0)
        cur = p_kw * 1000.0 / volt
        self.odometer_km += v * dt / 1000.0
        if kph > 20.0:
            self.ema_wh_km += (p_kw * 1000.0 / kph - self.ema_wh_km) * 0.0015 * dt
        est_range = self.soc / 100.0 * PACK_KWH * 1000.0 / max(self.ema_wh_km, 50.0)

        # thermal model (+ optional coolant-degradation fault)
        ua, heat_mult = UA0, 1.0
        if self.fault and el >= self.fault.start_s:
            k = min(1.0, (el - self.fault.start_s) / max(self.fault.ramp_s, 1.0))
            ua = UA0 * (1.0 + (self.fault.ua_factor - 1.0) * k)
            heat_mult = 1.0 + (self.fault.heat_mult - 1.0) * k
        t_cool = self.ambient_c + 5.0 + 0.15 * (self.t_batt - self.ambient_c)
        self.t_batt += (R_HEAT * cur * cur * heat_mult - ua * (self.t_batt - t_cool)) / THERMAL_C * dt
        spread = 1.2 + 0.015 * abs(cur)
        t_max, t_min = self.t_batt + spread / 2, self.t_batt - spread / 2
        if t_max >= WARN_ON_C:
            self.warning = 1
        elif t_max < WARN_OFF_C:
            self.warning = 0

        # ADAS: engages on the highway above 60 km/h, occasional spontaneous disengagement
        if self.adas:
            if self.rng.random() < dt / 1500.0:
                self.adas, self._adas_ready_at = 0, t + self.rng.uniform(25, 60)
        elif self.mode == "highway" and kph > 60.0 and t >= self._adas_ready_at:
            self.adas = 1

        n = self.rng.gauss
        return {
            "vehicle_speed_kph": kph,
            "motor_power_kw": p_kw,
            "soc_pct": self.soc,
            "pack_voltage_v": volt,
            "pack_current_a": cur,
            "battery_temp_max_c": t_max,
            "battery_temp_min_c": t_min,
            "coolant_temp_c": t_cool,
            "ambient_temp_c": self.ambient_c + n(0, 0.1),
            "tire_pressure_kpa": self.tire_kpa + n(0, 0.3),
            "est_range_km": est_range,
            "odometer_km": self.odometer_km,
            "adas_active": float(self.adas),
            "battery_overheat_warning": float(self.warning),
        }


def run_route(model: VehicleModel, route: list[tuple[str, float]], t0: float, dt: float = 1.0) -> Iterator[tuple[float, dict]]:
    """route = [("city", 600), ("highway", 900), ...] -> yields (t, signals) every dt seconds."""
    t = t0
    for mode, seconds in route:
        model.set_mode(mode, t)
        end = t + seconds
        while t < end:
            yield t, model.step(t, dt)
            t += dt


def day_route(rng: random.Random, minutes: float, highway_share: float) -> list[tuple[str, float]]:
    """Alternate city / highway legs until ``minutes`` are filled, roughly honouring ``highway_share`` (0-1)."""
    total, route, hw_left = minutes * 60.0, [], minutes * 60.0 * highway_share
    city_left = total - hw_left
    mode = "city"
    while hw_left > 1 or city_left > 1:
        if mode == "city" and city_left > 1:
            d = min(city_left, rng.uniform(240, 600)); city_left -= d
        elif mode == "highway" and hw_left > 1:
            d = min(hw_left, rng.uniform(300, 900)); hw_left -= d
        else:
            mode = "highway" if mode == "city" else "city"
            continue
        route.append((mode, d))
        mode = "highway" if mode == "city" else "city"
    return route
