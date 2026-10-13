"""Explicit linear charging surrogates for sensitivity experiments.

Does NOT modify original Montoya XML. Surrogates replace ``Instance.curves``
only for planning with frvcpy; physical evaluation always uses true curves.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from data import ChargingCurve, Instance

SurrogateName = Literal["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]


def linear_secant_curve(true: ChargingCurve) -> ChargingCurve:
    """Constant rate on the chord from empty (0) to full battery (Q).

    Rate = Q / T_full where T_full = F_s(Q) from the true curve.
    Underestimates true charging time in the high-SOC concave region relative
    to the early steep segment, and overestimates relative to a pure early-rate
    extrapolation for partial fills from empty.
    """
    if len(true.times) < 2:
        raise ValueError("curve needs >= 2 breakpoints")
    t_full = float(true.times[-1])
    q_full = float(true.energies[-1])
    if t_full <= 0 or q_full <= 0:
        raise ValueError("invalid full charge breakpoint")
    return ChargingCurve(true.station_type, (0.0, t_full), (0.0, q_full))


def linear_early_rate_curve(true: ChargingCurve) -> ChargingCurve:
    """Constant rate equal to the first true PWL segment, extrapolated to Q.

    Optimistic for concave charging: underestimates time to reach high SOC.
    """
    if len(true.times) < 2:
        raise ValueError("curve needs >= 2 breakpoints")
    dt = float(true.times[1] - true.times[0])
    de = float(true.energies[1] - true.energies[0])
    if dt <= 0 or de <= 0:
        raise ValueError("invalid first charging segment")
    q_full = float(true.energies[-1])
    t_full = q_full * dt / de
    return ChargingCurve(true.station_type, (0.0, t_full), (0.0, q_full))


def surrogate_curves(
    instance: Instance, name: SurrogateName
) -> dict[str, ChargingCurve]:
    if name == "TRUE-NL":
        return dict(instance.curves)
    builder = {
        "LINEAR-SECANT": linear_secant_curve,
        "LINEAR-EARLY-RATE": linear_early_rate_curve,
    }[name]
    return {stype: builder(curve) for stype, curve in instance.curves.items()}


def instance_with_charging_model(
    instance: Instance, name: SurrogateName
) -> Instance:
    """Shallow-replaced instance whose ``curves`` follow the named model."""
    return replace(instance, curves=surrogate_curves(instance, name))


def surrogate_rate_wh_per_hour(curve: ChargingCurve) -> float:
    """Constant rate (Wh/h) implied by a 2-point linear curve."""
    return (curve.energies[-1] - curve.energies[0]) / (curve.times[-1] - curve.times[0])


def document_surrogate(instance: Instance, name: SurrogateName) -> list[dict]:
    """Per-station-type documentation rows for tables/reports."""
    rows = []
    for stype, true in sorted(instance.curves.items()):
        if name == "TRUE-NL":
            rows.append(
                {
                    "station_type": stype,
                    "model": name,
                    "definition": "piecewise-linear Montoya F_s(q)",
                    "breakpoints_t": ";".join(str(t) for t in true.times),
                    "breakpoints_e_wh": ";".join(str(e) for e in true.energies),
                    "constant_rate_wh_per_h": "",
                    "note": "ground-truth XML curve",
                }
            )
            continue
        surr = surrogate_curves(instance, name)[stype]
        rows.append(
            {
                "station_type": stype,
                "model": name,
                "definition": (
                    "secant 0→Q"
                    if name == "LINEAR-SECANT"
                    else "first-segment rate extrapolated to Q"
                ),
                "breakpoints_t": ";".join(str(t) for t in surr.times),
                "breakpoints_e_wh": ";".join(str(e) for e in surr.energies),
                "constant_rate_wh_per_h": f"{surrogate_rate_wh_per_hour(surr):.6f}",
                "note": (
                    "optimistic vs concave high-SOC region"
                    if name == "LINEAR-EARLY-RATE"
                    else "chord average empty→full"
                ),
            }
        )
    return rows
