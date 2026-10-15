"""Central presentation labels for methods and search stages.

Internal CSV keys / checkpoint IDs stay unchanged (H2, hybrid_insert, …).
"""

from __future__ import annotations

METHOD_DISPLAY: dict[str, str] = {
    "H0": "Randomized Greedy Routing",
    "H0_strong": "Randomized Greedy + Local Search",
    "H1": "State-Only Hierarchical Pointer",
    "H2": "History Transformer Routing",
    "H3": "DeepSets Fleet Routing",
    "H4": "History Transformer + DeepSets (HAT-EVRP-NL)",
    "hybrid_H4": "History Transformer + DeepSets + frvcpy",
    "direct_CHARGE": "Learned Direct CHARGE Controller",
    "direct_plus_repair": "Direct CHARGE + frvcpy Repair",
}

# Compact axis/legend labels (avoid overlap in publication PNGs).
METHOD_SHORT: dict[str, str] = {
    "H0": "Greedy (H0)",
    "H0_strong": "Greedy+LS (H0_strong)",
    "H1": "State-only (H1)",
    "H2": "History (H2)",
    "H3": "DeepSets (H3)",
    "H4": "HAT-EVRP-NL (H4)",
}

SEARCH_DISPLAY: dict[str, str] = {
    "construct": "Neural Construction Only",
    "basic_ls": "Basic Local Search",
    "strong_ls": "Strong Local Search",
    "classical_ruin": "Classical Ruin-and-Repair",
    "neural_append": "Neural Append-Only Repair",
    "neural_insert": "Neural Position-Aware Repair",
    "hybrid_insert": "Classical + Neural Insertion Repair",
    "ls": "Local Search",
}

STAGE_DISPLAY: dict[str, str] = {
    "bc": "Behavior Cloning",
    "rl": "Standard REINFORCE",
    "rl_pomo": "POMO-style REINFORCE",
    "rl_repair": "Repair REINFORCE",
}


def method_label(key: str, *, short: bool = False) -> str:
    """Full display name, or compact axis/legend label when short=True."""
    if short:
        return METHOD_SHORT.get(key, key)
    name = METHOD_DISPLAY.get(key, key)
    if key not in METHOD_DISPLAY:
        return name
    return f"{name} ({key})"


def search_label(key: str, *, short: bool = False) -> str:
    name = SEARCH_DISPLAY.get(key, key)
    if short or key not in SEARCH_DISPLAY:
        return name if key not in SEARCH_DISPLAY else f"{name} ({key})"
    return f"{name} ({key})"


def stage_label(key: str) -> str:
    return STAGE_DISPLAY.get(key, key)


def wrap_label(text: str, width: int = 22) -> str:
    """Simple word wrap for matplotlib tick labels."""
    words = text.replace(" (", "\n(").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if "\n" in w:
            if cur:
                lines.append(cur)
            lines.append(w)
            cur = ""
            continue
        trial = f"{cur} {w}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)
