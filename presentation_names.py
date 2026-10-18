"""Central presentation labels for methods and search stages.

Internal CSV keys / checkpoint IDs stay unchanged (H2, hybrid_insert, …).
Reader-facing figures and tables MUST use these display names — never bare H1–H4.
"""

from __future__ import annotations

# Canonical long names (tables, captions, manuscript prose).
METHOD_DISPLAY: dict[str, str] = {
    "H0": "Greedy Construction",
    "H0_strong": "Greedy + Local Search",
    "H1": "State Pointer",
    "H2": "Route Transformer (RT)",
    "H3": "Fleet DeepSets",
    "H4": "Route Transformer + DeepSets (HAT-EVRP-NL)",
    "H4_open": "Route Transformer + Open-route DeepSets",
    "hybrid_H4": "Route Transformer + DeepSets + frvcpy",
    "direct_CHARGE": "Learned Direct CHARGE Controller",
    "direct_plus_repair": "Direct CHARGE + frvcpy Repair",
}

# Alias requested by publication style guide.
METHOD_DISPLAY_NAMES = METHOD_DISPLAY

# Compact chart labels (axes/legends). No internal codes.
METHOD_SHORT: dict[str, str] = {
    "H0": "Greedy",
    "H0_strong": "Greedy + LS",
    "H1": "State Pointer",
    "H2": "RT",
    "H3": "Fleet DeepSets",
    "H4": "RT + DeepSets",
    "H4_open": "RT + Open DeepSets",
}

# Multi-line axis labels when space is tight.
METHOD_WRAP: dict[str, str] = {
    "H0": "Greedy\nConstruction",
    "H0_strong": "Greedy +\nLocal Search",
    "H1": "State\nPointer",
    "H2": "Route\nTransformer",
    "H3": "Fleet\nDeepSets",
    "H4": "RT +\nDeepSets",
    "H4_open": "RT +\nOpen DeepSets",
}

PAIR_DISPLAY: dict[str, str] = {
    "H2-H1": "RT\n− State Pointer",
    "H3-H1": "Fleet DeepSets\n− State Pointer",
    "H4-H2": "RT + DeepSets\n− RT",
    "H4-H1": "RT + DeepSets\n− State Pointer",
}

SEARCH_DISPLAY: dict[str, str] = {
    "construct": "Neural construction",
    "basic_ls": "Basic local search",
    "strong_ls": "Strong local search",
    "classical_ruin": "Classical ruin–repair",
    "neural_append": "Neural append repair",
    "neural_insert": "Neural insert repair",
    "hybrid_insert": "Hybrid insert repair",
    "ls": "Local search",
}

STAGE_DISPLAY: dict[str, str] = {
    "bc": "Behavior Cloning",
    "rl": "Standard REINFORCE",
    "rl_pomo": "POMO-style REINFORCE",
    "rl_repair": "Repair REINFORCE",
}

# Stable colorblind-friendly method colors (shared across figures).
METHOD_COLORS: dict[str, str] = {
    "H0": "#999999",
    "H0_strong": "#666666",
    "H1": "#56B4E9",
    "H2": "#0072B2",
    "H3": "#E69F00",
    "H4": "#009E73",
    "H4_open": "#CC79A7",
}


def method_label(key: str, *, short: bool = False, wrap: bool = False) -> str:
    """Reader-facing method name (no internal H-codes)."""
    if wrap:
        return METHOD_WRAP.get(key, METHOD_SHORT.get(key, key))
    if short:
        return METHOD_SHORT.get(key, METHOD_DISPLAY.get(key, key))
    return METHOD_DISPLAY.get(key, key)


def pair_label(key: str) -> str:
    return PAIR_DISPLAY.get(key, key)


def search_label(key: str, *, short: bool = False) -> str:
    name = SEARCH_DISPLAY.get(key, key)
    return name


def stage_label(key: str) -> str:
    return STAGE_DISPLAY.get(key, key)


def method_color(key: str) -> str:
    return METHOD_COLORS.get(key, "#333333")


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


def methodology_name_note() -> str:
    """One-time reproducibility mapping for the manuscript."""
    return (
        "Internal ablation keys (H0–H4) map to display names: "
        "H0 Greedy Construction; H0_strong Greedy + Local Search; "
        "H1 State Pointer; H2 Route Transformer (RT); "
        "H3 Fleet DeepSets; H4 Route Transformer + DeepSets "
        "(HAT-EVRP-NL). RT encodes optimizer-enriched route-stop sequences, "
        "including charging-related features; a token is a stop representation, "
        "not necessarily a committed historical action. "
        "Reader-facing figures and tables use display names only."
    )
