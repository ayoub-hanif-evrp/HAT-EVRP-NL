"""Synthetic teacher generation, H1-H4 training, and verified decoding."""

from __future__ import annotations

import argparse
import csv
import copy
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from charging import (
    FixedRouteOracle,
    RoutePlan,
    charging_time,
    check_route,
    check_solution,
)
from data import Instance, generate_synthetic
from model import (
    DirectChargePolicy,
    HATPolicy,
    PolicyConfig,
    physical_target_socs,
)
from solver import randomized_greedy


VARIANTS = {
    "H1": PolicyConfig(history=False, deepsets=False),
    "H2": PolicyConfig(history=True, deepsets=False),
    "H3": PolicyConfig(history=False, deepsets=True),
    "H4": PolicyConfig(history=True, deepsets=True),
    "H4_open": PolicyConfig(history=True, deepsets=True, deepsets_open_only=True),
}


def _direct_features(
    instance: Instance,
    current: int,
    next_customer: int,
    soc: float,
    remaining: int,
    elapsed: float,
) -> torch.Tensor:
    node, target = instance.nodes[current], instance.nodes[next_customer]
    remaining_hours = max(0.0, instance.max_route_hours - elapsed)
    return torch.tensor(
        [
            node.x / 120.0,
            node.y / 120.0,
            target.x / 120.0,
            target.y / 120.0,
            soc / instance.battery_wh,
            instance.energy_wh[current][next_customer] / instance.battery_wh,
            remaining / max(1, len(instance.customers)),
            float(node.kind == 2),
            elapsed / instance.max_route_hours,
            remaining_hours / instance.max_route_hours,
        ],
        dtype=torch.float32,
    )


def _target_features(
    instance: Instance, arrival: float, station: int, target_soc: float
) -> torch.Tensor:
    duration = charging_time(
        instance.curves[instance.nodes[station].station_type], arrival, target_soc
    )
    return torch.tensor(
        [
            target_soc / instance.battery_wh,
            (target_soc - arrival) / instance.battery_wh,
            duration / instance.max_route_hours,
            float(instance.nodes[station].station_type == "fast"),
        ],
        dtype=torch.float32,
    )


def _station_features(instance: Instance, current: int) -> torch.Tensor:
    technology = {"fast": 0, "normal": 1, "slow": 2}
    rows = []
    for station in instance.stations:
        node = instance.nodes[station]
        one_hot = [0.0, 0.0, 0.0]
        one_hot[technology[node.station_type]] = 1.0
        rows.append(
            [
                node.x / 120.0,
                node.y / 120.0,
                instance.distances_km[current][station] / 170.0,
                instance.energy_wh[current][station] / instance.battery_wh,
                *one_hot,
                instance.curves[node.station_type].times[-1] / 2.04,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def direct_examples(
    instance: Instance, customer_routes: tuple[tuple[int, ...], ...]
) -> list[tuple]:
    """Extract GO/CHARGE, station, and absolute target-SOC labels from oracle plans."""
    oracle = FixedRouteOracle(instance)
    examples = []
    for customers in customer_routes:
        plan = oracle.solve(customers)
        if not plan.feasible:
            continue
        current = instance.depot
        soc = instance.battery_wh
        elapsed = 0.0
        customer_position = 0
        for node, amount in plan.stops[1:]:
            next_customer = (
                customers[customer_position]
                if customer_position < len(customers)
                else instance.depot
            )
            state = _direct_features(
                instance,
                current,
                next_customer,
                soc,
                len(customers) - customer_position,
                elapsed,
            )
            stations = _station_features(instance, current)
            travel = instance.travel_hours[current][node]
            if instance.nodes[node].kind == 2:
                arrival = soc - instance.energy_wh[current][node]
                target_soc = arrival + float(amount)
                required_target = instance.energy_wh[node][next_customer] + (
                    0.0
                    if next_customer == instance.depot
                    else min(
                        instance.energy_wh[next_customer][other]
                        for other in (instance.depot, *instance.stations)
                    )
                )
                candidates = physical_target_socs(
                    instance, arrival, node, required_target
                )
                quantity = min(
                    range(len(candidates)),
                    key=lambda index: abs(candidates[index] - target_soc),
                )
                target_feats = torch.stack(
                    [
                        _target_features(instance, arrival, node, candidate)
                        for candidate in candidates
                    ]
                )
                examples.append(
                    (
                        state,
                        stations,
                        1,
                        instance.stations.index(node),
                        quantity,
                        target_feats,
                    )
                )
                charge_dur = charging_time(
                    instance.curves[instance.nodes[node].station_type],
                    arrival,
                    target_soc,
                )
                elapsed += travel + charge_dur
                soc = target_soc
            else:
                examples.append((state, stations, 0, -1, -1, None))
                elapsed += travel + instance.service_times[node]
                soc -= instance.energy_wh[current][node]
                if node != instance.depot:
                    customer_position += 1
            current = node
    return examples


def _onward_energy_need(instance: Instance, node: int) -> float:
    if node == instance.depot:
        return 0.0
    return min(
        instance.energy_wh[node][other]
        for other in (instance.depot, *instance.stations)
    )


def _station_supports_next(
    instance: Instance, station: int, next_customer: int, full_soc: float
) -> bool:
    """True if a full battery at station can reach next, possibly via one more CS."""
    need = instance.energy_wh[station][next_customer] + _onward_energy_need(
        instance, next_customer
    )
    if need <= full_soc + 1e-7:
        return True
    for relay in instance.stations:
        if relay == station:
            continue
        if instance.energy_wh[station][relay] > full_soc + 1e-7:
            continue
        if (
            instance.energy_wh[relay][next_customer]
            + _onward_energy_need(instance, next_customer)
            <= full_soc + 1e-7
        ):
            return True
    return False


def decode_direct(
    model: DirectChargePolicy, instance: Instance, customers: tuple[int, ...]
) -> dict:
    """Execute the learned CHARGE policy without post-optimization."""
    current = instance.depot
    soc = instance.battery_wh
    elapsed = 0.0
    customer_position = 0
    stops: list[tuple[int, float | None]] = [(instance.depot, None)]
    max_decisions = 4 * (len(customers) + 1)
    reason = ""
    start = time.perf_counter()
    for _ in range(max_decisions):
        if customer_position == len(customers) and current == instance.depot:
            break
        next_customer = (
            customers[customer_position]
            if customer_position < len(customers)
            else instance.depot
        )
        state_features = _direct_features(
            instance,
            current,
            next_customer,
            soc,
            len(customers) - customer_position,
            elapsed,
        ).to(model.device)
        stations = _station_features(instance, current).to(model.device)
        state = model.encode(state_features)
        action_logits = model.action_head(state)
        energy_to_next = instance.energy_wh[current][next_customer]
        travel_to_next = instance.travel_hours[current][next_customer]
        service_next = instance.service_times[next_customer]
        arrival_at_next = soc - energy_to_next
        remaining_hours = instance.max_route_hours - elapsed
        if next_customer == instance.depot:
            onward_safe = True
            time_ok_go = travel_to_next <= remaining_hours + 1e-9
        else:
            onward_safe = arrival_at_next + 1e-7 >= _onward_energy_need(
                instance, next_customer
            )
            # Optimistic time lower bound: travel+service; return may include charging.
            time_ok_go = travel_to_next + service_next <= remaining_hours + 1e-9
        go_feasible = (
            energy_to_next <= soc + 1e-7 and onward_safe and time_ok_go
        )
        reachable = []
        for station in instance.stations:
            energy_to_station = instance.energy_wh[current][station]
            energy_ok = (
                energy_to_station <= soc + 1e-7
                and station != current
                and _station_supports_next(
                    instance, station, next_customer, instance.battery_wh
                )
            )
            if not energy_ok:
                reachable.append(False)
                continue
            arrival = soc - energy_to_station
            required_target = min(
                instance.battery_wh,
                instance.energy_wh[station][next_customer]
                + _onward_energy_need(instance, next_customer),
            )
            # If one hop is insufficient, charge toward full for a relay hop.
            if required_target > instance.battery_wh + 1e-7 or not (
                instance.energy_wh[station][next_customer]
                + _onward_energy_need(instance, next_customer)
                <= instance.battery_wh + 1e-7
            ):
                required_target = instance.battery_wh
            candidates = physical_target_socs(
                instance, arrival, station, required_target
            )
            travel_cs = instance.travel_hours[current][station]
            time_feasible = False
            for target_soc in candidates:
                if target_soc + 1e-7 < min(required_target, arrival + 1.0):
                    continue
                charge_dur = charging_time(
                    instance.curves[instance.nodes[station].station_type],
                    arrival,
                    target_soc,
                )
                after = (
                    travel_cs
                    + charge_dur
                    + instance.travel_hours[station][next_customer]
                    + instance.service_times[next_customer]
                )
                if after <= remaining_hours + 1e-9:
                    time_feasible = True
                    break
            reachable.append(time_feasible)
        action_mask = torch.tensor(
            [go_feasible, any(reachable)], dtype=torch.bool, device=model.device
        )
        if not bool(action_mask.any()):
            reason = "no feasible GO or CHARGE action"
            break
        action = int(action_logits.masked_fill(~action_mask, float("-inf")).argmax())
        if action == 0:
            elapsed += travel_to_next + service_next
            soc -= instance.energy_wh[current][next_customer]
            current = next_customer
            stops.append((current, None))
            if current != instance.depot:
                customer_position += 1
            continue

        station_logits = model.station_logits(state, stations)
        station_mask = torch.tensor(reachable, dtype=torch.bool, device=model.device)
        station_index = int(
            station_logits.masked_fill(~station_mask, float("-inf")).argmax()
        )
        station = instance.stations[station_index]
        arrival = soc - instance.energy_wh[current][station]
        direct_need = instance.energy_wh[station][next_customer] + _onward_energy_need(
            instance, next_customer
        )
        required_target = (
            min(instance.battery_wh, direct_need)
            if direct_need <= instance.battery_wh + 1e-7
            else instance.battery_wh
        )
        candidates = physical_target_socs(
            instance, arrival, station, required_target
        )
        target_feats = torch.stack(
            [
                _target_features(instance, arrival, station, candidate)
                for candidate in candidates
            ]
        ).to(model.device)
        quantity_logits = model.quantity_logits(
            state, stations[station_index], target_feats
        )
        travel_cs = instance.travel_hours[current][station]
        quantity_mask = []
        charge_durs = []
        for target_soc in candidates:
            if target_soc <= arrival + 1e-7:
                quantity_mask.append(False)
                charge_durs.append(float("inf"))
                continue
            if (
                direct_need <= instance.battery_wh + 1e-7
                and target_soc + 1e-7 < required_target
            ):
                quantity_mask.append(False)
                charge_durs.append(float("inf"))
                continue
            charge_dur = charging_time(
                instance.curves[instance.nodes[station].station_type],
                arrival,
                target_soc,
            )
            after = (
                travel_cs
                + charge_dur
                + instance.travel_hours[station][next_customer]
                + instance.service_times[next_customer]
            )
            quantity_mask.append(after <= remaining_hours + 1e-9)
            charge_durs.append(charge_dur)
        quantity_mask_t = torch.tensor(
            quantity_mask, dtype=torch.bool, device=model.device
        )
        if not bool(quantity_mask_t.any()):
            reason = "no positive target-SOC action"
            break
        masked_logits = quantity_logits.masked_fill(~quantity_mask_t, float("-inf"))
        if remaining_hours < 3.0:
            # Prefer the shortest feasible charge when time is scarce.
            best = None
            for index, ok in enumerate(quantity_mask):
                if not ok:
                    continue
                if best is None or charge_durs[index] < charge_durs[best] - 1e-12:
                    best = index
            quantity = int(best)
        else:
            quantity = int(masked_logits.argmax())
        target_soc = candidates[quantity]
        amount = target_soc - arrival
        charge_dur = charging_time(
            instance.curves[instance.nodes[station].station_type], arrival, target_soc
        )
        elapsed += travel_cs + charge_dur
        current, soc = station, target_soc
        stops.append((station, amount))
    concrete = tuple(stops)
    check = check_route(instance, concrete, customers)
    return {
        "feasible": check.feasible,
        "objective": check.duration_hours if check.feasible else float("inf"),
        "stops": concrete,
        "reason": reason or check.reason,
        "runtime": time.perf_counter() - start,
    }


def train_direct(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = make_teacher_data(
        args.size, args.train_instances, 30000 + args.seed * 100
    )
    examples = [
        example
        for instance, routes in teacher
        for example in direct_examples(instance, routes)
    ]
    model = DirectChargePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        example = examples[(step - 1) % len(examples)]
        state_features, station_features, action, station, quantity, target_feats = (
            example
        )
        state_features = state_features.to(device)
        station_features = station_features.to(device)
        state = model.encode(state_features)
        loss = F.cross_entropy(
            model.action_head(state).unsqueeze(0),
            torch.tensor([action], device=device),
        )
        if action == 1:
            loss = loss + F.cross_entropy(
                model.station_logits(state, station_features).unsqueeze(0),
                torch.tensor([station], device=device),
            )
            loss = loss + F.cross_entropy(
                model.quantity_logits(
                    state, station_features[station], target_feats.to(device)
                ).unsqueeze(0),
                torch.tensor([quantity], device=device),
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    version = getattr(args, "version", "c2")
    checkpoint = (
        args.checkpoints / f"CHARGE_n{args.size}_seed{args.seed}_{version}.pt"
    )
    torch.save(
        {"state_dict": model.state_dict(), "size": args.size, "version": version},
        checkpoint,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    output = (
        args.results / f"training_CHARGE_n{args.size}_seed{args.seed}_{version}.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_s", "device"])
        writer.writeheader()
        writer.writerow(
            {
                "step": args.steps,
                "loss": sum(losses[-min(50, len(losses)) :]) / min(50, len(losses)),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
            }
        )
    print(
        f"trained CHARGE on {len(examples)} expert actions; "
        f"final loss={losses[-1]:.6f}; saved {checkpoint}"
    )


def route_state(instance: Instance, customers: tuple[int, ...], plan: RoutePlan) -> dict:
    check = check_route(instance, plan.stops, customers)
    if not check.feasible:
        raise RuntimeError(check.reason)
    return {
        "customers": customers,
        "plan": plan,
        "duration": plan.duration_hours,
        "terminal_soc": check.end_soc_wh,
        "closed": False,
    }


def make_teacher_data(
    size: int,
    count: int,
    seed_start: int,
    teacher: str = "basic",
    generator_version: str = "v2",
) -> list[tuple[Instance, tuple[tuple[int, ...], ...]]]:
    """Build imitation teachers on synthetic instances.

    teacher='basic': Cycle-3 style (3 restarts, 1 basic LS pass, budget 800).
    teacher='strong': more diverse restarts + strong LS (exact FRVCP).
    """
    data = []
    candidate_seed = seed_start
    if teacher == "strong":
        restarts, passes, budget, operators = 8, 3, 2500, "strong"
    else:
        restarts, passes, budget, operators = 3, 1, 800, "basic"
    while len(data) < count and candidate_seed < seed_start + count * 60:
        instance = generate_synthetic(
            size, candidate_seed, generator_version=generator_version
        )
        result = randomized_greedy(
            instance,
            restarts=restarts,
            seed=candidate_seed,
            local_search_passes=passes,
            local_search_budget=budget,
            local_operators=operators,
        )
        if result.feasible and len(result.customer_routes) >= 1:
            data.append((instance, result.customer_routes))
        candidate_seed += 1
    if len(data) < count:
        raise RuntimeError(
            f"Only generated {len(data)}/{count} feasible teacher instances"
        )
    return data


def interleaved_actions(
    teacher_routes: tuple[tuple[int, ...], ...], rng: random.Random
) -> list[tuple[str, int | None, int | None]]:
    """Build a valid interleaved construction: preserve per-route order.

    Actions are ('extend', route_id, customer), ('close', route_id, None).
    Starting a route is the first extend of an unstarted expert route.
    """
    remaining = [list(route) for route in teacher_routes]
    started = [False] * len(teacher_routes)
    open_ids: set[int] = set()
    actions: list[tuple[str, int | None, int | None]] = []
    while any(remaining[i] for i in range(len(remaining))) or open_ids:
        extendable = [i for i in range(len(remaining)) if remaining[i]]
        closable = [i for i in open_ids if not remaining[i]]
        choices: list[tuple[str, int]] = []
        for route_id in extendable:
            choices.append(("extend", route_id))
        for route_id in closable:
            choices.append(("close", route_id))
        if not choices:
            break
        kind, route_id = rng.choice(choices)
        if kind == "close":
            actions.append(("close", route_id, None))
            open_ids.remove(route_id)
            continue
        customer = remaining[route_id].pop(0)
        actions.append(("extend", route_id, customer))
        started[route_id] = True
        open_ids.add(route_id)
    for route_id in sorted(open_ids):
        actions.append(("close", route_id, None))
    return actions


def expert_consistent_actions(
    teacher_routes: tuple[tuple[int, ...], ...],
    remaining: list[list[int]],
    live_index: dict[int, int],
) -> list[tuple[str, int, int | None]]:
    """All expert-consistent next actions at the current imitation state."""
    actions: list[tuple[str, int, int | None]] = []
    for expert_id, rem in enumerate(remaining):
        if rem:
            actions.append(("extend", expert_id, rem[0]))
        elif expert_id in live_index:
            actions.append(("close", expert_id, None))
    return actions


def _action_duration_delta(
    oracle: FixedRouteOracle,
    routes: list[dict],
    live_index: dict[int, int],
    kind: str,
    expert_id: int,
    customer: int | None,
) -> float:
    """Marginal full-route duration change for an expert-consistent action."""
    if kind == "close":
        return 0.0
    assert customer is not None
    if expert_id not in live_index:
        plan = oracle.solve((customer,))
        return plan.duration_hours if plan.feasible else float("inf")
    active = live_index[expert_id]
    prefix = routes[active]["customers"]
    before = routes[active]["duration"]
    plan = oracle.solve((*prefix, customer))
    if not plan.feasible:
        return float("inf")
    return plan.duration_hours - before


def teacher_loss(
    model: HATPolicy,
    instance: Instance,
    teacher_routes: tuple[tuple[int, ...], ...],
    rng: random.Random | None = None,
    multi_target: bool = False,
    fleet_counterfactual: bool = False,
) -> tuple[torch.Tensor, int, dict]:
    """Imitation loss on an interleaved multi-open-route trajectory.

    multi_target=False: classic single-path cross-entropy (Cycle-3).
    multi_target=True: -log sum of hierarchical probs over all expert-consistent
    next actions, then advance along one randomly chosen consistent action.
    fleet_counterfactual=True: when multiple open expert routes compete, rank
    consistent actions by FRVCP duration delta and CE on the lowest-delta action
    (plus a soft ranking term over route-choice probs). Creates a DeepSets-relevant
    credit signal that plain BC lacks.
    """
    rng = rng or random.Random(0)
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = []
    live_index: dict[int, int] = {}
    remaining = [list(route) for route in teacher_routes]
    unserved = set(instance.customers)
    losses = []
    decisions = 0
    multi_open_states = 0
    competing_states = 0
    total_states = 0
    multi_valid_states = 0
    fleet_cf_states = 0

    while any(remaining) or live_index:
        consistent = expert_consistent_actions(teacher_routes, remaining, live_index)
        if not consistent:
            break
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved
        )
        total_states += 1
        if len(open_indices) >= 2:
            multi_open_states += 1
        if len(consistent) > 1:
            multi_valid_states += 1
        route_scores = model.route_logits(route_vectors, open_indices, context)
        route_log_probs = F.log_softmax(route_scores, dim=0)

        joint_log_probs = []
        action_deltas = []
        state_competing = False
        for kind, expert_id, customer in consistent:
            action_deltas.append(
                _action_duration_delta(
                    oracle, routes, live_index, kind, expert_id, customer
                )
            )
            if kind == "extend":
                assert customer is not None
                if expert_id not in live_index:
                    route_pos = len(open_indices)
                    selected_vector = None
                    last_node = instance.depot
                    prefix: tuple[int, ...] = ()
                    is_new = True
                else:
                    active_index = live_index[expert_id]
                    route_pos = open_indices.index(active_index)
                    selected_vector = route_vectors[active_index]
                    prefix = routes[active_index]["customers"]
                    last_node = prefix[-1]
                    is_new = False
                order = sorted(unserved)
                customer_scores = model.customer_logits(
                    instance,
                    nodes,
                    selected_vector,
                    last_node,
                    context,
                    order,
                    allow_close=not is_new,
                )
                masks = [
                    oracle.solve((*prefix, candidate)).feasible for candidate in order
                ]
                if not is_new:
                    masks.append(True)
                if sum(masks[: len(order)]) >= 2 and len(open_indices) >= 1:
                    state_competing = True
                masked = customer_scores.masked_fill(
                    ~torch.tensor(masks, dtype=torch.bool, device=model.device),
                    float("-inf"),
                )
                cust_log = F.log_softmax(masked, dim=0)
                cust_pos = order.index(customer)
                joint_log_probs.append(route_log_probs[route_pos] + cust_log[cust_pos])
            else:
                active_index = live_index[expert_id]
                route_pos = open_indices.index(active_index)
                order = sorted(unserved)
                prefix = routes[active_index]["customers"]
                customer_scores = model.customer_logits(
                    instance,
                    nodes,
                    route_vectors[active_index],
                    prefix[-1] if prefix else instance.depot,
                    context,
                    order,
                    allow_close=True,
                )
                masks = [
                    oracle.solve((*prefix, candidate)).feasible for candidate in order
                ]
                masks.append(True)
                masked = customer_scores.masked_fill(
                    ~torch.tensor(masks, dtype=torch.bool, device=model.device),
                    float("-inf"),
                )
                cust_log = F.log_softmax(masked, dim=0)
                joint_log_probs.append(route_log_probs[route_pos] + cust_log[len(order)])

        # Count each decision state at most once (never >1 competing_frac).
        if state_competing or len(consistent) > 1:
            competing_states += 1

        use_fleet_cf = (
            fleet_counterfactual
            and len(open_indices) >= 2
            and len(consistent) >= 2
            and any(math.isfinite(d) for d in action_deltas)
        )
        if multi_target and not use_fleet_cf:
            losses.append(-torch.logsumexp(torch.stack(joint_log_probs), dim=0))
            kind, expert_id, customer = rng.choice(consistent)
        elif use_fleet_cf:
            fleet_cf_states += 1
            finite = [
                (i, d)
                for i, d in enumerate(action_deltas)
                if math.isfinite(d)
            ]
            finite.sort(key=lambda item: item[1])
            best_i = finite[0][0]
            # Hard CE on best duration-delta expert action.
            losses.append(-joint_log_probs[best_i])
            # Soft ranking: prefer lower-delta actions among consistent set.
            if len(finite) >= 2:
                rank_logits = torch.stack([joint_log_probs[i] for i, _ in finite])
                # Target distribution: softmax(-delta / temperature).
                temps = torch.tensor(
                    [-d / 0.25 for _, d in finite],
                    dtype=rank_logits.dtype,
                    device=rank_logits.device,
                )
                target = F.softmax(temps, dim=0)
                losses.append(-(target * F.log_softmax(rank_logits, dim=0)).sum())
            kind, expert_id, customer = consistent[best_i]
        else:
            # Single-target: pick one consistent action and CE on that pair.
            kind, expert_id, customer = rng.choice(consistent)
            chosen = None
            for index, action in enumerate(consistent):
                if action == (kind, expert_id, customer):
                    chosen = joint_log_probs[index]
                    break
            assert chosen is not None
            losses.append(-chosen)

        # Advance environment along the selected expert-consistent action.
        if kind == "extend":
            assert customer is not None
            if expert_id not in live_index:
                prefix = ()
                plan = oracle.solve((customer,))
                routes.append(route_state(instance, (customer,), plan))
                live_index[expert_id] = len(routes) - 1
            else:
                active_index = live_index[expert_id]
                prefix = routes[active_index]["customers"]
                plan = oracle.solve((*prefix, customer))
                routes[active_index] = route_state(
                    instance, (*prefix, customer), plan
                )
            remaining[expert_id].pop(0)
            unserved.remove(customer)
        else:
            routes[live_index[expert_id]]["closed"] = True
            del live_index[expert_id]
        decisions += 1

    stats = {
        "multi_open_frac": multi_open_states / max(1, total_states),
        "competing_frac": competing_states / max(1, total_states),
        "multi_valid_frac": multi_valid_states / max(1, total_states),
        "fleet_cf_frac": fleet_cf_states / max(1, total_states),
        "total_states": total_states,
    }
    return torch.stack(losses).mean(), decisions, stats


def _partial_after_destroy(
    instance: Instance,
    routes: list[tuple[int, ...]],
    plans: list,
    removed: set[int],
    oracle: FixedRouteOracle | None = None,
    *,
    keep_open: bool = True,
) -> tuple[list[dict], list] | None:
    """Rebuild kept routes after removing customers for neural repair.

    Surviving customer order is preserved. When ``keep_open`` is True (default),
    routes stay eligible for extension so the policy can re-insert removed
    customers into existing EV routes or open new ones. Charging is recomputed
    via frvcpy for each surviving sequence.
    """
    oracle = oracle or FixedRouteOracle(instance)
    kept_routes: list[dict] = []
    kept_plans: list = []
    for route, _plan in zip(routes, plans):
        kept = tuple(c for c in route if c not in removed)
        if not kept:
            continue
        rebuilt = oracle.solve(kept)
        if rebuilt is None or not rebuilt.feasible:
            return None
        state = route_state(instance, kept, rebuilt)
        # CRITICAL: leave open so decode() can extend existing routes.
        state["closed"] = not keep_open
        kept_routes.append(state)
        kept_plans.append(rebuilt)
    if not kept_routes and removed != {c for r in routes for c in r}:
        return None
    return kept_routes, kept_plans


def _accept_repaired(
    instance: Instance,
    repaired: dict,
    best_obj: float,
    fallback_routes: list[tuple[int, ...]],
    fallback_plans: list,
) -> tuple[list[tuple[int, ...]], list, float, bool]:
    """Accept neural repair only if feasible, covering, and strictly better."""
    if not repaired.get("feasible"):
        return fallback_routes, fallback_plans, best_obj, False
    cand_routes = [tuple(r) for r in repaired["routes"]]
    cand_plans = list(repaired["plans"])
    valid, _ = check_solution(instance, cand_plans)
    if not valid:
        return fallback_routes, fallback_plans, best_obj, False
    obj = sum(p.duration_hours for p in cand_plans)
    if obj < best_obj - 1e-10:
        return cand_routes, cand_plans, obj, True
    return fallback_routes, fallback_plans, best_obj, False


def neural_lns(
    model: HATPolicy,
    instance: Instance,
    routes: list[tuple[int, ...]],
    plans: list,
    max_iterations: int = 6,
    seed: int = 0,
    cost_lambda: float = 4.0,
    destroy_min: int = 2,
    destroy_max: int = 4,
    seed_classical: bool = False,
) -> tuple[list[tuple[int, ...]], list, int, dict]:
    """Ruin + insertion-capable neural-policy repair (search-labeled).

    Surviving routes stay open so the hierarchical decoder may append removed
    customers onto existing routes (preserving relative order) or open new
    routes. Falls back to the incumbent if repair fails or worsens the objective.
    """
    rng = random.Random(seed)
    oracle = FixedRouteOracle(instance)
    best_routes = [tuple(r) for r in routes]
    best_plans = list(plans)
    best_obj = sum(p.duration_hours for p in best_plans)
    improvements = 0
    if seed_classical:
        from solver_ls import ruin_and_repair

        best_routes, best_plans, classical_imp, _ = ruin_and_repair(
            best_routes,
            best_plans,
            oracle,
            instance,
            max_iterations=2,
            evaluation_budget=200,
            seed=seed,
            destroy_min=destroy_min,
            destroy_max=destroy_max,
        )
        improvements += classical_imp
        best_obj = sum(p.duration_hours for p in best_plans)

    for it in range(max_iterations):
        customers = [c for r in best_routes for c in r]
        if len(customers) < 2:
            break
        k = min(len(customers), rng.randint(destroy_min, destroy_max))
        removed = set(rng.sample(customers, k))
        partial = _partial_after_destroy(
            instance, best_routes, best_plans, removed, oracle, keep_open=True
        )
        if partial is None:
            continue
        kept_routes, _ = partial
        assert all(not r["closed"] for r in kept_routes)
        repaired = decode(
            model,
            instance,
            sample=True,
            seed=seed + 17 + it,
            cost_lambda=cost_lambda,
            initial_routes=copy.deepcopy(kept_routes),
            only_customers=set(removed),
        )
        best_routes, best_plans, best_obj, improved = _accept_repaired(
            instance, repaired, best_obj, best_routes, best_plans
        )
        if improved:
            improvements += 1
    return (
        best_routes,
        best_plans,
        improvements,
        {"schedule": "neural_lns_search", "improvements": improvements},
    )


def hybrid_lns(
    model: HATPolicy,
    instance: Instance,
    routes: list[tuple[int, ...]],
    plans: list,
    max_iterations: int = 6,
    seed: int = 0,
    cost_lambda: float = 4.0,
    destroy_min: int = 2,
    destroy_max: int = 4,
    classical_iterations: int = 6,
    classical_budget: int = 600,
) -> tuple[list[tuple[int, ...]], list, int, dict]:
    """Classical ruin first, then insertion-capable neural repair; keep best.

    Search-labeled. Classical ruin establishes a floor under a matched budget;
    neural attempts may only improve further via open-route reinsertion.
    """
    from solver_ls import ruin_and_repair

    rng = random.Random(seed)
    oracle = FixedRouteOracle(instance)
    best_routes = [tuple(r) for r in routes]
    best_plans = list(plans)
    best_routes, best_plans, classical_imp, _ = ruin_and_repair(
        best_routes,
        best_plans,
        oracle,
        instance,
        max_iterations=classical_iterations,
        evaluation_budget=classical_budget,
        seed=seed,
        destroy_min=destroy_min,
        destroy_max=destroy_max,
    )
    improvements = classical_imp
    best_obj = sum(p.duration_hours for p in best_plans)
    neural_imp = 0

    for it in range(max_iterations):
        customers = [c for r in best_routes for c in r]
        if len(customers) < 2:
            break
        k = min(len(customers), rng.randint(destroy_min, destroy_max))
        removed = set(rng.sample(customers, k))
        partial = _partial_after_destroy(
            instance, best_routes, best_plans, removed, oracle, keep_open=True
        )
        if partial is None:
            continue
        kept_routes, _ = partial
        repaired = decode(
            model,
            instance,
            sample=True,
            seed=seed + 31 + it,
            cost_lambda=cost_lambda,
            initial_routes=copy.deepcopy(kept_routes),
            only_customers=set(removed),
        )
        best_routes, best_plans, best_obj, improved = _accept_repaired(
            instance, repaired, best_obj, best_routes, best_plans
        )
        if improved:
            improvements += 1
            neural_imp += 1
    return (
        best_routes,
        best_plans,
        improvements,
        {
            "schedule": "hybrid_lns_search",
            "improvements": improvements,
            "neural_improvements": neural_imp,
            "classical_improvements": classical_imp,
        },
    )


def repair_reinforce_step(
    model: HATPolicy,
    instance: Instance,
    optimizer: torch.optim.Optimizer,
    k_repairs: int = 4,
    seed: int = 0,
    cost_lambda: float = 4.0,
    destroy_min: int = 2,
    destroy_max: int = 4,
) -> tuple[torch.Tensor, dict]:
    """POMO-style REINFORCE on the repair task after random destroy.

    Builds a greedy construction, destroys a customer subset, then samples
    K repair trajectories with a shared mean baseline.
    """
    model.eval()
    with torch.no_grad():
        base = decode(
            model, instance, sample=False, seed=seed, cost_lambda=cost_lambda
        )
    if not base["feasible"]:
        return torch.tensor(0.0), base
    rng = random.Random(seed + 7)
    routes = [tuple(r) for r in base["routes"]]
    plans = list(base["plans"])
    customers = [c for r in routes for c in r]
    if len(customers) < 2:
        return torch.tensor(0.0), base
    k = min(len(customers), rng.randint(destroy_min, destroy_max))
    removed = set(rng.sample(customers, k))
    partial = _partial_after_destroy(instance, routes, plans, removed)
    if partial is None:
        return torch.tensor(0.0), base
    kept_routes, _ = partial
    base_obj = sum(p.duration_hours for p in plans)

    model.train()
    rollouts = []
    for index in range(max(1, k_repairs)):
        result = decode(
            model,
            instance,
            sample=True,
            seed=seed * 100 + index,
            cost_lambda=cost_lambda,
            initial_routes=copy.deepcopy(kept_routes),
            only_customers=set(removed),
        )
        if result["feasible"]:
            # Reward improvement over the pre-destroy solution.
            reward = base_obj - result["objective"]
        else:
            reward = -50.0
        rollouts.append((result, reward))
    rewards = [reward for _, reward in rollouts]
    baseline = sum(rewards) / max(1, len(rewards))
    optimizer.zero_grad()
    losses = []
    for result, reward in rollouts:
        if result["log_prob"] is None:
            continue
        advantage = reward - baseline
        if abs(advantage) < 1e-9:
            continue
        losses.append(-advantage * result["log_prob"])
    if losses:
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        loss = torch.tensor(0.0)
    best = max(rollouts, key=lambda item: item[1])[0]
    return loss, best


def pomo_reinforce_step(
    model: HATPolicy,
    instance: Instance,
    optimizer: torch.optim.Optimizer,
    k: int = 8,
    seed: int = 0,
    cost_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """POMO-style multi-rollout REINFORCE with shared mean baseline.

    Forces diverse first customers (up to K) then samples the rest of the
    hierarchical trajectory. Reward = -full duration if feasible else -100.
    """
    customers = list(instance.customers)
    rng = random.Random(seed)
    starts = customers[:]
    rng.shuffle(starts)
    starts = starts[: max(1, min(k, len(starts)))]
    rollouts = []
    model.train()
    for index, first in enumerate(starts):
        result = decode(
            model,
            instance,
            sample=True,
            seed=seed * 100 + index,
            cost_lambda=cost_lambda,
            force_first_customer=first,
        )
        reward = -result["objective"] if result["feasible"] else -100.0
        rollouts.append((result, reward))
    rewards = [reward for _, reward in rollouts]
    baseline = sum(rewards) / max(1, len(rewards))
    optimizer.zero_grad()
    losses = []
    for result, reward in rollouts:
        if result["log_prob"] is None:
            continue
        advantage = reward - baseline
        if abs(advantage) < 1e-9:
            continue
        losses.append(-advantage * result["log_prob"])
    if losses:
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        loss = torch.tensor(0.0)
    best = min(
        (r for r, rew in rollouts if r["feasible"]),
        key=lambda r: r["objective"],
        default=rollouts[0][0],
    )
    return loss, best


def diagnose_deepsets(model: HATPolicy, instance: Instance) -> dict:
    """Compact checks that DeepSets is permutation-invariant and informative."""
    if not model.config.deepsets:
        return {"deepsets": False}
    oracle = FixedRouteOracle(instance)
    routes = []
    for customer in instance.customers[:2]:
        plan = oracle.solve((customer,))
        if plan.feasible:
            routes.append(route_state(instance, (customer,), plan))
    if len(routes) < 2:
        return {"deepsets": True, "skipped": True}
    unserved = set(instance.customers) - {routes[0]["customers"][0], routes[1]["customers"][0]}
    model.zero_grad(set_to_none=True)
    _, route_vectors, open_indices, context = model.encode_state(
        instance, routes, unserved
    )
    logits = model.route_logits(route_vectors, open_indices, context)
    loss = logits.sum()
    loss.backward()
    phi_grad = 0.0
    for parameter in model.fleet_phi.parameters():
        if parameter.grad is not None:
            phi_grad += float(parameter.grad.abs().sum())
    with torch.no_grad():
        _, route_perm, open_perm, context_perm = model.encode_state(
            instance, list(reversed(routes)), unserved
        )
        fleet_a = model.fleet_rho(model.fleet_phi(route_vectors).sum(dim=0))
        fleet_b = model.fleet_rho(model.fleet_phi(route_perm).sum(dim=0))
        perm_error = float((fleet_a - fleet_b).abs().max())
        routes_perturbed = copy.deepcopy(routes)
        routes_perturbed[1]["duration"] = min(
            instance.max_route_hours, routes_perturbed[1]["duration"] + 1.5
        )
        _, rv2, oi2, ctx2 = model.encode_state(instance, routes_perturbed, unserved)
        logits2 = model.route_logits(rv2, oi2, ctx2)
        active_delta = float((logits - logits2).abs().max())
    return {
        "deepsets": True,
        "phi_grad": phi_grad,
        "perm_error": perm_error,
        "active_logit_delta": active_delta,
    }


def decode(
    model: HATPolicy,
    instance: Instance,
    sample: bool = False,
    seed: int = 0,
    local_passes: int = 0,
    local_budget: int = 0,
    cost_lambda: float = 0.0,
    route_lambda: float = 0.0,
    local_operators: str = "basic",
    route_cost_top_k: int = 3,
    ruin_iterations: int = 0,
    ruin_budget: int = 0,
    force_first_customer: int | None = None,
    initial_routes: list[dict] | None = None,
    only_customers: set[int] | None = None,
) -> dict:
    """Greedy/sampled hierarchical decode with optional oracle-aware scoring.

    When ``cost_lambda > 0``, customer pointer logits are adjusted by
    ``logit - cost_lambda * (D(R⊕j) - D(R)) / T_max``, using FRVCP durations
    already required for feasibility masks (cache hits; no redundant solves).

    When ``route_lambda > 0``, route/new-route logits are biased by the best
    feasible one-customer extension potential among a shortlist of top learned
    route candidates (plus NEW_ROUTE). ``route_lambda=0`` and ``cost_lambda=0``
    preserve Cycle-3 decode behavior.
    """
    from solver import local_search

    start = time.perf_counter()
    if sample:
        torch.manual_seed(seed)
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = list(initial_routes) if initial_routes else []
    if only_customers is not None:
        unserved = set(only_customers)
    else:
        served = {c for route in routes for c in route["customers"]}
        unserved = set(instance.customers) - served
    log_probs: list[torch.Tensor] = []
    entropy = 0.0
    reason = ""
    max_decisions = 3 * len(instance.customers) + 5
    nodes = model.encode_nodes(instance)
    history_cache: dict = {}
    singleton_plans = {
        customer: oracle.solve((customer,)) for customer in instance.customers
    }
    singleton_ok = {
        customer: plan.feasible for customer, plan in singleton_plans.items()
    }
    forced_first_pending = force_first_customer is not None and force_first_customer in (
        unserved if only_customers is not None else set(instance.customers)
    )

    for decision in range(max_decisions):
        if not unserved and all(route["closed"] for route in routes):
            break
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved, nodes=nodes, history_cache=history_cache
        )
        route_scores = model.route_logits(route_vectors, open_indices, context)
        new_feasible = bool(unserved) and any(
            singleton_ok[customer] for customer in unserved
        )
        route_mask = [True] * len(open_indices) + [new_feasible]
        if forced_first_pending and not routes:
            # POMO-style: force NEW_ROUTE then the designated first customer.
            route_mask = [False] * len(open_indices) + [new_feasible]
        if not any(route_mask):
            reason = "no route action can serve remaining customers"
            break
        route_scores = route_scores.masked_fill(
            ~torch.tensor(route_mask, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        if route_lambda != 0.0 and unserved:
            scale = max(instance.max_route_hours, 1e-6)
            adjusted_routes = route_scores.clone()
            finite = [
                index
                for index in range(len(open_indices) + 1)
                if bool(route_mask[index])
            ]
            # Shortlist: top-k open routes by learned score, always include NEW_ROUTE if feasible.
            open_finite = [index for index in finite if index < len(open_indices)]
            open_finite.sort(
                key=lambda index: float(route_scores[index].detach()), reverse=True
            )
            shortlist = open_finite[: max(0, route_cost_top_k)]
            if new_feasible:
                shortlist.append(len(open_indices))
            for route_pos in shortlist:
                if route_pos == len(open_indices):
                    potentials = [
                        plan.duration_hours
                        for customer, plan in singleton_plans.items()
                        if customer in unserved and plan.feasible
                    ]
                    if not potentials:
                        continue
                    potential = min(potentials)
                else:
                    route_index = open_indices[route_pos]
                    prefix = routes[route_index]["customers"]
                    base = float(routes[route_index]["duration"])
                    potentials = []
                    for customer in sorted(unserved):
                        plan = oracle.solve((*prefix, customer))
                        if plan.feasible:
                            potentials.append(plan.duration_hours - base)
                    if not potentials:
                        continue
                    potential = min(potentials)
                adjusted_routes[route_pos] = (
                    adjusted_routes[route_pos] - float(route_lambda) * potential / scale
                )
            route_scores = adjusted_routes.masked_fill(
                ~torch.tensor(route_mask, dtype=torch.bool, device=model.device),
                float("-inf"),
            )
        route_dist = torch.distributions.Categorical(logits=route_scores)
        route_action = route_dist.sample() if sample else torch.argmax(route_scores)
        log_probs.append(route_dist.log_prob(route_action))
        entropy += float(route_dist.entropy().detach())
        route_position = int(route_action)
        is_new = route_position == len(open_indices)
        selected_index = None if is_new else open_indices[route_position]
        selected_vector = None if is_new else route_vectors[selected_index]
        prefix = () if is_new else routes[selected_index]["customers"]
        last_node = instance.depot if is_new else prefix[-1]
        order = sorted(unserved)
        customer_scores = model.customer_logits(
            instance,
            nodes,
            selected_vector,
            last_node,
            context,
            order,
            allow_close=not is_new,
        )
        if is_new:
            candidate_plans = [singleton_plans[candidate] for candidate in order]
            masks = [plan.feasible for plan in candidate_plans]
            base_duration = 0.0
        else:
            candidate_plans = [
                oracle.solve((*prefix, candidate)) for candidate in order
            ]
            masks = [plan.feasible for plan in candidate_plans]
            masks.append(True)
            base_duration = float(routes[selected_index]["duration"])
        if not any(masks):
            reason = "selected route has no feasible action"
            break
        if cost_lambda != 0.0:
            adjusted = customer_scores.clone()
            scale = max(instance.max_route_hours, 1e-6)
            for index, plan in enumerate(candidate_plans):
                if not plan.feasible:
                    continue
                marginal = (plan.duration_hours - base_duration) / scale
                adjusted[index] = adjusted[index] - float(cost_lambda) * marginal
            customer_scores = adjusted
        customer_scores = customer_scores.masked_fill(
            ~torch.tensor(masks, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        if forced_first_pending and is_new and force_first_customer in order:
            # Force the designated first customer (POMO multi-start).
            forced_pos = order.index(force_first_customer)
            customer_mask = [False] * len(order)
            customer_mask[forced_pos] = True
            customer_scores = customer_scores.masked_fill(
                ~torch.tensor(customer_mask, dtype=torch.bool, device=model.device),
                float("-inf"),
            )
            forced_first_pending = False
        customer_dist = torch.distributions.Categorical(logits=customer_scores)
        customer_action = (
            customer_dist.sample() if sample else torch.argmax(customer_scores)
        )
        log_probs.append(customer_dist.log_prob(customer_action))
        entropy += float(customer_dist.entropy().detach())
        action = int(customer_action)
        if not is_new and action == len(order):
            routes[selected_index]["closed"] = True
            continue
        customer = order[action]
        plan = (
            singleton_plans[customer]
            if is_new
            else oracle.solve((*prefix, customer))
        )
        if is_new:
            routes.append(route_state(instance, (customer,), plan))
        else:
            routes[selected_index] = route_state(
                instance, (*prefix, customer), plan
            )
        unserved.remove(customer)
    else:
        reason = "decision budget exceeded"

    customer_routes = [route["customers"] for route in routes]
    plans = [route["plan"] for route in routes]
    improvements = 0
    search_diag = None
    ruin_diag = None
    if local_passes > 0 and local_budget > 0 and not unserved:
        customer_routes, plans, improvements, search_diag = local_search(
            customer_routes,
            plans,
            oracle,
            local_passes,
            local_budget,
            operators=local_operators,
        )
    if ruin_iterations > 0 and ruin_budget > 0 and not unserved and plans:
        from solver_ls import ruin_and_repair

        customer_routes, plans, ruin_improvements, ruin_diag = ruin_and_repair(
            customer_routes,
            plans,
            oracle,
            instance,
            max_iterations=ruin_iterations,
            evaluation_budget=ruin_budget,
            seed=seed + 91,
        )
        improvements += ruin_improvements
    valid, validation_reason = check_solution(instance, plans) if not unserved else (
        False,
        f"{len(unserved)} customers unserved",
    )
    if not valid:
        reason = reason or validation_reason
    from charging import literature_compatible_objective_hours

    objective = sum(plan.duration_hours for plan in plans) if valid else float("inf")
    objective_lit = (
        literature_compatible_objective_hours(instance, objective)
        if valid
        else float("inf")
    )
    return {
        "feasible": valid,
        "objective": objective,
        "objective_lit": objective_lit,
        "routes": tuple(customer_routes),
        "plans": plans,
        "runtime": time.perf_counter() - start,
        "oracle_calls": oracle.calls,
        "reason": reason,
        "log_prob": torch.stack(log_probs).sum() if log_probs else None,
        "entropy": entropy / max(1, len(log_probs)),
        "decisions": decision + 1,
        "local_improvements": improvements,
        "cost_lambda": float(cost_lambda),
        "route_lambda": float(route_lambda),
        "search_diagnostics": None if search_diag is None else search_diag.as_dict(),
        "ruin_diagnostics": None if ruin_diag is None else ruin_diag.as_dict(),
    }


def train_model(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = getattr(args, "version", "c2")
    teacher = getattr(args, "teacher", "basic")
    generator_version = getattr(args, "generator", "v2")
    multi_target = bool(getattr(args, "multi_target", False))
    fleet_counterfactual = bool(getattr(args, "fleet_counterfactual", False))
    # Fresh seed bands per generator family to avoid instance reuse across cycles.
    if generator_version == "v2c":
        train_seed_start = 920000 + args.seed * 1000
        val_seed_start = 930000 + args.seed * 1000
    else:
        train_seed_start = 700000 + args.seed * 1000
        val_seed_start = 800000 + args.seed * 1000
    train_data = make_teacher_data(
        args.size,
        args.train_instances,
        train_seed_start,
        teacher=teacher,
        generator_version=generator_version,
    )
    validation = make_teacher_data(
        args.size,
        args.val_instances,
        val_seed_start,
        teacher=teacher,
        generator_version=generator_version,
    )
    model = HATPolicy(VARIANTS[args.model]).to(device)
    init_ckpt = getattr(args, "init_checkpoint", None)
    if init_ckpt:
        init_path = Path(init_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print({"warm_start": str(init_path), "from_version": payload.get("version")})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    start = time.perf_counter()
    best_score = (-1.0, float("-inf"))
    best_state = copy.deepcopy(model.state_dict())
    multi_open_sum = 0.0
    train_rng = random.Random(args.seed + 17)
    # Checkpoint selection uses the same cost weight as intended deployment.
    val_cost_lambda = float(getattr(args, "cost_lambda", 0.0))

    def _validation_metrics() -> dict:
        model.eval()
        with torch.no_grad():
            decoded = [
                decode(
                    model,
                    val_inst,
                    sample=False,
                    cost_lambda=val_cost_lambda,
                )
                for val_inst, _ in validation
            ]
            greedy_diag = [
                decode(model, val_inst, sample=False, cost_lambda=0.0)
                for val_inst, _ in validation
            ]
        feasible = [result for result in decoded if result["feasible"]]
        mean_objective = (
            sum(result["objective"] for result in feasible) / len(feasible)
            if feasible
            else float("nan")
        )
        greedy_feas = [result for result in greedy_diag if result["feasible"]]
        greedy_obj = (
            sum(result["objective"] for result in greedy_feas) / len(greedy_feas)
            if greedy_feas
            else float("nan")
        )
        return {
            "feasible_rate": len(feasible) / len(decoded),
            "validation_objective": mean_objective,
            "entropy": sum(result["entropy"] for result in decoded) / len(decoded),
            "greedy0_objective": greedy_obj,
            "val_cost_lambda": val_cost_lambda,
        }

    diag = diagnose_deepsets(model, train_data[0][0])
    print(
        {
            "diagnostics": diag,
            "teacher": teacher,
            "generator": generator_version,
            "multi_target": multi_target,
            "fleet_counterfactual": fleet_counterfactual,
            "train_seed_start": train_seed_start,
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
            "val_cost_lambda": val_cost_lambda,
        }
    )

    for step in range(1, args.steps + 1):
        instance, routes = train_data[(step - 1) % len(train_data)]
        optimizer.zero_grad()
        loss, decisions, stats = teacher_loss(
            model,
            instance,
            routes,
            rng=train_rng,
            multi_target=multi_target,
            fleet_counterfactual=fleet_counterfactual,
        )
        multi_open_sum += stats["multi_open_frac"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = _validation_metrics()
            row = {
                "stage": "bc",
                "step": step,
                "loss": float(loss.detach()),
                "feasible_rate": metrics["feasible_rate"],
                "validation_objective": metrics["validation_objective"],
                "entropy": metrics["entropy"],
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": decisions,
                "multi_open_frac": multi_open_sum / step,
                "competing_frac": stats["competing_frac"],
                "fleet_cf_frac": stats.get("fleet_cf_frac", 0.0),
                "greedy0_objective": metrics["greedy0_objective"],
                "val_cost_lambda": metrics["val_cost_lambda"],
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -row["validation_objective"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    pomo_k = int(getattr(args, "pomo_k", 0) or 0)
    for rl_step in range(1, args.rl_steps + 1):
        instance, _ = train_data[(rl_step - 1) % len(train_data)]
        if pomo_k > 0:
            rl_loss, sampled = pomo_reinforce_step(
                model,
                instance,
                optimizer,
                k=min(pomo_k, len(instance.customers)),
                seed=args.seed + rl_step,
                cost_lambda=float(getattr(args, "cost_lambda", 0.0)),
            )
        else:
            model.eval()
            with torch.no_grad():
                greedy = decode(model, instance, sample=False, seed=args.seed + rl_step)
            model.train()
            sampled = decode(model, instance, sample=True, seed=args.seed + rl_step)
            sampled_reward = -sampled["objective"] if sampled["feasible"] else -100.0
            greedy_reward = -greedy["objective"] if greedy["feasible"] else -100.0
            advantage = sampled_reward - greedy_reward
            if sampled["log_prob"] is not None and abs(advantage) > 1e-9:
                optimizer.zero_grad()
                rl_loss = -advantage * sampled["log_prob"]
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                rl_loss = torch.tensor(0.0)
        if rl_step == args.rl_steps or (
            args.eval_every > 0 and rl_step % args.eval_every == 0
        ):
            metrics = _validation_metrics()
            row = {
                "stage": "rl_pomo" if pomo_k > 0 else "rl",
                "step": rl_step,
                "loss": float(rl_loss.detach())
                if hasattr(rl_loss, "detach")
                else float(rl_loss),
                "feasible_rate": metrics["feasible_rate"],
                "validation_objective": metrics["validation_objective"],
                "entropy": metrics["entropy"],
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": sampled["decisions"],
                "multi_open_frac": multi_open_sum / max(1, args.steps),
                "competing_frac": "",
                "fleet_cf_frac": "",
                "greedy0_objective": metrics["greedy0_objective"],
                "val_cost_lambda": metrics["val_cost_lambda"],
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -row["validation_objective"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    repair_rl_steps = int(getattr(args, "repair_rl_steps", 0) or 0)
    repair_k = int(getattr(args, "repair_k", 4) or 4)
    for repair_step in range(1, repair_rl_steps + 1):
        instance, _ = train_data[(repair_step - 1) % len(train_data)]
        rl_loss, sampled = repair_reinforce_step(
            model,
            instance,
            optimizer,
            k_repairs=min(repair_k, max(2, len(instance.customers) // 2)),
            seed=args.seed + 5000 + repair_step,
            cost_lambda=float(getattr(args, "cost_lambda", 0.0)),
        )
        if repair_step == repair_rl_steps or (
            args.eval_every > 0 and repair_step % args.eval_every == 0
        ):
            metrics = _validation_metrics()
            row = {
                "stage": "rl_repair",
                "step": repair_step,
                "loss": float(rl_loss.detach())
                if hasattr(rl_loss, "detach")
                else float(rl_loss),
                "feasible_rate": metrics["feasible_rate"],
                "validation_objective": metrics["validation_objective"],
                "entropy": metrics["entropy"],
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": sampled.get("decisions", ""),
                "multi_open_frac": multi_open_sum / max(1, args.steps),
                "competing_frac": "",
                "fleet_cf_frac": "",
                "greedy0_objective": metrics["greedy0_objective"],
                "val_cost_lambda": metrics["val_cost_lambda"],
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -row["validation_objective"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    model.load_state_dict(best_state)
    final_diag = diagnose_deepsets(model, validation[0][0])
    print({"final_diagnostics": final_diag})
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.checkpoints / f"{args.model}_n{args.size}_seed{args.seed}_{version}.pt"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(VARIANTS[args.model]),
            "model": args.model,
            "size": args.size,
            "seed": args.seed,
            "version": version,
            "teacher": teacher,
            "generator": generator_version,
            "multi_target": multi_target,
            "fleet_counterfactual": fleet_counterfactual,
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
            "repair_rl_steps": repair_rl_steps,
            "diagnostics": final_diag,
        },
        checkpoint,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.results
        / f"training_{args.model}_n{args.size}_seed{args.seed}_{version}.csv"
    )
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"saved {checkpoint} and {log_path}; total {time.perf_counter() - start:.2f}s"
    )


def load_policy(path: str | Path, device: torch.device | None = None) -> HATPolicy:
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HATPolicy(PolicyConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_direct_policy(
    path: str | Path, device: torch.device | None = None
) -> DirectChargePolicy:
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DirectChargePolicy().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*VARIANTS, "CHARGE"), required=True)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--rl-steps", type=int, default=10)
    parser.add_argument("--train-instances", type=int, default=12)
    parser.add_argument("--val-instances", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="c3")
    parser.add_argument("--teacher", choices=("basic", "strong"), default="basic")
    parser.add_argument("--generator", choices=("v1", "v2", "v2c"), default="v2")
    parser.add_argument(
        "--multi-target",
        action="store_true",
        help="Expert-consistent multi-action imitation loss",
    )
    parser.add_argument(
        "--fleet-counterfactual",
        action="store_true",
        help="Duration-delta ranking loss when multiple open expert routes compete",
    )
    parser.add_argument(
        "--pomo-k",
        type=int,
        default=0,
        help="POMO multi-rollout count during RL (0 = single-sample RL)",
    )
    parser.add_argument(
        "--cost-lambda",
        type=float,
        default=0.0,
        help="Optional cost-aware decode weight during POMO RL",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start weights from an existing checkpoint (no official XML)",
    )
    parser.add_argument(
        "--repair-rl-steps",
        type=int,
        default=0,
        help="After POMO, fine-tune with destroy/repair REINFORCE",
    )
    parser.add_argument(
        "--repair-k",
        type=int,
        default=4,
        help="Number of repair rollouts per repair-RL step",
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if args.model == "CHARGE":
        train_direct(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
