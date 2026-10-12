"""Synthetic teacher generation, H1-H4 training, and verified decoding."""

from __future__ import annotations

import argparse
import csv
import copy
from dataclasses import asdict
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
from model import DirectChargePolicy, HATPolicy, PolicyConfig
from solver import randomized_greedy


VARIANTS = {
    "H1": PolicyConfig(history=False, deepsets=False),
    "H2": PolicyConfig(history=True, deepsets=False),
    "H3": PolicyConfig(history=False, deepsets=True),
    "H4": PolicyConfig(history=True, deepsets=True),
}


def _direct_features(
    instance: Instance, current: int, next_customer: int, soc: float, remaining: int
) -> torch.Tensor:
    node, target = instance.nodes[current], instance.nodes[next_customer]
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
) -> list[tuple[torch.Tensor, torch.Tensor, int, int, int]]:
    """Extract GO/CHARGE, station, and target-SOC labels from oracle plans."""
    oracle = FixedRouteOracle(instance)
    examples = []
    targets = DirectChargePolicy.targets
    for customers in customer_routes:
        plan = oracle.solve(customers)
        if not plan.feasible:
            continue
        current = instance.depot
        soc = instance.battery_wh
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
            )
            stations = _station_features(instance, current)
            if instance.nodes[node].kind == 2:
                arrival = soc - instance.energy_wh[current][node]
                target_soc = arrival + float(amount)
                quantity = min(
                    range(len(targets)),
                    key=lambda index: abs(targets[index] - target_soc / instance.battery_wh),
                )
                examples.append(
                    (state, stations, 1, instance.stations.index(node), quantity)
                )
                soc = target_soc
            else:
                examples.append((state, stations, 0, -1, -1))
                soc -= instance.energy_wh[current][node]
                if node != instance.depot:
                    customer_position += 1
            current = node
    return examples


def decode_direct(
    model: DirectChargePolicy, instance: Instance, customers: tuple[int, ...]
) -> dict:
    """Execute the learned CHARGE policy without post-optimization."""
    current = instance.depot
    soc = instance.battery_wh
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
        ).to(model.device)
        stations = _station_features(instance, current).to(model.device)
        state = model.encode(state_features)
        action_logits = model.action_head(state)
        energy_to_next = instance.energy_wh[current][next_customer]
        arrival_at_next = soc - energy_to_next
        if next_customer == instance.depot:
            onward_safe = True
        else:
            onward_safe = arrival_at_next + 1e-7 >= min(
                instance.energy_wh[next_customer][node]
                for node in (instance.depot, *instance.stations)
            )
        go_feasible = energy_to_next <= soc + 1e-7 and onward_safe
        reachable = [
            instance.energy_wh[current][station] <= soc + 1e-7
            and station != current
            and (
                instance.energy_wh[station][next_customer]
                + (
                    0.0
                    if next_customer == instance.depot
                    else min(
                        instance.energy_wh[next_customer][node]
                        for node in (instance.depot, *instance.stations)
                    )
                )
                <= instance.battery_wh + 1e-7
            )
            for station in instance.stations
        ]
        action_mask = torch.tensor(
            [go_feasible, any(reachable)], dtype=torch.bool, device=model.device
        )
        if not bool(action_mask.any()):
            reason = "no feasible GO or CHARGE action"
            break
        action = int(action_logits.masked_fill(~action_mask, float("-inf")).argmax())
        if action == 0:
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
        quantity_logits = model.quantity_logits(state, stations[station_index])
        required_target = instance.energy_wh[station][next_customer] + (
            0.0
            if next_customer == instance.depot
            else min(
                instance.energy_wh[next_customer][node]
                for node in (instance.depot, *instance.stations)
            )
        )
        quantity_mask = torch.tensor(
            [
                fraction * instance.battery_wh > arrival + 1e-7
                and fraction * instance.battery_wh + 1e-7 >= required_target
                for fraction in DirectChargePolicy.targets
            ],
            dtype=torch.bool,
            device=model.device,
        )
        if not bool(quantity_mask.any()):
            reason = "no positive target-SOC action"
            break
        quantity = int(
            quantity_logits.masked_fill(~quantity_mask, float("-inf")).argmax()
        )
        target_soc = DirectChargePolicy.targets[quantity] * instance.battery_wh
        amount = target_soc - arrival
        # Evaluate exact curve here; check_route independently recomputes it.
        charging_time(
            instance.curves[instance.nodes[station].station_type], arrival, target_soc
        )
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
        state_features, station_features, action, station, quantity = examples[
            (step - 1) % len(examples)
        ]
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
                model.quantity_logits(state, station_features[station]).unsqueeze(0),
                torch.tensor([quantity], device=device),
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoints / f"CHARGE_n{args.size}_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "size": args.size}, checkpoint)
    args.results.mkdir(parents=True, exist_ok=True)
    output = args.results / f"training_CHARGE_n{args.size}_seed{args.seed}.csv"
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
    size: int, count: int, seed_start: int
) -> list[tuple[Instance, tuple[tuple[int, ...], ...]]]:
    data = []
    candidate_seed = seed_start
    while len(data) < count and candidate_seed < seed_start + count * 20:
        instance = generate_synthetic(size, candidate_seed)
        result = randomized_greedy(
            instance,
            restarts=3,
            seed=candidate_seed,
            local_search_passes=1,
            local_search_budget=800,
        )
        if result.feasible:
            data.append((instance, result.customer_routes))
        candidate_seed += 1
    if len(data) < count:
        raise RuntimeError(f"Only generated {len(data)}/{count} feasible teacher instances")
    return data


def teacher_loss(
    model: HATPolicy,
    instance: Instance,
    teacher_routes: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, int]:
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = []
    unserved = set(instance.customers)
    losses = []
    decisions = 0

    for teacher_route in teacher_routes:
        active_index: int | None = None
        for customer in teacher_route:
            nodes, route_vectors, open_indices, context = model.encode_state(
                instance, routes, unserved
            )
            route_scores = model.route_logits(route_vectors, open_indices, context)
            if active_index is None:
                route_target = len(open_indices)
                selected_vector = None
                last_node = instance.depot
                prefix: tuple[int, ...] = ()
            else:
                route_target = open_indices.index(active_index)
                selected_vector = route_vectors[active_index]
                prefix = routes[active_index]["customers"]
                last_node = prefix[-1]
            losses.append(
                F.cross_entropy(
                    route_scores.unsqueeze(0),
                    torch.tensor([route_target], device=model.device),
                )
            )
            order = sorted(unserved)
            customer_scores = model.customer_logits(
                instance,
                nodes,
                selected_vector,
                last_node,
                context,
                order,
                allow_close=active_index is not None,
            )
            masks = [
                oracle.solve((*prefix, candidate)).feasible for candidate in order
            ]
            if active_index is not None:
                masks.append(True)
            masked_scores = customer_scores.masked_fill(
                ~torch.tensor(masks, dtype=torch.bool, device=model.device),
                float("-inf"),
            )
            customer_target = order.index(customer)
            losses.append(
                F.cross_entropy(
                    masked_scores.unsqueeze(0),
                    torch.tensor([customer_target], device=model.device),
                )
            )
            plan = oracle.solve((*prefix, customer))
            if active_index is None:
                routes.append(route_state(instance, (customer,), plan))
                active_index = len(routes) - 1
            else:
                routes[active_index] = route_state(
                    instance, (*prefix, customer), plan
                )
            unserved.remove(customer)
            decisions += 1

        # Explicit CLOSE decision for the completed teacher route.
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved
        )
        route_scores = model.route_logits(route_vectors, open_indices, context)
        route_target = open_indices.index(active_index)
        losses.append(
            F.cross_entropy(
                route_scores.unsqueeze(0),
                torch.tensor([route_target], device=model.device),
            )
        )
        order = sorted(unserved)
        prefix = routes[active_index]["customers"]
        customer_scores = model.customer_logits(
            instance,
            nodes,
            route_vectors[active_index],
            prefix[-1],
            context,
            order,
            allow_close=True,
        )
        masks = [oracle.solve((*prefix, candidate)).feasible for candidate in order]
        masks.append(True)
        masked_scores = customer_scores.masked_fill(
            ~torch.tensor(masks, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        losses.append(
            F.cross_entropy(
                masked_scores.unsqueeze(0),
                torch.tensor([len(order)], device=model.device),
            )
        )
        routes[active_index]["closed"] = True
        decisions += 1
    return torch.stack(losses).mean(), decisions


def decode(
    model: HATPolicy,
    instance: Instance,
    sample: bool = False,
    seed: int = 0,
) -> dict:
    start = time.perf_counter()
    if sample:
        torch.manual_seed(seed)
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = []
    unserved = set(instance.customers)
    log_probs: list[torch.Tensor] = []
    entropy = 0.0
    reason = ""
    max_decisions = 3 * len(instance.customers) + 5

    for decision in range(max_decisions):
        if not unserved and all(route["closed"] for route in routes):
            break
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved
        )
        route_scores = model.route_logits(route_vectors, open_indices, context)
        new_feasible = bool(unserved) and any(
            oracle.solve((customer,)).feasible for customer in unserved
        )
        route_mask = [True] * len(open_indices) + [new_feasible]
        if not any(route_mask):
            reason = "no route action can serve remaining customers"
            break
        route_scores = route_scores.masked_fill(
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
        masks = [oracle.solve((*prefix, candidate)).feasible for candidate in order]
        if not is_new:
            masks.append(True)
        if not any(masks):
            reason = "selected route has no feasible action"
            break
        customer_scores = customer_scores.masked_fill(
            ~torch.tensor(masks, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
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
        plan = oracle.solve((*prefix, customer))
        if is_new:
            routes.append(route_state(instance, (customer,), plan))
        else:
            routes[selected_index] = route_state(
                instance, (*prefix, customer), plan
            )
        unserved.remove(customer)
    else:
        reason = "decision budget exceeded"

    plans = [route["plan"] for route in routes]
    valid, validation_reason = check_solution(instance, plans) if not unserved else (
        False,
        f"{len(unserved)} customers unserved",
    )
    if not valid:
        reason = reason or validation_reason
    objective = sum(plan.duration_hours for plan in plans) if valid else float("inf")
    return {
        "feasible": valid,
        "objective": objective,
        "routes": tuple(route["customers"] for route in routes),
        "plans": plans,
        "runtime": time.perf_counter() - start,
        "oracle_calls": oracle.calls,
        "reason": reason,
        "log_prob": torch.stack(log_probs).sum() if log_probs else None,
        "entropy": entropy / max(1, len(log_probs)),
        "decisions": decision + 1,
    }


def train_model(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = make_teacher_data(args.size, args.train_instances, 10000 + args.seed * 100)
    validation = make_teacher_data(args.size, args.val_instances, 20000 + args.seed * 100)
    model = HATPolicy(VARIANTS[args.model]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    start = time.perf_counter()
    best_score = (-1.0, float("-inf"))
    best_state = copy.deepcopy(model.state_dict())

    for step in range(1, args.steps + 1):
        instance, routes = train_data[(step - 1) % len(train_data)]
        optimizer.zero_grad()
        loss, decisions = teacher_loss(model, instance, routes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                decoded = [decode(model, instance) for instance, _ in validation]
            feasible = [result for result in decoded if result["feasible"]]
            mean_objective = (
                sum(result["objective"] for result in feasible) / len(feasible)
                if feasible
                else float("nan")
            )
            row = {
                "stage": "bc",
                "step": step,
                "loss": float(loss.detach()),
                "feasible_rate": len(feasible) / len(decoded),
                "validation_objective": mean_objective,
                "entropy": sum(result["entropy"] for result in decoded) / len(decoded),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": decisions,
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -mean_objective)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    # Small self-critical REINFORCE fine-tune.
    for rl_step in range(1, args.rl_steps + 1):
        instance, _ = train_data[(rl_step - 1) % len(train_data)]
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
        if rl_step == args.rl_steps:
            model.eval()
            with torch.no_grad():
                decoded = [decode(model, val_instance) for val_instance, _ in validation]
            feasible = [result for result in decoded if result["feasible"]]
            mean_objective = (
                sum(result["objective"] for result in feasible) / len(feasible)
                if feasible
                else float("nan")
            )
            row = {
                "stage": "rl",
                "step": rl_step,
                "loss": float(rl_loss.detach()),
                "feasible_rate": len(feasible) / len(decoded),
                "validation_objective": mean_objective,
                "entropy": sum(result["entropy"] for result in decoded) / len(decoded),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": sampled["decisions"],
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -mean_objective)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoints / f"{args.model}_n{args.size}_seed{args.seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(VARIANTS[args.model]),
            "model": args.model,
            "size": args.size,
            "seed": args.seed,
        },
        checkpoint,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    log_path = args.results / f"training_{args.model}_n{args.size}_seed{args.seed}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {checkpoint} and {log_path}; total {time.perf_counter() - start:.2f}s")


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
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if args.model == "CHARGE":
        train_direct(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
