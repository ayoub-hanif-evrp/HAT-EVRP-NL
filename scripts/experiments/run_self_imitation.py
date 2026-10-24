"""REINFORCE with self-imitation, plus an imitation-only control.

Elites come from the narrow neural-guided decoder and are kept only when they
beat the frozen behavior-cloning greedy route. The protocol is written before
the first update. Confirmation data are not read.
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bc_rl import leave_one_out_baselines, reinforce_loss, reward_value
from charging import FixedRouteOracle
from data import generate_synthetic
from train import decode, load_policy

STUDY = ROOT / "results" / "raw" / "algorithmic_superiority_v1"
OUT = STUDY / "self_imitation"
DEV_INSTANCES = STUDY / "splits" / "instances.pkl"
DEV_HASHES = STUDY / "splits" / "hashes.csv"
BC_PATTERN = "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt"
PARENT = "5fa3e382f6cb408fd35edf10714be693cb1bad0c"

PROTOCOL = {
    "name": "self_imitation_reinforce_v1",
    "parent_commit": PARENT,
    "start_from": BC_PATTERN,
    "seeds": [42, 43, 44, 45, 46],
    "modes": ["imitation", "rl_imitation"],
    "train_seed_start": 46_100_000,
    "n_train": 40,
    "generator": "v2",
    "size": 10,
    "teacher_filter": False,
    "steps": 80,
    "batch_size": 2,
    "on_policy_samples": 2,
    "sample_seed0": 980_000,
    "lr": 2e-5,
    "lr_note": "Fixed before this run. Same learning rate used previously when REINFORCE started from these behavior-cloning checkpoints. Not tuned here.",
    "imitation_coef": 0.1,
    "elite_tolerance_h": 1e-4,
    "elite_reference": "greedy objective of the initial behavior-cloning checkpoint, frozen before updates",
    "decoder_budget": {
        "name": "narrow",
        "max_deviations": 4,
        "alternatives": ["customer_rank2"],
        "max_solver_executions": 400,
    },
    "reward": "normalized",
    "baseline": "leave_one_out on the on-policy samples",
    "entropy_coef": 0.0,
    "grad_clip": 1.0,
    "eval_every": 20,
    "selection": "lowest penalized greedy duration on dev_v2, all generated instances",
    "returned_route": "one greedy decode, cost_lambda 0, no local search",
    "practical_threshold_relative": 0.01,
    "rl_claim_rule": (
        "On physically coverable dev_v2 instances, rl_imitation greedy duration is at least 1% "
        "below the imitation-only greedy duration, on at least 4 of 5 seeds, with a two-sided "
        "Student-t 95% interval entirely below zero. Beating the original behavior-cloning "
        "checkpoint is not sufficient."
    ),
    "description": (
        "imitation trains only the detached elite-action cross-entropy. "
        "rl_imitation adds leave-one-out REINFORCE on two on-policy samples. "
        "Both modes run the same narrow guided search and the same number of samples."
    ),
}


def _write_json(path: Path, payload: dict) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"{path} already exists and does not match this script")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _actions_from_trace(trace: list) -> list[dict]:
    return [{"route": step["route_chosen"], "customer": step["customer_chosen"]} for step in trace]


def _customer_alternatives(step: dict) -> list[dict]:
    if len(step["customer_ranked"]) < 2:
        return []
    alternative = step["customer_ranked"][1]
    if alternative == step["customer_chosen"]:
        return []
    return [{"route": step["route_chosen"], "customer": alternative}]


def narrow_elite(policy, instance, reference_h: float) -> tuple[list[dict] | None, int]:
    """Return the action trace of a narrow-budget route that beats the frozen reference."""
    oracle = FixedRouteOracle(instance)
    trace: list = []
    with torch.no_grad():
        greedy = decode(policy, instance, sample=False, seed=0, oracle=oracle, trace=trace, cost_lambda=0.0)
    best_actions = None
    best_h = float(reference_h)
    if greedy["feasible"] and float(greedy["objective"]) < best_h - PROTOCOL["elite_tolerance_h"]:
        best_actions = _actions_from_trace(trace)
        best_h = float(greedy["objective"])
    budget = PROTOCOL["decoder_budget"]
    for decision, step in enumerate(trace[: int(budget["max_deviations"])]):
        if oracle.solver_executions >= int(budget["max_solver_executions"]):
            break
        for choice in _customer_alternatives(step):
            forced = _actions_from_trace(trace[:decision]) + [choice]
            alt_trace: list = []
            with torch.no_grad():
                result = decode(
                    policy,
                    instance,
                    sample=False,
                    seed=0,
                    oracle=oracle,
                    trace=alt_trace,
                    forced_actions=forced,
                    cost_lambda=0.0,
                )
            if result["feasible"] and float(result["objective"]) < best_h - PROTOCOL["elite_tolerance_h"]:
                best_actions = _actions_from_trace(alt_trace)
                best_h = float(result["objective"])
    return best_actions, oracle.solver_executions


def _penalized(result: dict, instance) -> float:
    if result["feasible"]:
        return float(result["objective"])
    return float(len(instance.customers) * instance.max_route_hours)


def eval_greedy(policy, instances, coverable: list[int]) -> dict:
    policy.eval()
    penalized = []
    covered = []
    feasible = 0
    with torch.no_grad():
        for instance, flag in zip(instances, coverable):
            result = decode(policy, instance, sample=False, seed=0, cost_lambda=0.0)
            value = _penalized(result, instance)
            penalized.append(value)
            feasible += int(bool(result["feasible"]))
            if flag:
                covered.append(value)
    return {
        "penalized_h": sum(penalized) / len(penalized),
        "coverable_h": sum(covered) / len(covered),
        "feasible_rate": feasible / len(instances),
    }


def _references(policy, instances) -> list[float]:
    policy.eval()
    values = []
    with torch.no_grad():
        for instance in instances:
            result = decode(policy, instance, sample=False, seed=0, cost_lambda=0.0)
            values.append(float(result["objective"]) if result["feasible"] else float("inf"))
    return values


def _imitation_loss(policy, instance, actions: list[dict]) -> torch.Tensor | None:
    replay = decode(policy, instance, sample=False, seed=0, forced_actions=actions, cost_lambda=0.0)
    log_prob = replay["log_prob"]
    if log_prob is None or not torch.isfinite(log_prob).all():
        return None
    return -log_prob


def _on_policy(policy, instance, step: int, slot: int, train_index: int) -> list[dict]:
    rows = []
    for trial in range(PROTOCOL["on_policy_samples"]):
        seed = PROTOCOL["sample_seed0"] + step * 64 + train_index * 4 + trial
        if policy.training:
            result = decode(policy, instance, sample=True, seed=seed, cost_lambda=0.0)
        else:
            with torch.no_grad():
                result = decode(policy, instance, sample=True, seed=seed, cost_lambda=0.0)
        rows.append(result)
    return rows


def train_one(seed: int, mode: str, train_instances, dev_instances, coverable, config: dict) -> dict:
    out_dir = OUT / "runs" / mode / f"H2_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "run_meta.json").exists():
        return json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    policy = load_policy(ROOT / BC_PATTERN.format(seed=seed))
    references = _references(policy, train_instances)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=PROTOCOL["lr"])
    rng = random.Random(5000 + seed)
    queue: list[int] = []
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    best_score = float("inf")
    best_step = 0
    log_rows = []
    elites = 0
    solver_executions = 0
    started = time.perf_counter()
    for step in range(PROTOCOL["steps"]):
        if len(queue) < PROTOCOL["batch_size"]:
            queue = list(range(len(train_instances)))
            rng.shuffle(queue)
        batch = [queue.pop() for _ in range(PROTOCOL["batch_size"])]
        optimizer.zero_grad()
        total = None
        for slot, index in enumerate(batch):
            instance = train_instances[index]
            actions, executions = narrow_elite(policy, instance, references[index])
            solver_executions += executions
            if actions is not None:
                elites += 1
                imitation = _imitation_loss(policy, instance, actions)
                if imitation is not None:
                    term = PROTOCOL["imitation_coef"] * imitation
                    total = term if total is None else total + term
            if mode == "rl_imitation":
                sampled = _on_policy(policy, instance, step, slot, index)
            else:
                policy.eval()
                with torch.no_grad():
                    sampled = _on_policy(policy, instance, step, slot, index)
                policy.train()
            if mode == "rl_imitation":
                rewards = [
                    reward_value(
                        result,
                        n_customers=len(instance.customers),
                        max_route_hours=instance.max_route_hours,
                        mode="normalized",
                    )
                    for result in sampled
                ]
                baselines = leave_one_out_baselines(rewards)
                advantages = [reward - baseline for reward, baseline in zip(rewards, baselines)]
                rl_term = reinforce_loss(
                    [result["log_prob"] for result in sampled],
                    advantages,
                    [None for _ in sampled],
                    entropy_coef=0.0,
                )
                total = rl_term if total is None else total + rl_term
        if total is not None and torch.isfinite(total):
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), PROTOCOL["grad_clip"])
            optimizer.step()
        if step == 0 or (step + 1) % PROTOCOL["eval_every"] == 0 or step + 1 == PROTOCOL["steps"]:
            metrics = eval_greedy(policy, dev_instances, coverable)
            policy.train()
            log_rows.append({"step": step + 1, "elites": elites, "solver_executions": solver_executions, **metrics})
            print(
                f"{mode} seed {seed} step {step + 1} penalized {metrics['penalized_h']:.3f} "
                f"coverable {metrics['coverable_h']:.3f} elites {elites}",
                flush=True,
            )
            if metrics["penalized_h"] < best_score:
                best_score = metrics["penalized_h"]
                best_step = step + 1
                best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    final = eval_greedy(policy, dev_instances, coverable)
    torch.save({"config": config, "state_dict": best_state, "step": best_step}, out_dir / "best.pt")
    with (out_dir / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    meta = {
        "seed": seed,
        "mode": mode,
        "best_step": best_step,
        "best_penalized_h": best_score,
        "final_penalized_h": final["penalized_h"],
        "final_coverable_h": final["coverable_h"],
        "final_feasible_rate": final["feasible_rate"],
        "elites": elites,
        "solver_executions": solver_executions,
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    _write_json(OUT / "frozen_protocol.json", PROTOCOL)
    train_path = OUT / "train_instances.pkl"
    if train_path.exists():
        train_instances = pickle.loads(train_path.read_bytes())
    else:
        train_instances = [
            generate_synthetic(PROTOCOL["size"], PROTOCOL["train_seed_start"] + offset, generator_version=PROTOCOL["generator"])
            for offset in range(PROTOCOL["n_train"])
        ]
        train_path.write_bytes(pickle.dumps(train_instances))
    blob = pickle.loads(DEV_INSTANCES.read_bytes())
    dev_instances = blob["splits"]["dev_v2"]
    coverable = []
    for row in csv.DictReader(DEV_HASHES.open(encoding="utf-8")):
        if row["split"] == "dev_v2":
            coverable.append(int(row["physically_coverable"]))
    if len(coverable) != len(dev_instances):
        raise SystemExit("development coverability rows do not match instances")
    template = torch.load(ROOT / BC_PATTERN.format(seed=42), map_location="cpu", weights_only=False)
    failures = []
    for seed in PROTOCOL["seeds"]:
        for mode in PROTOCOL["modes"]:
            try:
                train_one(seed, mode, train_instances, dev_instances, coverable, template["config"])
            except Exception as exc:
                failures.append({"seed": seed, "mode": mode, "error": str(exc)})
                print({"failed": True, "seed": seed, "mode": mode, "error": str(exc)}, flush=True)
                raise
    (OUT / "status.json").write_text(json.dumps({"completed": not failures, "failures": failures}, indent=2), encoding="utf-8")
    print({"completed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
