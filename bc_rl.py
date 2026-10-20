"""Primary training pipeline: behavior cloning, then REINFORCE.

No third training phase. Legacy POMO and repair-RL stay in ``train.py`` and
are refused by this entry point.

RT (internal key H2) encodes optimizer-enriched route-stop sequences, including
charging features. A token is a stop representation, not necessarily one
committed historical action.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle
import platform
import random
import time
from pathlib import Path

import torch

from artifact_io import environment_metadata, safe_save_checkpoint, safe_write_csv
from data import generate_synthetic
from train import (
    VARIANTS,
    decode,
    make_teacher_data,
    teacher_loss,
    write_training_csv,
)
from model import HATPolicy


def _inherited_bc_updates(source_bc: Path) -> int:
    meta_path = source_bc.parent / "run_meta.json"
    if not meta_path.exists():
        return 0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta.get("completed_bc_updates") or 0)


def cyclic_batch_indices(step: int, batch_size: int, n_items: int) -> list[int]:
    """Original fixed stride. With 300 steps and batch 2 this misses part of a 500-set."""
    if n_items <= 0 or step < 1 or batch_size < 1:
        raise ValueError("cyclic indices need a positive step, batch, and dataset")
    return [(step - 1 + offset) % n_items for offset in range(batch_size)]


class EpochShuffler:
    """Deterministic shuffle that exhausts every index before any index repeats."""

    def __init__(self, n_items: int, seed: int):
        if n_items <= 0:
            raise ValueError("cannot shuffle an empty dataset")
        self.n_items = int(n_items)
        self._rng = random.Random(int(seed))
        self._queue: list[int] = []

    def take(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be positive")
        chosen: list[int] = []
        while len(chosen) < count:
            if not self._queue:
                order = list(range(self.n_items))
                self._rng.shuffle(order)
                self._queue = order
            chosen.append(self._queue.pop(0))
        return chosen


def reward_value(
    result: dict,
    *,
    n_customers: int,
    max_route_hours: float,
    mode: str = "normalized",
    epsilon: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Map a decode result to a scalar reward.

    ``legacy``: R = -Z_full if feasible else -100 (Cycle-3).

    ``normalized``:
      R_feasible = -Z_full / (n * T_max)
      R_infeasible = -1 - epsilon - alpha * unserved_fraction
                     - beta * normalized_violation
    Violation is included only when ``normalized_violation`` is a finite
    number already present on the result. Missing violations are omitted,
    not invented. With at most n routes each of duration <= T_max,
    R_feasible is in [-1, 0]. epsilon > 0 keeps infeasible rewards strictly
    below that band.
    """
    if mode == "legacy":
        if result.get("feasible"):
            return -float(result["objective"])
        return -100.0
    if mode != "normalized":
        raise ValueError(f"unknown reward mode {mode}")
    n = max(int(n_customers), 1)
    tmax = max(float(max_route_hours), 1e-9)
    if result.get("feasible"):
        return -float(result["objective"]) / (n * tmax)
    penalty = -1.0 - float(epsilon)
    unserved = result.get("n_unserved")
    if unserved is not None:
        penalty -= float(alpha) * (float(unserved) / n)
    violation = result.get("normalized_violation")
    if violation is not None and math.isfinite(float(violation)):
        penalty -= float(beta) * max(float(violation), 0.0)
    return penalty


def feasible_reward_bounded(n_routes: int, n_customers: int, z_full: float, tmax: float) -> bool:
    """True when a feasible solution cannot outrank the infeasible band."""
    if n_routes > n_customers:
        return False
    scale = n_customers * tmax
    reward = -z_full / scale
    return -1.0 - 1e-9 <= reward <= 1e-9


def leave_one_out_baselines(rewards: list[float]) -> list[float]:
    """Baseline for rollout i is the mean of the other rollouts.

    A single rollout has no peer; the fallback baseline equals that reward
    so the advantage is zero rather than an invented constant.
    """
    if len(rewards) <= 1:
        return list(rewards)
    total = float(sum(rewards))
    denom = len(rewards) - 1
    return [(total - reward) / denom for reward in rewards]


def scale_advantages(advantages: list[float], mode: str = "none") -> list[float]:
    if mode == "none" or len(advantages) < 2:
        return list(advantages)
    if mode != "std":
        raise ValueError(mode)
    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = math.sqrt(var)
    if std < 1e-6:
        return [0.0 for _ in advantages]
    return [(a - mean) / std for a in advantages]


def reinforce_loss(
    log_probs: list[torch.Tensor],
    advantages: list[float],
    entropy_tensors: list[torch.Tensor | None],
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """REINFORCE with detached advantages. Entropy is differentiable when present."""
    if len(log_probs) != len(advantages):
        raise ValueError("log_probs and advantages length mismatch")
    device = log_probs[0].device
    total = torch.zeros((), device=device)
    used = 0
    for log_prob, advantage, entropy in zip(log_probs, advantages, entropy_tensors):
        if log_prob is None or not torch.isfinite(log_prob).all():
            continue
        adv = torch.tensor(float(advantage), device=device, dtype=log_prob.dtype)
        # Advantage is a reward difference; it must not carry policy gradients.
        adv = adv.detach()
        term = -adv * log_prob
        if entropy_coef and entropy is not None and entropy.requires_grad:
            term = term - float(entropy_coef) * entropy
        if torch.isfinite(term):
            total = total + term
            used += 1
    if used == 0:
        return total
    return total / used


def selection_score(feasible_rate: float, penalized_mean: float) -> tuple[float, float]:
    """Higher is better. Feasibility first, then lower penalized duration."""
    if not math.isfinite(penalized_mean):
        penalized_mean = float("inf")
    return (float(feasible_rate), -float(penalized_mean))


def choose_deployed(bc_score: tuple[float, float], rl_score: tuple[float, float] | None) -> str:
    if rl_score is None or rl_score <= bc_score:
        return "bc"
    return "rl"


def validation_summary(decoded: list[dict], instances) -> dict:
    """Feasibility and quality stay separate.

    ``penalized_mean_h`` assigns infeasible runs n*T_max so a mean is never
    computed only on successes and then presented as if every run succeeded.
    ``feasible_mean_h`` is the mean among feasible runs and may be NaN.
    """
    n = len(decoded)
    feasible_obj = []
    penalized = []
    for result, instance in zip(decoded, instances):
        cap = len(instance.customers) * float(instance.max_route_hours)
        if result.get("feasible") and math.isfinite(float(result["objective"])):
            feasible_obj.append(float(result["objective"]))
            penalized.append(float(result["objective"]))
        else:
            penalized.append(cap)
    return {
        "n": n,
        "n_feasible": len(feasible_obj),
        "feasible_rate": (len(feasible_obj) / n) if n else float("nan"),
        "feasible_mean_h": (
            sum(feasible_obj) / len(feasible_obj) if feasible_obj else float("nan")
        ),
        "penalized_mean_h": sum(penalized) / n if n else float("nan"),
    }


def _instance_seed(instance, fallback: int) -> int:
    name = getattr(instance, "name", "") or ""
    marker = name.rfind("_s")
    if marker >= 0 and name[marker + 2 :].isdigit():
        return int(name[marker + 2 :])
    return fallback


def instance_profile(instance, seed: int, generator: str) -> dict:
    stations = [node for node in instance.nodes if node.kind == 2]
    tech: dict[str, int] = {}
    for node in stations:
        tech[node.station_type or "unknown"] = tech.get(node.station_type or "unknown", 0) + 1
    return {
        "seed": _instance_seed(instance, seed),
        "generator": generator,
        "name": instance.name,
        "n_customers": len(instance.customers),
        "n_stations": len(stations),
        "tech_counts": tech,
        "battery_wh": instance.battery_wh,
        "max_route_hours": instance.max_route_hours,
    }


def _eval_split(model, split, cost_lambda: float, route_lambda: float) -> dict:
    model.eval()
    decoded = []
    instances = []
    with torch.no_grad():
        for instance, _routes in split:
            decoded.append(
                decode(
                    model,
                    instance,
                    sample=False,
                    cost_lambda=cost_lambda,
                    route_lambda=route_lambda,
                )
            )
            instances.append(instance)
    summary = validation_summary(decoded, instances)
    summary["entropy"] = (
        sum(float(row["entropy"]) for row in decoded) / len(decoded) if decoded else float("nan")
    )
    return summary


def _finite_grads(model) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def load_or_build_teachers(args, train_seed_start: int, val_seed_start: int, generator_version: str):
    """Build expert data once and reuse it across model/seed runs."""
    key = {
        "size": int(args.size),
        "n_train": int(args.train_instances),
        "n_val": int(args.val_instances),
        "train_seed_start": int(train_seed_start),
        "val_seed_start": int(val_seed_start),
        "teacher": getattr(args, "teacher", "basic"),
        "generator": generator_version,
    }
    cache = getattr(args, "teacher_cache", None)
    path = Path(cache) if cache else None
    if path is not None and path.exists():
        blob = pickle.loads(path.read_bytes())
        if blob.get("key") != key:
            raise SystemExit(f"teacher cache key mismatch at {path}")
        print({"teacher_cache": "loaded", "path": str(path), "n_train": len(blob["train"])})
        return blob["train"], blob["val"]
    started = time.perf_counter()
    train_data = make_teacher_data(
        args.size,
        args.train_instances,
        train_seed_start,
        teacher=key["teacher"],
        generator_version=generator_version,
    )
    validation = make_teacher_data(
        args.size,
        args.val_instances,
        val_seed_start,
        teacher=key["teacher"],
        generator_version=generator_version,
    )
    print({"teacher_build_s": round(time.perf_counter() - started, 3), "n_train": len(train_data), "n_val": len(validation)})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"key": key, "train": train_data, "val": validation}))
        print({"teacher_cache": "wrote", "path": str(path)})
    return train_data, validation


def train_primary(args) -> dict:
    """BC then RL. Refuses POMO and repair-RL."""
    if int(getattr(args, "pomo_k", 0) or 0) > 0:
        raise SystemExit("primary pipeline does not run a POMO training phase")
    if int(getattr(args, "repair_rl_steps", 0) or 0) > 0:
        raise SystemExit("primary pipeline does not run a repair-RL phase")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = getattr(args, "version", "bcrl_v1")
    if version in {"c2", "c3"} or str(version).endswith("_c3"):
        raise SystemExit(f"refusing to write primary artifacts under historical version {version}")

    run_root_early = Path(getattr(args, "run_dir", Path("results/raw/bc_rl_v1")))
    run_id_early = getattr(args, "run_id", None)
    if run_id_early:
        done = run_root_early / str(run_id_early)
        if all((done / name).exists() for name in ("run_meta.json", "bc_best.pt", "rl_best.pt", "deployed_best.pt")):
            print({"skip_complete": str(done)})
            return json.loads((done / "run_meta.json").read_text(encoding="utf-8"))

    generator_version = getattr(args, "generator", "v2")
    # data_seed fixes the instance band. args.seed still controls initialization
    # and sampling, so matched models can share one dataset.
    data_seed = int(getattr(args, "data_seed", None) if getattr(args, "data_seed", None) is not None else args.seed)
    # Disjoint from Cycle-3 bands (700000/800000 and 920000/930000).
    train_seed_start = 1_100_000 + data_seed * 10_000
    val_seed_start = 1_200_000 + data_seed * 10_000
    train_data, validation = load_or_build_teachers(
        args, train_seed_start, val_seed_start, generator_version
    )
    train_seeds = list(range(train_seed_start, train_seed_start + len(train_data)))
    val_seeds = list(range(val_seed_start, val_seed_start + len(validation)))
    if set(train_seeds) & set(val_seeds):
        raise RuntimeError("train and validation seeds overlap")

    model = HATPolicy(VARIANTS[args.model]).to(device)
    bc_lr = float(getattr(args, "bc_lr", None) or args.lr)
    rl_lr = float(getattr(args, "rl_lr", None) or min(args.lr, 1e-4))
    grad_clip = float(getattr(args, "grad_clip", 1.0))
    cost_lambda = float(getattr(args, "cost_lambda", 0.0))
    route_lambda = float(getattr(args, "route_lambda", 0.0))
    reward_mode = getattr(args, "reward", "normalized")
    epsilon = float(getattr(args, "reward_epsilon", 0.05))
    alpha = float(getattr(args, "reward_alpha", 1.0))
    beta = float(getattr(args, "reward_beta", 1.0))
    rollouts = max(1, int(getattr(args, "rollouts", 4)))
    baseline_name = getattr(args, "baseline", "loo" if rollouts > 1 else "greedy")
    entropy_coef = float(getattr(args, "entropy_coef", 0.0))
    adv_scale = getattr(args, "advantage_scale", "none")
    batch_size = max(1, int(getattr(args, "batch_size", 1)))
    instance_order = getattr(args, "instance_order", "shuffled")
    if instance_order not in {"shuffled", "cyclic"}:
        raise SystemExit(f"unknown instance order {instance_order}")
    source_bc = getattr(args, "rl_from_bc", None)
    source_bc = Path(source_bc) if source_bc else None
    bc_patience = int(getattr(args, "bc_patience", 0) or 0)
    rl_patience = int(getattr(args, "rl_patience", 0) or 0)
    multi_target = bool(getattr(args, "multi_target", False))
    fleet_counterfactual = bool(getattr(args, "fleet_counterfactual", False))

    if source_bc is not None:
        blob = torch.load(source_bc, map_location=device, weights_only=False)
        if blob.get("model") not in (None, args.model):
            raise SystemExit(
                f"checkpoint model {blob.get('model')} does not match --model {args.model}"
            )
        model.load_state_dict(blob["state_dict"])
    bc_optimizer = torch.optim.Adam(model.parameters(), lr=bc_lr)
    rows: list[dict] = []
    reward_rows: list[dict] = []
    profiles = []
    for offset, (instance, _routes) in enumerate(train_data):
        profiles.append(instance_profile(instance, train_seed_start + offset, generator_version))
    for offset, (instance, _routes) in enumerate(validation):
        row = instance_profile(instance, val_seed_start + offset, generator_version)
        row["split"] = "validation"
        profiles.append(row)

    best_bc_score = (-1.0, float("-inf"))
    best_bc_state = copy.deepcopy(model.state_dict())
    bc_stall = 0
    train_rng = random.Random(args.seed + 17)
    start = time.perf_counter()
    run_root = Path(getattr(args, "run_dir", Path("results/raw/bc_rl_v1")))
    run_root.mkdir(parents=True, exist_ok=True)

    def _log_eval(stage: str, step: int, loss: float) -> dict:
        metrics = _eval_split(model, validation, cost_lambda, route_lambda)
        score = selection_score(metrics["feasible_rate"], metrics["penalized_mean_h"])
        row = {
            "stage": stage,
            "step": step,
            "loss": loss,
            "feasible_rate": metrics["feasible_rate"],
            "n_feasible": metrics["n_feasible"],
            "n_eval": metrics["n"],
            "feasible_mean_h": metrics["feasible_mean_h"],
            "penalized_mean_h": metrics["penalized_mean_h"],
            "validation_objective": metrics["penalized_mean_h"],
            "entropy": metrics["entropy"],
            "elapsed_s": time.perf_counter() - start,
            "device": str(device),
        }
        rows.append(row)
        print(row)
        return {"score": score, "metrics": metrics, "row": row}

    bc_sampler = EpochShuffler(len(train_data), int(args.seed) + 101) if instance_order == "shuffled" else None
    rl_sampler = EpochShuffler(len(train_data), int(args.seed) + 303) if instance_order == "shuffled" else None

    def _batch_pairs(step: int, sampler: EpochShuffler | None):
        if sampler is None:
            indices = cyclic_batch_indices(step, batch_size, len(train_data))
        else:
            indices = sampler.take(batch_size)
        return [train_data[index] for index in indices]

    # ----- Phase 1: behavior cloning (skipped when RL resumes from a saved BC checkpoint) -----
    model.train()
    bc_steps = 0 if source_bc is not None else args.steps
    for step in range(1, bc_steps + 1):
        bc_optimizer.zero_grad()
        loss_acc = None
        decisions = 0
        for instance, routes in _batch_pairs(step, bc_sampler):
            loss, decisions, _stats = teacher_loss(
                model,
                instance,
                routes,
                rng=train_rng,
                multi_target=multi_target,
                fleet_counterfactual=fleet_counterfactual,
            )
            loss = loss / batch_size
            if loss_acc is None:
                loss_acc = loss
            else:
                loss_acc = loss_acc + loss
        assert loss_acc is not None
        if not torch.isfinite(loss_acc):
            raise RuntimeError(f"non-finite BC loss at step {step}")
        loss_acc.backward()
        if not _finite_grads(model):
            bc_optimizer.zero_grad()
            print({"bc_step": step, "skipped": "non-finite grad"})
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            bc_optimizer.step()
        do_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if do_eval:
            packed = _log_eval("bc", step, float(loss_acc.detach()))
            if packed["score"] > best_bc_score:
                best_bc_score = packed["score"]
                best_bc_state = copy.deepcopy(model.state_dict())
                bc_stall = 0
            else:
                bc_stall += 1
            model.train()
            if bc_patience and bc_stall >= bc_patience:
                print({"bc_early_stop": step, "patience": bc_patience})
                break

    model.load_state_dict(best_bc_state)
    bc_ref = _log_eval("bc_best", 0, 0.0)
    bc_state = copy.deepcopy(model.state_dict())
    kl_coef = float(getattr(args, "bc_kl", 0.0) or 0.0)
    reference = None
    if kl_coef > 0:
        if cost_lambda != 0.0 or route_lambda != 0.0:
            raise SystemExit("bc-kl reference scoring is defined for unadjusted construction logits")
        reference = HATPolicy(VARIANTS[args.model]).to(device)
        reference.load_state_dict(bc_state)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    # ----- Phase 2: REINFORCE from the best BC checkpoint -----
    rl_optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)
    rl_best_score = (-1.0, float("-inf"))
    rl_best_state = None
    rl_best_source = "no_rl_validation_copied_bc"
    rl_stall = 0
    rl_start = _log_eval("rl_start", 0, 0.0)
    model.train()

    for rl_step in range(1, args.rl_steps + 1):
        batch_losses = []
        batch_rewards = []
        batch_reward_std = []
        batch_adv_abs = []
        batch_kl = []
        n_infeasible = 0
        oracle_calls = 0
        entropies = []
        for b, (instance, _routes) in enumerate(_batch_pairs(rl_step, rl_sampler)):
            sampled_rows = []
            for k in range(rollouts):
                sampled = decode(
                    model,
                    instance,
                    sample=True,
                    seed=args.seed + 10007 * rl_step + 17 * b + k,
                    cost_lambda=cost_lambda,
                    route_lambda=route_lambda,
                    reference_model=reference,
                )
                sampled_rows.append(sampled)
                oracle_calls += int(sampled.get("oracle_calls") or 0)
                if not sampled.get("feasible"):
                    n_infeasible += 1
            rewards = [
                reward_value(
                    row,
                    n_customers=len(instance.customers),
                    max_route_hours=instance.max_route_hours,
                    mode=reward_mode,
                    epsilon=epsilon,
                    alpha=alpha,
                    beta=beta,
                )
                for row in sampled_rows
            ]
            if baseline_name == "loo":
                baselines = leave_one_out_baselines(rewards)
            elif baseline_name == "greedy":
                model.eval()
                with torch.no_grad():
                    greedy = decode(
                        model,
                        instance,
                        sample=False,
                        seed=args.seed + rl_step,
                        cost_lambda=cost_lambda,
                        route_lambda=route_lambda,
                    )
                model.train()
                greedy_reward = reward_value(
                    greedy,
                    n_customers=len(instance.customers),
                    max_route_hours=instance.max_route_hours,
                    mode=reward_mode,
                    epsilon=epsilon,
                    alpha=alpha,
                    beta=beta,
                )
                baselines = [greedy_reward] * len(rewards)
                oracle_calls += int(greedy.get("oracle_calls") or 0)
            else:
                raise ValueError(baseline_name)
            advantages = scale_advantages(
                [r - bsl for r, bsl in zip(rewards, baselines)],
                adv_scale,
            )
            log_probs = []
            entropy_tensors = []
            for row in sampled_rows:
                log_probs.append(row["log_prob"])
                entropy_tensors.append(row.get("entropy_tensor"))
                entropies.append(float(row.get("entropy") or 0.0))
            # Drop rows whose log-prob is missing (decode failed before any action).
            paired = [
                (lp, adv, ent)
                for lp, adv, ent in zip(log_probs, advantages, entropy_tensors)
                if lp is not None
            ]
            if not paired:
                continue
            if all(abs(adv) <= 1e-12 for _lp, adv, _ent in paired):
                reward_rows.append(
                    {
                        "stage": "rl",
                        "step": rl_step,
                        "skipped": "zero_advantage",
                        "mean_reward": sum(rewards) / len(rewards),
                    }
                )
                continue
            loss = reinforce_loss(
                [p[0] for p in paired],
                [p[1] for p in paired],
                [p[2] for p in paired],
                entropy_coef=entropy_coef,
            )
            kl_terms = [
                row["log_prob"] - row["ref_log_prob"]
                for row in sampled_rows
                if row.get("log_prob") is not None and row.get("ref_log_prob") is not None
            ]
            if kl_coef > 0 and kl_terms:
                kl_mean = torch.stack(kl_terms).mean()
                loss = loss + kl_coef * kl_mean
                batch_kl.append(float(kl_mean.detach()))
            reward_mean = sum(rewards) / len(rewards)
            batch_reward_std.append(math.sqrt(sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)))
            batch_adv_abs.append(sum(abs(adv) for _lp, adv, _ent in paired) / len(paired))
            batch_losses.append(loss)
            batch_rewards.extend(rewards)
        if not batch_losses:
            grad_norm = 0.0
            loss_value = 0.0
        else:
            rl_optimizer.zero_grad()
            rl_loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(rl_loss):
                print({"rl_step": rl_step, "skipped": "non-finite loss"})
                grad_norm = float("nan")
                loss_value = float("nan")
            else:
                rl_loss.backward()
                if not _finite_grads(model):
                    rl_optimizer.zero_grad()
                    print({"rl_step": rl_step, "skipped": "non-finite grad"})
                    grad_norm = float("nan")
                    loss_value = float(rl_loss.detach())
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                    rl_optimizer.step()
                    loss_value = float(rl_loss.detach())
        reward_rows.append(
            {
                "stage": "rl",
                "step": rl_step,
                "loss": loss_value,
                "mean_reward": (sum(batch_rewards) / len(batch_rewards)) if batch_rewards else "",
                "n_infeasible_rollouts": n_infeasible,
                "n_rollouts": rollouts * batch_size,
                "oracle_calls": oracle_calls,
                "grad_norm": grad_norm,
                "entropy": (sum(entropies) / len(entropies)) if entropies else "",
                "reward_std": (sum(batch_reward_std) / len(batch_reward_std)) if batch_reward_std else "",
                "advantage_abs": (sum(batch_adv_abs) / len(batch_adv_abs)) if batch_adv_abs else "",
                "bc_kl": (sum(batch_kl) / len(batch_kl)) if batch_kl else "",
                "reward_mode": reward_mode,
                "baseline": baseline_name,
                "learning_rate": rl_lr,
            }
        )
        do_eval = rl_step == args.rl_steps or (args.eval_every > 0 and rl_step % args.eval_every == 0)
        if do_eval:
            packed = _log_eval("rl", rl_step, loss_value)
            if packed["score"] > rl_best_score:
                rl_best_score = packed["score"]
                rl_best_state = copy.deepcopy(model.state_dict())
                rl_best_source = "best_post_rl_validation"
                rl_stall = 0
            else:
                rl_stall += 1
            model.train()
            if rl_patience and rl_stall >= rl_patience:
                print({"rl_early_stop": rl_step, "patience": rl_patience})
                break

    if rl_best_state is None:
        rl_best_state = copy.deepcopy(bc_state)
        rl_best_score = bc_ref["score"]
        rl_improved = False
    else:
        rl_improved = rl_best_score > bc_ref["score"]
    deployed_from = choose_deployed(bc_ref["score"], rl_best_score if rl_improved else None)
    deployed_state = rl_best_state if deployed_from == "rl" else bc_state

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = getattr(args, "run_id", None) or f"{args.model}_n{args.size}_seed{args.seed}_{version}_{stamp}"
    out = run_root / str(run_id)
    if source_bc is not None and out.resolve() == source_bc.resolve().parent:
        raise SystemExit("refusing to write an RL rerun into the original checkpoint directory")
    out.mkdir(parents=True, exist_ok=True)

    def _payload(state, role: str) -> dict:
        return {
            "state_dict": state,
            "config": dataclasses.asdict(VARIANTS[args.model]),
            "model": args.model,
            "size": args.size,
            "seed": args.seed,
            "version": version,
            "role": role,
            "pipeline": "bc_then_rl",
            "reward_mode": reward_mode,
            "baseline": baseline_name,
            "rollouts": rollouts,
            "bc_lr": bc_lr,
            "rl_lr": rl_lr,
            "entropy_coef": entropy_coef,
            "cost_lambda": cost_lambda,
            "route_lambda": route_lambda,
            "multi_target": multi_target,
            "fleet_counterfactual": fleet_counterfactual,
            "train_seed_start": train_seed_start,
            "val_seed_start": val_seed_start,
            "generator": generator_version,
            "provenance": environment_metadata(),
            "git_note": "primary BC→RL; historical Cycle-3 checkpoints were not modified",
        }

    def _save(path: Path, state, role: str) -> Path:
        return safe_save_checkpoint(
            path,
            lambda p: torch.save(_payload(state, role), p),
            overwrite=False,
        )

    bc_path = _save(out / "bc_best.pt", bc_state, "bc_best")
    rl_path = _save(out / "rl_best.pt", rl_best_state, "rl_best")
    dep_path = _save(out / "deployed_best.pt", deployed_state, "deployed_best")
    log_path = safe_write_csv(out / "training.csv", lambda p: write_training_csv(p, rows), overwrite=False)
    reward_path = safe_write_csv(
        out / "reward.csv", lambda p: write_training_csv(p, reward_rows), overwrite=False
    )
    meta = {
        "run_id": run_id,
        "pipeline": ["bc", "rl"] if source_bc is None else ["saved_bc", "rl"],
        "instance_order": instance_order,
        "bc_source_checkpoint": str(source_bc) if source_bc is not None else "",
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "platform": platform.platform(),
        "data_seed": data_seed,
        "train_seed_start": train_seed_start,
        "val_seed_start": val_seed_start,
        "n_train": len(train_data),
        "n_val": len(validation),
        "bc_lr": bc_lr,
        "rl_lr": rl_lr,
        "reward": {
            "mode": reward_mode,
            "epsilon": epsilon,
            "alpha": alpha,
            "beta": beta,
            "feasible": "-Z_full/(n*T_max)" if reward_mode == "normalized" else "-Z_full",
            "infeasible": (
                "-1-epsilon-alpha*unserved_fraction-beta*normalized_violation"
                if reward_mode == "normalized"
                else "-100"
            ),
        },
        "rollouts": rollouts,
        "baseline": baseline_name,
        "entropy_coef": entropy_coef,
        "entropy_regularizer_differentiable": entropy_coef > 0,
        "bc_kl": kl_coef,
        "bc_score": list(bc_ref["score"]),
        "rl_start_score": list(rl_start["score"]),
        "rl_best_score": list(rl_best_score),
        "rl_improved_on_validation": bool(rl_improved),
        "rl_best_source": rl_best_source,
        "completed_bc_updates": (
            _inherited_bc_updates(source_bc)
            if source_bc is not None
            else max((int(r["step"]) for r in rows if r.get("stage") == "bc"), default=0)
        ),
        "completed_rl_updates": max((int(r["step"]) for r in rows if r.get("stage") == "rl"), default=0),
        "requested_bc_updates": int(args.steps),
        "requested_rl_updates": int(args.rl_steps),
        "deployed_from": deployed_from,
        "checkpoints": {
            "bc_best": str(bc_path),
            "rl_best": str(rl_path),
            "deployed_best": str(dep_path),
        },
        "profiles": profiles,
        "official_test_used_for_selection": False,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print({"saved": str(out), "deployed_from": deployed_from, "rl_improved": rl_improved})
    return meta
