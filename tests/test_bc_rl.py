"""BC→RL primary pipeline: rewards, baselines, checkpoints, gradients."""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class RewardAndBaselineTests(unittest.TestCase):
    def test_normalized_feasible_is_bounded_and_beats_infeasible(self):
        from bc_rl import feasible_reward_bounded, reward_value

        n, tmax = 10, 10.0
        # At most n routes, each <= T_max ⇒ Z_full <= n*T_max.
        self.assertTrue(feasible_reward_bounded(n_routes=4, n_customers=n, z_full=40.0, tmax=tmax))
        feasible = reward_value(
            {"feasible": True, "objective": 40.0},
            n_customers=n,
            max_route_hours=tmax,
            mode="normalized",
        )
        infeasible = reward_value(
            {"feasible": False, "n_unserved": 2},
            n_customers=n,
            max_route_hours=tmax,
            mode="normalized",
            epsilon=0.05,
        )
        worst_feasible = reward_value(
            {"feasible": True, "objective": n * tmax},
            n_customers=n,
            max_route_hours=tmax,
            mode="normalized",
        )
        self.assertGreater(feasible, worst_feasible - 1e-9)
        self.assertGreaterEqual(worst_feasible, -1.0 - 1e-9)
        self.assertLess(infeasible, -1.0)
        self.assertLess(infeasible, feasible)
        # Missing violation is omitted, not invented as a nonzero beta term.
        without = reward_value(
            {"feasible": False, "n_unserved": 0},
            n_customers=n,
            max_route_hours=tmax,
            mode="normalized",
            epsilon=0.05,
            beta=5.0,
        )
        with_viol = reward_value(
            {"feasible": False, "n_unserved": 0, "normalized_violation": 0.2},
            n_customers=n,
            max_route_hours=tmax,
            mode="normalized",
            epsilon=0.05,
            beta=5.0,
        )
        self.assertAlmostEqual(without, -1.05)
        self.assertLess(with_viol, without)

    def test_legacy_reward_unchanged(self):
        from bc_rl import reward_value

        self.assertEqual(
            reward_value({"feasible": True, "objective": 12.5}, n_customers=10, max_route_hours=10, mode="legacy"),
            -12.5,
        )
        self.assertEqual(
            reward_value({"feasible": False}, n_customers=10, max_route_hours=10, mode="legacy"),
            -100.0,
        )

    def test_leave_one_out_excludes_self_and_single_fallback(self):
        from bc_rl import leave_one_out_baselines

        bases = leave_one_out_baselines([1.0, 3.0, 5.0])
        self.assertAlmostEqual(bases[0], 4.0)
        self.assertAlmostEqual(bases[1], 3.0)
        self.assertAlmostEqual(bases[2], 2.0)
        self.assertEqual(leave_one_out_baselines([2.5]), [2.5])

    def test_advantages_are_detached_and_entropy_can_backprop(self):
        from bc_rl import reinforce_loss

        logit = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.1]))
        dist = torch.distributions.Categorical(logits=logit)
        action = torch.tensor(1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        loss = reinforce_loss([log_prob], [0.5], [entropy], entropy_coef=0.01)
        loss.backward()
        self.assertIsNotNone(logit.grad)
        self.assertTrue(torch.isfinite(logit.grad).all())
        self.assertGreater(float(logit.grad.norm()), 0.0)

    def test_zero_advantage_loss_has_no_policy_grad_without_entropy(self):
        from bc_rl import reinforce_loss

        logit = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        dist = torch.distributions.Categorical(logits=logit)
        log_prob = dist.log_prob(torch.tensor(0))
        loss = reinforce_loss([log_prob], [0.0], [None], entropy_coef=0.0)
        loss.backward()
        self.assertTrue(logit.grad is None or float(logit.grad.abs().sum()) == 0.0)

    def test_deployed_keeps_bc_when_rl_is_worse(self):
        from bc_rl import choose_deployed, selection_score

        bc = selection_score(1.0, 20.0)
        worse = selection_score(1.0, 25.0)
        better = selection_score(1.0, 18.0)
        self.assertEqual(choose_deployed(bc, worse), "bc")
        self.assertEqual(choose_deployed(bc, better), "rl")
        self.assertEqual(choose_deployed(bc, None), "bc")

    def test_validation_does_not_hide_infeasible_runs(self):
        from bc_rl import validation_summary

        class Inst:
            customers = (1, 2)
            max_route_hours = 5.0

        summary = validation_summary(
            [
                {"feasible": True, "objective": 4.0},
                {"feasible": False, "objective": float("inf")},
            ],
            [Inst(), Inst()],
        )
        self.assertEqual(summary["n_feasible"], 1)
        self.assertAlmostEqual(summary["feasible_mean_h"], 4.0)
        # Infeasible contributes n*T_max = 10, not a dropped row.
        self.assertAlmostEqual(summary["penalized_mean_h"], 7.0)


class CompatibilityTests(unittest.TestCase):
    def test_primary_refuses_extra_phases_and_historical_version(self):
        from bc_rl import train_primary

        with self.assertRaises(SystemExit):
            train_primary(Namespace(pomo_k=2, repair_rl_steps=0))
        with self.assertRaises(SystemExit):
            train_primary(Namespace(pomo_k=0, repair_rl_steps=3, version="c3"))

    def test_historical_c3_checkpoint_still_loads(self):
        ckpt = ROOT / "checkpoints" / "H2_n10_seed42_c3.pt"
        if not ckpt.exists():
            self.skipTest("Cycle-3 checkpoint not present")
        from train import load_policy

        model = load_policy(ckpt)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_reader_names(self):
        from presentation_names import method_label

        self.assertEqual(method_label("H2"), "Route Transformer (RT)")
        self.assertEqual(method_label("H4"), "Route Transformer + DeepSets (HAT-EVRP-NL)")
        self.assertEqual(method_label("H1"), "State Pointer")
        self.assertEqual(method_label("H3"), "Fleet DeepSets")
        self.assertNotIn("H2", method_label("H2"))


class DecodeGradientTests(unittest.TestCase):
    def test_sampled_decode_logprob_and_entropy_have_grad(self):
        from data import generate_synthetic
        from model import HATPolicy, PolicyConfig
        from train import decode

        torch.manual_seed(0)
        instance = generate_synthetic(5, 3, generator_version="v2")
        model = HATPolicy(PolicyConfig(history=True, deepsets=False, d_model=32, layers=1, heads=4))
        result = decode(model, instance, sample=True, seed=1)
        self.assertIsNotNone(result["log_prob"])
        self.assertIsNotNone(result["entropy_tensor"])
        self.assertGreaterEqual(result["n_unserved"], 0)
        loss = -result["log_prob"] - 0.01 * result["entropy_tensor"]
        loss.backward()
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm())
        self.assertGreater(grad_norm, 0.0)
        self.assertTrue(math_isfinite(grad_norm))


class P2AccountingTests(unittest.TestCase):
    def test_hybrid_oracle_counts_include_both_phases(self):
        import sys

        sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
        from eval_bc_rl_official import combine_oracle_counts

        combined = combine_oracle_counts(
            {"oracle_calls": 55, "oracle_requests": 62, "oracle_cache_hits": 7},
            {"oracle_solver_executions": 400, "oracle_requests": 500, "oracle_cache_hits": 100},
        )
        self.assertEqual(combined["oracle_calls"], 455)
        self.assertEqual(combined["oracle_requests"], 562)
        self.assertEqual(combined["oracle_cache_hits"], 107)

    def test_construction_only_when_hybrid_did_not_run(self):
        import sys

        sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
        from eval_bc_rl_official import combine_oracle_counts

        combined = combine_oracle_counts(
            {"oracle_calls": 55, "oracle_requests": 62, "oracle_cache_hits": 7},
            None,
        )
        self.assertEqual(combined["oracle_calls"], 55)
        self.assertEqual(combined["oracle_requests"], 62)


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    unittest.main()
