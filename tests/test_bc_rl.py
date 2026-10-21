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


class InstanceSamplerTests(unittest.TestCase):
    def test_cyclic_rl_budget_misses_part_of_the_training_set(self):
        from bc_rl import cyclic_batch_indices

        seen = []
        for step in range(1, 301):
            seen.extend(cyclic_batch_indices(step, batch_size=2, n_items=500))
        self.assertEqual(min(seen), 0)
        self.assertEqual(max(seen), 300)
        self.assertEqual(len(set(seen)), 301)

    def test_shuffler_covers_every_index_before_repeating(self):
        from bc_rl import EpochShuffler

        sampler = EpochShuffler(500, seed=42)
        first = []
        while len(first) < 500:
            first.extend(sampler.take(2))
        self.assertEqual(sorted(first[:500]), list(range(500)))
        second = []
        while len(second) < 500:
            second.extend(sampler.take(2))
        self.assertEqual(sorted(second[:500]), list(range(500)))
        self.assertNotEqual(first[:500], second[:500])

    def test_shuffler_is_reproducible_and_in_range(self):
        from bc_rl import EpochShuffler

        left = EpochShuffler(17, seed=7)
        right = EpochShuffler(17, seed=7)
        a = left.take(40)
        b = right.take(40)
        self.assertEqual(a, b)
        self.assertTrue(all(0 <= index < 17 for index in a))

    def test_epoch_boundary_does_not_skip_the_last_index(self):
        from bc_rl import EpochShuffler

        sampler = EpochShuffler(5, seed=1)
        drawn = sampler.take(4)
        boundary = sampler.take(2)
        self.assertEqual(len(set(drawn + boundary[:1])), 5)
        self.assertEqual(len(boundary), 2)


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


class NeuralConstructionTests(unittest.TestCase):
    def test_primary_evaluation_does_not_call_postprocessing(self):
        import inspect
        import sys

        sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
        from eval_bc_rl_official import evaluate_neural_construction

        source = inspect.getsource(evaluate_neural_construction)
        self.assertNotIn("local_search", source)
        self.assertNotIn("ruin_and_repair", source)
        self.assertNotIn("hybrid_lns", source)
        self.assertNotIn("local_passes", source)

    def test_construction_decode_does_not_invoke_search(self):
        import solver
        import solver_ls
        from data import generate_synthetic
        from model import HATPolicy, PolicyConfig
        from train import decode

        def forbidden(*_args, **_kwargs):
            raise AssertionError("classical postprocessing was called")

        instance = generate_synthetic(5, 3, generator_version="v2")
        model = HATPolicy(PolicyConfig(history=True, deepsets=False, d_model=32, layers=1, heads=4))
        original_ls = solver.local_search
        original_ruin = solver_ls.ruin_and_repair
        solver.local_search = forbidden
        solver_ls.ruin_and_repair = forbidden
        try:
            result = decode(model, instance, sample=False, seed=0)
        finally:
            solver.local_search = original_ls
            solver_ls.ruin_and_repair = original_ruin
        self.assertIn("feasible", result)

    def test_distribution_kl_is_zero_for_identical_policies(self):
        import copy

        from data import generate_synthetic
        from model import HATPolicy, PolicyConfig
        from train import decode

        torch.manual_seed(0)
        instance = generate_synthetic(5, 3, generator_version="v2")
        model = HATPolicy(PolicyConfig(history=True, deepsets=False, d_model=32, layers=1, heads=4))
        reference = copy.deepcopy(model)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        result = decode(model, instance, sample=True, seed=2, reference_model=reference)
        self.assertIsNotNone(result["distribution_kl"])
        self.assertAlmostEqual(float(result["distribution_kl"].detach()), 0.0, places=4)

    def test_distribution_kl_gradient_moves_toward_frozen_bc(self):
        import copy

        from data import generate_synthetic
        from model import HATPolicy, PolicyConfig
        from train import _masked_categorical_kl, decode

        live_logits = torch.tensor([1.5, -0.5, 0.2, float("-inf")], requires_grad=True)
        frozen_logits = torch.tensor([0.0, 0.0, 0.0, float("-inf")])
        identical = _masked_categorical_kl(frozen_logits, frozen_logits)
        self.assertAlmostEqual(float(identical), 0.0, places=6)
        divergence = _masked_categorical_kl(live_logits, frozen_logits)
        self.assertGreater(float(divergence.detach()), 0.0)
        divergence.backward()
        self.assertTrue(torch.isfinite(live_logits.grad).all())
        self.assertEqual(float(live_logits.grad[-1]), 0.0)
        stepped = (live_logits - 0.5 * live_logits.grad).detach()
        after = _masked_categorical_kl(stepped, frozen_logits)
        self.assertLess(float(after.detach()), float(divergence.detach()))

        torch.manual_seed(0)
        instance = generate_synthetic(5, 3, generator_version="v2")
        frozen = HATPolicy(PolicyConfig(history=True, deepsets=False, d_model=32, layers=1, heads=4))
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        live = copy.deepcopy(frozen)
        with torch.no_grad():
            for parameter in live.parameters():
                parameter.add_(0.2)
        for parameter in live.parameters():
            parameter.requires_grad_(True)
        result = decode(live, instance, sample=False, seed=0, reference_model=frozen)
        self.assertGreater(float(result["distribution_kl"].detach()), 1e-4)
        result["distribution_kl"].backward()
        self.assertTrue(any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in live.parameters()))
        for parameter in frozen.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    unittest.main()
