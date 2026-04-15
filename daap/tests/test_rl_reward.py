"""
Tests for daap/rl/reward.py — reward computation, normalization, edge cases.
Pure unit tests. No SQLite, no network, no agentscope.
"""

from daap.rl.reward import (
    compute_reward,
    _normalize_rating,
    _cost_efficiency,
    _latency_efficiency,
    _estimate_cost_from_tokens,
    COST_BASELINE_USD,
    COST_CEILING_USD,
    LATENCY_BASELINE_SECONDS,
    LATENCY_CEILING_SECONDS,
)


class TestNormalizeRating:
    def test_rating_1_maps_to_zero(self):
        assert _normalize_rating(1) == 0.0

    def test_rating_5_maps_to_one(self):
        assert _normalize_rating(5) == 1.0

    def test_rating_3_maps_to_half(self):
        assert abs(_normalize_rating(3) - 0.5) < 1e-9

    def test_rating_clamped_below(self):
        assert _normalize_rating(0) == 0.0

    def test_rating_clamped_above(self):
        assert _normalize_rating(6) == 1.0


class TestCostEfficiency:
    def test_at_baseline_is_one(self):
        assert _cost_efficiency(COST_BASELINE_USD) == 1.0

    def test_below_baseline_is_one(self):
        assert _cost_efficiency(0.0) == 1.0

    def test_at_ceiling_is_zero(self):
        assert _cost_efficiency(COST_CEILING_USD) == 0.0

    def test_above_ceiling_is_zero(self):
        assert _cost_efficiency(COST_CEILING_USD * 10) == 0.0

    def test_midpoint_is_half(self):
        mid = (COST_BASELINE_USD + COST_CEILING_USD) / 2
        eff = _cost_efficiency(mid)
        assert abs(eff - 0.5) < 0.01


class TestLatencyEfficiency:
    def test_at_baseline_is_one(self):
        assert _latency_efficiency(LATENCY_BASELINE_SECONDS) == 1.0

    def test_below_baseline_is_one(self):
        assert _latency_efficiency(0.0) == 1.0

    def test_at_ceiling_is_zero(self):
        assert _latency_efficiency(LATENCY_CEILING_SECONDS) == 0.0

    def test_above_ceiling_is_zero(self):
        assert _latency_efficiency(LATENCY_CEILING_SECONDS * 2) == 0.0


class TestComputeReward:
    def _make_result(self, success=True, latency=10.0, input_tokens=1000, output_tokens=500):
        return {
            "success": success,
            "latency_seconds": latency,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "error": None,
        }

    def test_rated_5_star_fast_cheap_gives_high_reward(self):
        result = self._make_result(success=True, latency=8.0, input_tokens=500, output_tokens=200)
        reward = compute_reward(result, rating=5)
        assert reward > 0.85, f"Expected > 0.85, got {reward}"

    def test_rated_1_star_gives_low_reward(self):
        # rating=1 → 0.0*0.6 + cost_eff*0.2 + lat_eff*0.2 ≤ 0.4 (cost+latency terms only)
        result = self._make_result(success=True, latency=8.0)
        reward = compute_reward(result, rating=1)
        assert reward < 0.45, f"Expected < 0.45, got {reward}"

    def test_failed_unrated_gives_low_reward(self):
        result = self._make_result(success=False)
        reward = compute_reward(result, rating=None)
        # success=0 → 0*0.5 + cost_eff*0.25 + lat_eff*0.25 ≤ 0.5
        assert reward <= 0.5, f"Expected <= 0.5, got {reward}"

    def test_successful_unrated_gives_moderate_reward(self):
        # success=1, latency=90s (high) → lower lat_eff pulls reward down
        result = self._make_result(success=True, latency=90.0)
        reward = compute_reward(result, rating=None)
        assert 0.5 < reward < 0.9, f"Expected 0.5-0.9, got {reward}"

    def test_reward_always_in_unit_interval(self):
        for rating in [None, 1, 3, 5]:
            for success in [True, False]:
                for latency in [1.0, 60.0, 200.0]:
                    result = self._make_result(success=success, latency=latency)
                    r = compute_reward(result, rating=rating)
                    assert 0.0 <= r <= 1.0, f"Out of [0,1]: {r}"

    def test_graceful_on_empty_result(self):
        reward = compute_reward({})
        assert 0.0 <= reward <= 1.0

    def test_rated_5_dominates(self):
        result = {"success": True, "latency_seconds": 5.0,
                  "total_input_tokens": 100, "total_output_tokens": 50}
        reward = compute_reward(result, rating=5)
        assert reward > 0.9
