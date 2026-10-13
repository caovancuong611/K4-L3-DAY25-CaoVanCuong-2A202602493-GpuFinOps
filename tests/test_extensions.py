"""Tests for the "Your Turn" extensions (student-written; the original 15 tests are untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_aware

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}
L4 = {"on_demand_hr": 0.8, "spot_hr": 0.35, "reserved_1yr_hr": 0.6, "reserved_3yr_hr": 0.45}


# --- Ext 1 ---------------------------------------------------------------
def test_recommend_tier_default_policy_unchanged():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_reserved_term_depends_on_lifetime():
    long = pricing.recommend_tier_detailed(24, False, H100, gpu_type="H100", job_days=3 * 365)
    short = pricing.recommend_tier_detailed(24, False, H100, gpu_type="H100", job_days=365)
    assert (long["tier"], long["term"]) == ("reserved", "3yr")
    assert (short["tier"], short["term"]) == ("reserved", "1yr")


def test_reserved_is_billed_24_7():
    # 4h/day: paying a 24h reservation is worse than on-demand
    d = pricing.recommend_tier_detailed(4, False, L4, gpu_type="L4", job_days=3 * 365)
    assert d["tier"] == "on_demand"
    assert abs(d["options"]["reserved_3yr"] - 24 * 0.45) < 1e-9


def test_interrupt_rate_raises_spot_cost():
    calm = pricing.recommend_tier_detailed(12, True, H100, interrupt_rate=0.01)
    noisy = pricing.recommend_tier_detailed(12, True, H100, interrupt_rate=0.30)
    assert calm["options"]["spot"] < noisy["options"]["spot"]
    assert pricing.SPOT_INTERRUPT_RATE["H100"] < pricing.SPOT_INTERRUPT_RATE["A10G"]


def test_m3_new_policy_is_not_more_optimistic_than_legacy():
    r = m3_purchasing.run(verbose=False)
    assert r["optimized_monthly"] >= r["legacy_monthly"]       # legacy under-billed reserved
    assert {"spot", "reserved"} <= {x["tier"] for x in r["recommendations"]}


# --- Ext 2 ---------------------------------------------------------------
def test_right_size_never_picks_a_gpu_that_does_not_fit():
    r1 = m1_efficiency_audit.run(verbose=False)
    by_id = {x["gpu_id"]: x for x in r1["rightsizing"]}
    assert by_id["gpu-a10g-0"]["proposed"] == "L4"            # fits: 24GB, MBU 47%
    assert by_id["gpu-a100-0"]["proposed"] == "A100"          # 68GB used: no cheaper card fits
    assert all(x["proposed_mbu"] <= m1_efficiency_audit.TARGET_MBU for x in r1["rightsizing"])
    assert r1["rightsize_monthly"] > 0


# --- Ext 3 ---------------------------------------------------------------
def test_cache_break_even():
    # Anthropic-style: 1.25x write, 0.1x read -> 0.25 / 0.9 reads
    assert abs(pricing.cache_break_even_reads(1.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert not pricing.cache_is_worth_it(0, 1.25, 0.10)
    assert pricing.cache_is_worth_it(1, 1.25, 0.10)
    # storage makes caching harder to justify
    assert not pricing.cache_is_worth_it(3, 1.25, 0.10, storage_cost=5.0)
    assert pricing.cache_is_worth_it(10, 1.25, 0.10, storage_cost=5.0)
    assert pricing.cache_break_even_reads(read_discount=1.0) == float("inf")


def test_m2_cache_is_net_of_overhead():
    r2 = m2_inference_levers.run(verbose=False)
    c = r2["cache"]
    assert 0 < c["net_saving_daily"] < c["naive_saving_daily"]
    assert c["tiers"]["small"]["break_even_reads"] > c["tiers"]["large"]["break_even_reads"]


# --- Ext 4 ---------------------------------------------------------------
def test_reasoning_budget_shares():
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rb["energy_share"] > rb["cost_share"] > rb["traffic_share"] > 0
    assert rb["caps"][0.10]["wh"] <= rb["caps"][0.05]["wh"] <= rb["caps"][0.02]["wh"]


# --- Ext 5 ---------------------------------------------------------------
def test_best_regions_and_carbon_move():
    best = sustainability.best_regions()
    assert best["cleanest"] == "europe-north1" and best["cheapest"] == "us-east-wa"
    assert sustainability.best_regions(weight_cost=0.0)["balanced"] == "europe-north1"
    r6 = m6_carbon_aware.run(verbose=False)
    assert r6["carbon_saved_kg"] > 0 and 0 < r6["carbon_saved_pct"] < 100
