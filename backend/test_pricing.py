"""Tests for the pricing / entitlement engine."""

import importlib

import pricing as P


def fresh_store():
    return P.Store(path=None)


def test_catalog_caps_at_max_value():
    assert P.PRICING["max_value"] == 20
    for b in P.PRICING["bundles"]:
        assert b["exams"] <= P.MAX_VALUE
    pro = P.PRICING["subscriptions"][0]
    assert pro["monthly_exams"] <= P.MAX_VALUE
    unlimited = P.PRICING["subscriptions"][1]
    assert unlimited["daily_cap"] == P.UNLIMITED_DAILY_CAP == 20


def test_topup_cheaper_than_every_bundle():
    cheapest_bundle = min(b["per_exam"] for b in P.PRICING["bundles"])
    assert P.TOPUP_PRICE < cheapest_bundle


def test_one_free_exam_per_email():
    s = fresh_store()
    acc = s.get_or_create(email="a@x.com")
    assert s.consume(acc) == "free"
    # Second time: no entitlement -> blocked.
    try:
        s.consume(acc)
        assert False, "should have blocked"
    except PermissionError as e:
        assert str(e) == "no_entitlement"


def test_free_exam_per_api_key_independent():
    s = fresh_store()
    a = s.get_or_create(api_key="key-1")
    b = s.get_or_create(api_key="key-2")
    assert s.consume(a) == "free"
    assert s.consume(b) == "free"  # different api key => its own free exam


def test_email_and_apikey_merge_to_one_account():
    s = fresh_store()
    a = s.get_or_create(email="merge@x.com")
    again = s.get_or_create(email="merge@x.com", api_key="k9")
    assert a.id == again.id  # same account, not a second free exam


def test_bundle_credits_used_after_free():
    s = fresh_store()
    acc = s.get_or_create(email="buyer@x.com")
    s.consume(acc)                       # free
    bundle = P.PRICING["bundles"][0]     # 5 exams
    s.add_credits(acc, bundle["exams"])
    sources = [s.consume(acc) for _ in range(5)]
    assert sources == ["credits"] * 5
    assert acc.credits == 0


def test_subscription_monthly_allowance_then_topup():
    s = fresh_store()
    acc = s.get_or_create(email="pro@x.com")
    assert s.consume(acc) == "free"   # free exam spent before subscribing
    s.subscribe(acc, "sub_pro")
    allowance = P.PRICING["subscriptions"][0]["monthly_exams"]
    for _ in range(allowance):
        assert s.consume(acc) == "subscription"
    # Allowance exhausted -> blocked until top-up.
    try:
        s.consume(acc)
        assert False
    except PermissionError:
        pass
    s.add_credits(acc, 3)  # top-up
    assert s.consume(acc) == "credits"


def test_unlimited_daily_cap():
    s = fresh_store()
    acc = s.get_or_create(email="unlimited@x.com")
    s.subscribe(acc, "sub_unlimited")
    for _ in range(P.UNLIMITED_DAILY_CAP):
        assert s.consume(acc) == "subscription"
    try:
        s.consume(acc)
        assert False, "daily cap should block"
    except PermissionError as e:
        assert str(e) == "daily_cap_reached"


def test_persistence_roundtrip(tmp_path):
    db = tmp_path / "db.json"
    s1 = P.Store(path=str(db))
    acc = s1.get_or_create(email="persist@x.com")
    s1.consume(acc)
    s1.add_credits(acc, 4)
    # Reload from disk.
    s2 = P.Store(path=str(db))
    reloaded = s2.find(email="persist@x.com")
    assert reloaded is not None
    assert reloaded.free_used is True
    assert reloaded.credits == 4


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                # crude tmp_path support for the one test that needs it
                if "tmp_path" in fn.__code__.co_varnames:
                    import tempfile, pathlib
                    with tempfile.TemporaryDirectory() as d:
                        fn(pathlib.Path(d))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
