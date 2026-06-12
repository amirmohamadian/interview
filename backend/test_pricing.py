"""Tests for conversion-optimized pricing catalog and credit flows."""

import os
import tempfile
import unittest

import database as db
import pricing as P
from credits import CreditError, consume_exam_credit, purchase_product, register_user


class PricingCatalogTests(unittest.TestCase):
    def test_max_exams_is_20(self):
        self.assertEqual(P.MAX_EXAMS, 20)

    def test_bundle_ladder_is_10_15_20(self):
        sizes = [b.credits for b in P.CREDIT_BUNDLES]
        self.assertEqual(sizes, [10, 15, 20])

    def test_unlimited_subscription_capped_at_20(self):
        unlimited = next(s for s in P.SUBSCRIPTIONS if s.id == "sub_unlimited")
        self.assertEqual(unlimited.credits, 20)

    def test_subscriber_refills_cheaper_than_matching_bundles(self):
        for bundle in P.CREDIT_BUNDLES:
            refill = next(
                (r for r in P.SUBSCRIBER_REFILLS if r.credits == bundle.credits),
                None,
            )
            if refill:
                self.assertLess(refill.per_exam_cents, bundle.per_exam_cents)
                savings = P.bundle_savings_vs_refill(bundle, refill)
                self.assertGreaterEqual(savings, 35)

    def test_subscriptions_cheaper_per_exam_than_bundles(self):
        for sub in P.SUBSCRIPTIONS:
            cheapest_bundle = min(b.per_exam_cents for b in P.CREDIT_BUNDLES)
            self.assertLess(sub.per_exam_cents, cheapest_bundle)

    def test_catalog_includes_conversion_notes(self):
        catalog = P.pricing_catalog()
        self.assertEqual(catalog["conversion_notes"]["primary_cta"], "subscription")
        self.assertIn("10 / 15 / 20", catalog["conversion_notes"]["bundle_ladder"])


class CreditFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db.DB_PATH = os.path.join(self._tmpdir.name, "test.db")
        db.init_db()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_one_free_exam_per_email(self):
        result = register_user(email="free@example.com")
        self.assertEqual(len(result["free_granted"]), 1)
        self.assertEqual(result["user"]["balance"], 1)

        again = register_user(email="free@example.com")
        self.assertEqual(len(again["free_granted"]), 0)

    def test_one_free_exam_per_api_key(self):
        result = register_user(api_key="key-alpha")
        self.assertEqual(result["user"]["balance"], 1)

        other = register_user(api_key="key-beta")
        self.assertEqual(other["user"]["balance"], 1)

    def test_email_and_api_key_each_get_free_credit(self):
        result = register_user(email="both@example.com", api_key="key-both")
        self.assertEqual(len(result["free_granted"]), 2)
        self.assertEqual(result["user"]["balance"], 2)

    def test_subscriber_refill_requires_subscription(self):
        register_user(email="nosub@example.com")
        with self.assertRaises(CreditError) as ctx:
            purchase_product("refill_10", email="nosub@example.com")
        self.assertEqual(ctx.exception.code, "subscription_required")

    def test_subscriber_can_buy_discounted_refill(self):
        register_user(email="sub@example.com")
        purchase_product("sub_pro", email="sub@example.com")
        result = purchase_product("refill_5", email="sub@example.com")
        self.assertEqual(result["purchase"]["credits_granted"], 5)
        self.assertGreater(result["account"]["user"]["balance"], 0)

    def test_consume_blocks_without_credits(self):
        register_user(email="empty@example.com")
        consume_exam_credit(email="empty@example.com")
        with self.assertRaises(CreditError) as ctx:
            consume_exam_credit(email="empty@example.com")
        self.assertEqual(ctx.exception.code, "insufficient_credits")

    def test_bundle_purchase_adds_credits(self):
        register_user(email="buyer@example.com")
        purchase_product("bundle_15", email="buyer@example.com")
        account = register_user(email="buyer@example.com")
        # 1 free + 15 bundle
        self.assertEqual(account["user"]["balance"], 16)


if __name__ == "__main__":
    unittest.main()
