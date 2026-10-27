"""Offline control-loop tests. These are not real model performance results."""
import json
from pathlib import Path
import tempfile
import unittest

from run import Harbour
from llm_master import MasterHarbour, parse_validate, THOUGHTS


def proposal(report, name="warn_storm", arguments=None):
    return json.dumps({"Thought": THOUGHTS[0],
                       "Action": {"name": name, "arguments": arguments or {}},
                       "Final Report": report["facts"][0]})


def extract_report(messages):
    return json.loads(messages[1]["content"].split("TRUSTED_REPORT\n", 1)[1]
                      .split("\nUNTRUSTED_DATA\n", 1)[0])


class FakeBackend:
    metadata = {"model": "OFFLINE_TEST_DOUBLE", "not_live_model": True}

    def __init__(self, mode="valid"):
        self.mode = mode
        self.calls = 0

    def call(self, messages):
        self.calls += 1
        if self.mode == "invalid" or (self.mode == "retry" and self.calls == 1):
            return "not JSON", {"test_double": True}
        report = extract_report(messages)
        if self.mode == "close":
            return proposal(report, "close_harbour"), {"test_double": True}
        return proposal(report), {"test_double": True}


class MasterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def harbour(self, mode="valid", **kwargs):
        return MasterHarbour(3, 1, FakeBackend(mode),
                             log_path=self.directory / "live.jsonl", **kwargs)

    def test_unsupported_fact_and_unknown_action_rejected(self):
        h = self.harbour(max_ticks=40)
        report = h.report(0)
        data = json.loads(proposal(report))
        data["Final Report"] = "Ship 999 carries gold."
        with self.assertRaises(ValueError):
            parse_validate(json.dumps(data), report)
        with self.assertRaises(ValueError):
            parse_validate(proposal(report, "delete_everything"), report)

    def test_arguments_and_nonexistent_ship_rejected(self):
        h = self.harbour()
        h.update_ships(8)
        report = h.report(8)
        for name, arguments in (
            ("hold_ship", {"ship_id": 999}),
            ("assign_workers", {"berth": "B1", "n": -1}),
            ("assign_workers", {"berth": "B1", "n": True}),
            ("assign_workers", {"berth": "B1", "n": 17}),
            ("assign_workers", {"berth": "B9", "n": 1}),
        ):
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(ValueError):
                    parse_validate(proposal(report, name, arguments), report)

    def test_both_protected_actions_blocked_without_state_change(self):
        h = self.harbour()
        h.update_ships(8)
        cargo = h.ships[0]["left"]
        result, approval = h.execute({"name": "close_harbour", "arguments": {}}, 8)
        self.assertFalse(approval)
        self.assertIn("BLOCKED", result)
        self.assertFalse(h.closed)
        result, approval = h.execute({"name": "release_cargo", "arguments": {"ship_id": 1}}, 8)
        self.assertFalse(approval)
        self.assertEqual(h.ships[0]["left"], cargo)
        self.assertEqual(h.released_cargo, 0)
        self.assertEqual(h.stats["blocked_approvals"], 2)

    def test_forged_approval_field_is_rejected(self):
        h = self.harbour()
        report = h.report(0)
        data = json.loads(proposal(report, "close_harbour"))
        data["approved"] = True
        with self.assertRaises(ValueError):
            parse_validate(json.dumps(data), report)

    def test_assignments_have_real_effect_and_are_disjoint(self):
        h = self.harbour()
        h.update_ships(8)
        h.update_ships(16)
        h.execute({"name": "assign_workers", "arguments": {"berth": "B1", "n": 4}}, 16)
        h.execute({"name": "assign_workers", "arguments": {"berth": "B2", "n": 4}}, 16)
        self.assertEqual(len(h.orders["B1"]), 4)
        self.assertTrue(set(h.orders["B1"]).isdisjoint(h.orders["B2"]))
        with self.assertRaises(ValueError):
            h.execute({"name": "assign_workers", "arguments": {"berth": "B2", "n": 16}}, 16)

    def test_warning_memory_and_hold(self):
        h = self.harbour()
        start = min(h.storm_starts)
        h.update_weather(start - 5)
        h.execute({"name": "warn_storm", "arguments": {}}, start - 5)
        self.assertTrue(all(w["unsafe_until"] == start + 15 for w in h.workers))
        h.ships = [{"id": 99, "arrival": 0, "dock": None, "berth": None, "left": 9}]
        h.execute({"name": "hold_ship", "arguments": {"ship_id": 99}}, 1)
        h.update_ships(2)
        self.assertIsNone(h.ships[0]["dock"])
        h.update_ships(21)
        self.assertEqual(h.ships[0]["dock"], 21)

    def test_retry_and_action_limit(self):
        h = self.harbour(mode="retry", max_ticks=40)
        result = h.run()
        self.assertEqual(result["calls"], 3)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertEqual(result["retries"], 1)
        self.assertEqual(result["fallbacks"], 0)
        h = self.harbour(max_ticks=1, actions_per_turn=3)
        self.assertEqual(h.run()["calls"], 3)

    def test_exhausted_retries_fall_back_to_reflex(self):
        h = self.harbour(mode="invalid")
        result = h.run()
        expected = Harbour(3, 1, "reflex").run()
        for key, value in expected.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["calls"], 90)
        self.assertEqual(result["retries"], 60)
        self.assertEqual(result["fallbacks"], 30)

    def test_replay_reuses_responses_and_checks_prompts(self):
        live = self.harbour(mode="retry", max_ticks=40)
        first = live.run()
        replay = MasterHarbour(3, 1, replay=self.directory / "live.jsonl",
                               log_path=self.directory / "replay.jsonl", max_ticks=40)
        second = replay.run()
        for key in first:
            if key not in ("live_calls", "replay_calls"):
                self.assertEqual(first[key], second[key], key)
        self.assertEqual(second["live_calls"], 0)
        self.assertEqual(second["replay_calls"], 3)
        with self.assertRaises(RuntimeError):
            MasterHarbour(3, 2, replay=self.directory / "live.jsonl",
                          log_path=self.directory / "other.jsonl", max_ticks=40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
