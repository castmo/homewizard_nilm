"""Unit tests for the HomeWizard NILM ML Disaggregation Engine."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock
import unittest

# Mock homeassistant package in sys.modules for standalone testing
for mod in [
    "voluptuous",
    "homeassistant",
    "homeassistant.config_entries",
    "homeassistant.const",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.config_validation",
    "homeassistant.helpers.dispatcher",
    "homeassistant.helpers.entity",
    "homeassistant.helpers.entity_platform",
    "homeassistant.helpers.event",
    "homeassistant.helpers.selector",
    "homeassistant.helpers.storage",
    "homeassistant.components",
    "homeassistant.components.persistent_notification",
    "homeassistant.components.sensor",
    "homeassistant.components.select",
    "homeassistant.components.button",
    "homeassistant.components.text",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from custom_components.homewizard_nilm.const import (
    CONSUMER_UNKNOWN_ID,
    DEFAULT_CONSUMERS,
)
from custom_components.homewizard_nilm.ml_engine import (
    ApplianceClassifier,
    NILMTracker,
    PowerSample,
)


def feed_step(
    tracker: NILMTracker,
    start_ts: float,
    pre_w: tuple[float, float, float],
    post_w: tuple[float, float, float],
) -> tuple:
    """Feed 3 pre-step samples and 3 post-step samples (10s apart) into the tracker."""
    last_on = None
    all_off = []
    t = start_ts
    for _ in range(3):
        sample = PowerSample(
            timestamp=t,
            total_w=sum(pre_w),
            l1_w=pre_w[0],
            l2_w=pre_w[1],
            l3_w=pre_w[2],
        )
        on_ev, off_evs = tracker.process_sample(sample)
        if on_ev:
            last_on = on_ev
        all_off.extend(off_evs)
        t += 10.0

    for _ in range(3):
        sample = PowerSample(
            timestamp=t,
            total_w=sum(post_w),
            l1_w=post_w[0],
            l2_w=post_w[1],
            l3_w=post_w[2],
        )
        on_ev, off_evs = tracker.process_sample(sample)
        if on_ev:
            last_on = on_ev
        all_off.extend(off_evs)
        t += 10.0

    return last_on, all_off, t


class TestHomeWizardNILMEngine(unittest.TestCase):
    """Test suite for NILMTracker and ApplianceClassifier."""

    def setUp(self) -> None:
        """Create a fresh classifier and tracker."""
        self.classifier = ApplianceClassifier()
        self.tracker = NILMTracker(
            classifier=self.classifier,
            min_power_delta=80.0,
            confidence_threshold=0.75,
        )

    def test_default_consumers_present(self) -> None:
        """Verify all 11 requested consumers are present by default."""
        expected_ids = {
            "boiler",
            "stove",
            "oven",
            "washer",
            "dryer",
            "dishwasher",
            "infrared_heating",
            "floor_heating_bathroom_1",
            "floor_heating_bathroom_2",
            "floor_heating_elsas_bedroom",
            "floor_heating_master_bedroom",
        }
        self.assertEqual(set(DEFAULT_CONSUMERS.keys()), expected_ids)
        self.assertEqual(set(self.tracker.consumers.keys()), expected_ids)

    def test_cold_start_prompts_user_and_learns(self) -> None:
        """Verify cold start prompts user, then auto-identifies once trained."""
        t = 1_700_000_000.0

        # 1st activation of Floor heating Elsa's bedroom (600W on L2) -> Cold Start (unassigned)
        ev1, _, t = feed_step(
            self.tracker, t, (200.0, 100.0, 100.0), (200.0, 700.0, 100.0)
        )
        self.assertIsNotNone(ev1)
        assert ev1 is not None
        self.assertIsNone(ev1.assigned_consumer_id)
        self.assertEqual(len(self.tracker.get_pending_events()), 1)

        # User labels the event as "floor_heating_elsas_bedroom"
        self.tracker.label_event("floor_heating_elsas_bedroom", ev1.event_id)
        self.assertEqual(ev1.assigned_consumer_id, "floor_heating_elsas_bedroom")
        self.assertEqual(len(self.tracker.get_pending_events()), 0)
        self.assertAlmostEqual(
            self.tracker.get_consumer_power("floor_heating_elsas_bedroom"),
            600.0,
            places=1,
        )

        # Turn off Floor heating Elsa's bedroom
        _, off_evs, t = feed_step(
            self.tracker, t, (200.0, 700.0, 100.0), (200.0, 100.0, 100.0)
        )
        self.assertEqual(len(off_evs), 1)
        self.assertEqual(
            self.tracker.get_consumer_power("floor_heating_elsas_bedroom"), 0.0
        )

        # 2nd activation of Floor heating Elsa's bedroom (1 sample gives ~0.73, still prompts once more to be sure)
        ev2, _, t = feed_step(
            self.tracker, t, (200.0, 100.0, 100.0), (200.0, 705.0, 100.0)
        )
        assert ev2 is not None
        self.assertEqual(ev2.predicted_consumer_id, "floor_heating_elsas_bedroom")
        self.tracker.label_event("floor_heating_elsas_bedroom", ev2.event_id)
        _, _, t = feed_step(
            self.tracker, t, (200.0, 705.0, 100.0), (200.0, 100.0, 100.0)
        )

        # 3rd activation: ML model now has high confidence (>= 0.75) and auto-assigns!
        ev3, _, t = feed_step(
            self.tracker, t, (200.0, 100.0, 100.0), (200.0, 698.0, 100.0)
        )
        assert ev3 is not None
        self.assertEqual(ev3.assigned_consumer_id, "floor_heating_elsas_bedroom")
        self.assertGreaterEqual(ev3.confidence, 0.75)
        self.assertEqual(len(self.tracker.get_pending_events()), 0)

    def test_unknown_selection_does_not_train_or_credit(self) -> None:
        """Verify answering 'unknown' clears pending state without training ML."""
        t = 1_700_010_000.0
        ev, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (100.0, 100.0, 1500.0)
        )
        assert ev is not None
        self.assertIsNone(ev.assigned_consumer_id)

        self.tracker.label_event(CONSUMER_UNKNOWN_ID, ev.event_id)
        self.assertEqual(ev.assigned_consumer_id, CONSUMER_UNKNOWN_ID)
        self.assertEqual(len(self.classifier.samples), 0)
        self.assertEqual(len(self.tracker.get_pending_events()), 0)

    def test_add_new_consumer_and_retroactive_energy(self) -> None:
        """Verify adding a new consumer on the fly credits accumulated energy."""
        t = 1_700_020_000.0
        # 1200W load turns on
        ev, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (1300.0, 100.0, 100.0)
        )
        assert ev is not None
        # Let it run for 30 minutes (1800 seconds) while user hasn't answered notification yet
        for _ in range(18):
            t += 100.0
            self.tracker.process_sample(
                PowerSample(
                    timestamp=t, total_w=1500.0, l1_w=1300.0, l2_w=100.0, l3_w=100.0
                )
            )

        # User adds a brand new consumer "Sauna Heater" and labels the pending event
        new_cid = self.tracker.register_consumer("Sauna Heater")
        self.assertEqual(new_cid, "sauna_heater")
        self.tracker.label_event(new_cid, ev.event_id)

        # 1200W for ~1820s is ~0.6067 kWh; verify retroactive energy was credited
        self.assertGreater(self.tracker.consumer_energy_kwh["sauna_heater"], 0.60)
        self.assertEqual(len(self.classifier.samples), 1)
        self.assertEqual(self.classifier.samples[0].consumer_id, "sauna_heater")


if __name__ == "__main__":
    unittest.main()
