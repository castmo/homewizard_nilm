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
    format_duration_sv,
    format_energy_sv,
    generate_idle_diagram_svg,
    slugify_consumer_name,
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

    def test_spike_rejection_ignores_short_spike(self) -> None:
        """Verify short transient power spikes (< 15s) are ignored without user prompts."""
        t = 1_000.0
        # Pre-step baseline: 300W (100W per phase)
        for _ in range(3):
            ev, off = self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=300.0, l1_w=100.0, l2_w=100.0, l3_w=100.0)
            )
            self.assertIsNone(ev)
            self.assertEqual(off, [])
            t += 1.0

        # High power spike (+2000W) starts at t=1003.0
        for _ in range(3):
            ev, off = self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=2300.0, l1_w=2100.0, l2_w=100.0, l3_w=100.0)
            )
            self.assertIsNone(ev)  # Staged for stabilization, no prompt emitted yet
            self.assertEqual(off, [])
            t += 1.0

        self.assertIsNotNone(self.tracker._staged_step)
        assert self.tracker._staged_step is not None
        self.assertEqual(self.tracker._staged_step.start_sample.timestamp, 1003.0)

        # Transient spike ends after 4 seconds (t=1007.0), power drops back to 300W
        t = 1007.0
        ev, off = self.tracker.process_sample(
            PowerSample(timestamp=t, total_w=300.0, l1_w=100.0, l2_w=100.0, l3_w=100.0)
        )
        self.assertIsNone(ev)
        self.assertEqual(off, [])
        # Staged step was dropped as transient spike
        self.assertIsNone(self.tracker._staged_step)
        self.assertEqual(len(self.tracker.get_pending_events()), 0)
        self.assertEqual(len(self.tracker.events), 0)

    def test_stabilization_and_turn_on_details(self) -> None:
        """Verify loads that stabilize emit prompts with start time, duration, and consumed energy."""
        t = 1_000.0
        # Pre-step baseline: 300W
        for _ in range(3):
            self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=300.0, l1_w=100.0, l2_w=100.0, l3_w=100.0)
            )
            t += 1.0

        # Turn on a real 1500W load at t=1003.0 (stabilization window = 15.0s)
        # Feed samples every 1 second until t=1017.0 (14s elapsed)
        for _ in range(15):
            ev, _ = self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=1800.0, l1_w=1600.0, l2_w=100.0, l3_w=100.0)
            )
            self.assertIsNone(ev)  # Still in stabilization window
            t += 1.0

        # At t=1018.0 (1018.0 - 1003.0 = 15.0s elapsed >= 15.0s stabilization_time)
        ev, _ = self.tracker.process_sample(
            PowerSample(timestamp=t, total_w=1800.0, l1_w=1600.0, l2_w=100.0, l3_w=100.0)
        )
        self.assertIsNotNone(ev)
        assert ev is not None
        # Start time is the original step time (t=1003.0), NOT t=1018.0
        self.assertEqual(ev.started_at, 1003.0)
        self.assertAlmostEqual(ev.features.delta_p, 1500.0, places=1)
        self.assertEqual(ev.duration_seconds(now=1018.0), 15.0)
        self.assertEqual(ev.duration_str(now=1018.0), "15s")
        # 1500W for 15s is ~0.00625 kWh (6.25 Wh)
        self.assertGreater(ev.accumulated_kwh, 0.005)
        self.assertIn("Wh", ev.consumed_energy_str())

        # Test dictionary representation for HA attributes
        summary = ev.to_summary_dict(self.tracker.consumers)
        self.assertIn("started_time", summary)
        self.assertIn("duration", summary)
        self.assertEqual(summary["duration_seconds"], 15)
        self.assertGreater(summary["accumulated_wh"], 5.0)
        self.assertGreater(summary["accumulated_kwh"], 0.005)
        self.assertIn("consumed_energy", summary)

    def test_pending_queue_advances_to_next_prompt(self) -> None:
        """Verify pending events are sorted latest-first and advance on labeling."""
        t = 1_000.0
        # Turn on appliance 1 (1500W load)
        ev1, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (1600.0, 100.0, 100.0)
        )
        assert ev1 is not None

        # Turn on appliance 2 (800W load on L2) later in time
        ev2, _, t = feed_step(
            self.tracker, t, (1600.0, 100.0, 100.0), (1600.0, 900.0, 100.0)
        )
        assert ev2 is not None

        # Two pending events queued; latest (ev2) is first item in reverse order!
        pending = self.tracker.get_pending_events()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0].event_id, ev2.event_id)
        self.assertEqual(pending[1].event_id, ev1.event_id)

        # Labeling the default pending event (no event_id specified) labels the latest event (ev2)
        labeled = self.tracker.label_event("boiler")
        assert labeled is not None
        self.assertEqual(labeled.event_id, ev2.event_id)
        self.assertEqual(ev2.assigned_consumer_id, "boiler")

        # Queue advances to prompt 1 (ev1)
        pending_after_first = self.tracker.get_pending_events()
        self.assertEqual(len(pending_after_first), 1)
        self.assertEqual(pending_after_first[0].event_id, ev1.event_id)

        # User labels remaining prompt as "stove"
        self.tracker.label_event("stove", ev1.event_id)

        # Queue is now empty (0 pending)
        self.assertEqual(len(self.tracker.get_pending_events()), 0)

    def test_dismiss_all_pending(self) -> None:
        """Verify dismissing all pending events marks them unknown and clears queue."""
        t = 2_000.0
        ev1, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (1500.0, 100.0, 100.0)
        )
        ev2, _, t = feed_step(
            self.tracker, t, (1500.0, 100.0, 100.0), (1500.0, 800.0, 100.0)
        )
        assert ev1 is not None
        assert ev2 is not None

        self.assertEqual(len(self.tracker.get_pending_events()), 2)

        # Dismiss all pending actions
        dismissed = self.tracker.dismiss_all_pending()
        self.assertEqual(len(dismissed), 2)
        self.assertEqual(len(self.tracker.get_pending_events()), 0)
        self.assertEqual(ev1.assigned_consumer_id, CONSUMER_UNKNOWN_ID)
        self.assertEqual(ev2.assigned_consumer_id, CONSUMER_UNKNOWN_ID)

    def test_swedish_formatting_and_wattage(self) -> None:
        """Verify Swedish duration, energy, and wattage formatting."""
        # Duration formatting
        self.assertEqual(format_duration_sv(15.0), "15 sek")
        self.assertEqual(format_duration_sv(125.0), "2 min 5 sek")
        self.assertEqual(format_duration_sv(120.0), "2 min")
        self.assertEqual(format_duration_sv(3665.0), "1 h 1 min")

        # Energy formatting
        self.assertIn("Wh", format_energy_sv(0.0077))
        self.assertIn("kWh", format_energy_sv(0.250))
        self.assertIn(",", format_energy_sv(0.250))  # Swedish decimal comma

        # ApplianceEvent wattage and Swedish properties
        t = 100.0
        ev, _, _ = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (2250.0, 100.0, 100.0)
        )
        assert ev is not None
        ev.accumulated_kwh = 0.05

        self.assertEqual(ev.wattage_str(), "+2 150 W")
        self.assertIn("sek", ev.duration_str_sv(now=t + 45.0))
        self.assertIn("50 Wh", ev.consumed_energy_str_sv())
        self.assertIn("0,050 kWh", ev.consumed_energy_str_sv())

        summary = ev.to_summary_dict(self.tracker.consumers)
        self.assertEqual(summary["wattage"], 2150)
        self.assertEqual(summary["wattage_str"], "+2 150 W")
        self.assertIn("duration_sv", summary)
        self.assertIn("consumed_energy_sv", summary)

    def test_slugify_swedish_names(self) -> None:
        """Verify Swedish characters are cleanly transliterated in slugify."""
        self.assertEqual(slugify_consumer_name("Tvättmaskin"), "tvattmaskin")
        self.assertEqual(
            slugify_consumer_name("Golvvärme badrum 1"), "golvvarme_badrum_1"
        )
        self.assertEqual(slugify_consumer_name("Spishäll & ugn"), "spishall_ugn")
        self.assertEqual(slugify_consumer_name("Övrigt"), "ovrigt")

    def test_event_diagram_svg_active_and_ended(self) -> None:
        """Verify SVG timeline diagram generation for active and completed events."""
        t = 500.0
        ev, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (1950.0, 100.0, 100.0)
        )
        assert ev is not None
        ev.accumulated_kwh = 0.025

        # 1. Active event: still running
        self.assertTrue(ev.is_active)
        self.assertIsNone(ev.ended_at)
        self.assertIsNone(ev.ended_time_str())
        self.assertEqual(ev.status_str_sv(), "Körs fortfarande")

        svg_active = ev.generate_diagram_svg()
        self.assertIn("<svg", svg_active)
        self.assertIn("Start: kl.", svg_active)
        self.assertIn("Pågår nu...", svg_active)
        self.assertIn("Körs nu", svg_active)
        self.assertIn("+1 850 W", svg_active)

        summary_active = ev.to_summary_dict(self.tracker.consumers)
        self.assertEqual(summary_active["status_sv"], "Körs fortfarande")
        self.assertIsNone(summary_active["ended_time"])
        self.assertEqual(summary_active["diagram_svg"], svg_active)

        # 2. Turn off event: ended
        ev.is_active = False
        ev.ended_at = t + 180.0
        self.assertFalse(ev.is_active)
        self.assertIsNotNone(ev.ended_time_str())
        self.assertEqual(ev.status_str_sv(), "Avslutad")

        svg_ended = ev.generate_diagram_svg()
        self.assertIn("<svg", svg_ended)
        self.assertIn("Start: kl.", svg_ended)
        self.assertIn("Slut: kl.", svg_ended)
        self.assertIn("Avslutad", svg_ended)
        self.assertIn("Kördes i", svg_ended)

        summary_ended = ev.to_summary_dict(self.tracker.consumers)
        self.assertEqual(summary_ended["status_sv"], "Avslutad")
        self.assertIsNotNone(summary_ended["ended_time"])
        self.assertEqual(summary_ended["diagram_svg"], svg_ended)

    def test_low_confidence_does_not_predict(self) -> None:
        """Verify distant/low-confidence signatures don't produce false predictions like 0.0%."""
        # Train classifier only on stove (+2000W on L1)
        t = 100.0
        ev1, _, t = feed_step(
            self.tracker, t, (100.0, 100.0, 100.0), (2100.0, 100.0, 100.0)
        )
        assert ev1 is not None
        self.tracker.label_event("stove", ev1.event_id)

        # Now a 100W light on L3 turns on -> completely different signature
        ev2, _, t = feed_step(
            self.tracker, t, (2100.0, 100.0, 100.0), (2100.0, 100.0, 200.0)
        )
        assert ev2 is not None
        # Confidence should be below threshold and predicted_consumer_id should be None
        self.assertIsNone(ev2.predicted_consumer_id)
        self.assertEqual(ev2.confidence, 0.0)

        summary = ev2.to_summary_dict(self.tracker.consumers)
        self.assertEqual(summary["predicted_consumer"], "Ingen matchning ännu")
        self.assertEqual(summary["confidence_pct"], 0.0)

    def test_single_sample_spike_with_full_buffer_ignored(self) -> None:
        """Verify that when buffer is full, a single high sample followed by normal power is ignored."""
        t = 5000.0
        # Fill buffer with 10 samples of 300W baseline
        for _ in range(10):
            ev, off = self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=300.0, l1_w=100.0, l2_w=100.0, l3_w=100.0)
            )
            self.assertIsNone(ev)
            self.assertEqual(off, [])
            t += 10.0

        # One single spike sample of 2500W at t=5100
        ev, off = self.tracker.process_sample(
            PowerSample(timestamp=t, total_w=2500.0, l1_w=2300.0, l2_w=100.0, l3_w=100.0)
        )
        self.assertIsNone(ev)
        self.assertEqual(off, [])
        t += 10.0

        # Power drops right back to 300W on the next sample
        for _ in range(3):
            ev, off = self.tracker.process_sample(
                PowerSample(timestamp=t, total_w=300.0, l1_w=100.0, l2_w=100.0, l3_w=100.0)
            )
            self.assertIsNone(ev)
            self.assertEqual(off, [])
            t += 10.0

        # No false turn-on or +0W event was generated!
        self.assertEqual(len(self.tracker.get_pending_events()), 0)

    def test_idle_diagram_svg(self) -> None:
        """Verify idle banner SVG generation."""
        svg = generate_idle_diagram_svg()
        self.assertIn("<svg", svg)
        self.assertIn("Alla förbrukare är identifierade", svg)

    def test_active_consumers_and_manual_start_stop(self) -> None:
        """Verify active consumers list, manual starting, and manual stopping."""
        t = 1_700_050_000.0
        self.assertEqual(self.tracker.get_active_consumers(), [])

        # 1. Start Boiler manually at 2000W on L1
        ev_boiler = self.tracker.start_consumer_manually("boiler", power_w=2000.0, phase="L1", timestamp=t)
        self.assertIsNotNone(ev_boiler)
        self.assertEqual(ev_boiler.assigned_consumer_id, "boiler")
        self.assertTrue(ev_boiler.is_active)
        self.assertEqual(self.tracker.get_consumer_power("boiler"), 2000.0)

        # 2. Check get_active_consumers()
        active = self.tracker.get_active_consumers()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["consumer_id"], "boiler")
        self.assertEqual(active[0]["power_w"], 2000.0)

        # 3. Start Stove manually (using default typical wattage ~2000W)
        ev_stove = self.tracker.start_consumer_manually("stove", timestamp=t + 10.0)
        self.assertEqual(len(self.tracker.get_active_consumers()), 2)

        # 4. Stop Boiler manually
        stopped = self.tracker.stop_active_consumer("boiler", timestamp=t + 60.0)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(self.tracker.get_consumer_power("boiler"), 0.0)
        self.assertFalse(ev_boiler.is_active)

        # 5. Only Stove remains active
        active_after = self.tracker.get_active_consumers()
        self.assertEqual(len(active_after), 1)
        self.assertEqual(active_after[0]["consumer_id"], "stove")

        # 6. Stop all remaining
        self.tracker.stop_active_consumer("all", timestamp=t + 120.0)
        self.assertEqual(self.tracker.get_active_consumers(), [])


if __name__ == "__main__":
    unittest.main()
