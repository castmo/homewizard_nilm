"""Sensor platform for HomeWizard NILM Appliance Disaggregator."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_NEW_CONSUMER_ADDED, SIGNAL_STATE_UPDATED
from .coordinator import HomeWizardNILMCoordinator


def get_hub_device_info(entry: ConfigEntry) -> DeviceInfo:
    """Return DeviceInfo for the HomeWizard NILM Disaggregator hub."""
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="HomeWizard NILM Disaggregator",
        manufacturer="HomeWizard NILM ML",
        model="Non-Intrusive Load Monitor (ML Signature Classifier)",
        entry_type=DeviceEntryType.SERVICE,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeWizard NILM sensors from a config entry."""
    coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SensorEntity] = [
        NILMPendingEventsSensor(coordinator, entry),
        NILMLastEventSensor(coordinator, entry),
    ]

    registered_consumers: set[str] = set()

    for cid, cname in coordinator.tracker.consumers.items():
        registered_consumers.add(cid)
        entities.append(NILMConsumerPowerSensor(coordinator, entry, cid, cname))
        entities.append(NILMConsumerEnergySensor(coordinator, entry, cid, cname))

    async_add_entities(entities)

    @callback
    def _async_on_new_consumer(consumer_id: str) -> None:
        """Dynamically create Power and Energy sensors when a user adds a new consumer."""
        if consumer_id in registered_consumers:
            return
        registered_consumers.add(consumer_id)
        friendly_name = coordinator.tracker.consumers.get(consumer_id, consumer_id)
        async_add_entities(
            [
                NILMConsumerPowerSensor(
                    coordinator, entry, consumer_id, friendly_name
                ),
                NILMConsumerEnergySensor(
                    coordinator, entry, consumer_id, friendly_name
                ),
            ]
        )

    entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            f"{SIGNAL_NEW_CONSUMER_ADDED}_{entry.entry_id}",
            _async_on_new_consumer,
        )
    )


class BaseNILMSensor(SensorEntity):
    """Base class for HomeWizard NILM sensors."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the base sensor."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)

    async def async_added_to_hass(self) -> None:
        """Subscribe to coordinator state updates."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}",
                self.async_write_ha_state,
            )
        )


class NILMConsumerPowerSensor(BaseNILMSensor):
    """Live active power sensor (W) for an individual consumer."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_suggested_display_precision = 1
    _attr_icon = "mdi:flash"

    def __init__(
        self,
        coordinator: HomeWizardNILMCoordinator,
        entry: ConfigEntry,
        consumer_id: str,
        consumer_name: str,
    ) -> None:
        """Initialize the consumer power sensor."""
        super().__init__(coordinator, entry)
        self.consumer_id = consumer_id
        self.consumer_name = consumer_name
        self._attr_unique_id = f"{entry.entry_id}_{consumer_id}_power"
        self._attr_name = f"{consumer_name} Power"

    @property
    def native_value(self) -> float:
        """Return current disaggregated power (W) for this consumer."""
        return self.coordinator.tracker.get_consumer_power(self.consumer_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return diagnostic attributes for this consumer."""
        learned_count = sum(
            1
            for s in self.coordinator.classifier.samples
            if s.consumer_id == self.consumer_id
        )
        active_events = [
            ev.event_id
            for ev in self.coordinator.tracker.events.values()
            if ev.is_active and ev.assigned_consumer_id == self.consumer_id
        ]
        return {
            "consumer_id": self.consumer_id,
            "is_running": len(active_events) > 0,
            "active_event_ids": active_events,
            "learned_signatures": learned_count,
        }


class NILMConsumerEnergySensor(BaseNILMSensor, RestoreSensor):
    """Cumulative energy consumption sensor (kWh) for an individual consumer."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 3
    _attr_icon = "mdi:lightning-bolt"

    def __init__(
        self,
        coordinator: HomeWizardNILMCoordinator,
        entry: ConfigEntry,
        consumer_id: str,
        consumer_name: str,
    ) -> None:
        """Initialize the consumer energy sensor."""
        super().__init__(coordinator, entry)
        self.consumer_id = consumer_id
        self.consumer_name = consumer_name
        self._attr_unique_id = f"{entry.entry_id}_{consumer_id}_energy"
        self._attr_name = f"{consumer_name} Energy"

    async def async_added_to_hass(self) -> None:
        """Restore previous energy state if storage had 0."""
        await super().async_added_to_hass()
        last_data = await self.async_get_last_sensor_data()
        if last_data and last_data.native_value is not None:
            try:
                restored_val = float(last_data.native_value)
                current_val = self.coordinator.tracker.consumer_energy_kwh.get(
                    self.consumer_id, 0.0
                )
                if restored_val > current_val:
                    self.coordinator.tracker.consumer_energy_kwh[self.consumer_id] = (
                        restored_val
                    )
            except (TypeError, ValueError):
                pass

    @property
    def native_value(self) -> float:
        """Return total accumulated energy in kWh."""
        val = self.coordinator.tracker.consumer_energy_kwh.get(self.consumer_id, 0.0)
        return round(val, 4)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return attributes for the energy sensor."""
        return {
            "consumer_id": self.consumer_id,
            "consumer_name": self.consumer_name,
        }


class NILMPendingEventsSensor(BaseNILMSensor):
    """Sensor showing the number of uncertain appliance events awaiting user selection."""

    _attr_icon = "mdi:help-network-outline"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the pending events sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_pending_identifications"
        self._attr_name = "Pending Identifications"

    @property
    def native_value(self) -> int:
        """Return count of events awaiting user confirmation."""
        return len(self.coordinator.tracker.get_pending_events())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Provide full details of the oldest pending event and queue."""
        pending = self.coordinator.tracker.get_pending_events()
        consumers = self.coordinator.tracker.consumers
        if not pending:
            return {
                "current_pending_event": None,
                "current_event_id": None,
                "current_power_w": 0,
                "current_phase": None,
                "current_started_time": None,
                "current_duration": None,
                "current_consumed_energy": None,
                "current_suggested_consumer": None,
                "current_confidence_pct": None,
                "current_summary": "No pending identifications",
                "pending_queue": [],
                "available_consumers": list(consumers.values()),
            }
        latest = pending[0]
        summary = latest.to_summary_dict(consumers)
        return {
            "current_pending_event": summary,
            "current_event_id": latest.event_id,
            "current_power_w": round(latest.features.delta_p, 1),
            "current_phase": latest.features.dominant_phase,
            "current_started_time": latest.started_time_str(),
            "current_duration": latest.duration_str(),
            "current_consumed_energy": latest.consumed_energy_str(),
            "current_suggested_consumer": summary.get("predicted_consumer", "None"),
            "current_confidence_pct": summary.get("confidence_pct", 0.0),
            "current_summary": (
                f"+{round(latest.features.delta_p)} W on {latest.features.dominant_phase} "
                f"({latest.duration_str()} ago, {latest.consumed_energy_str()})"
            ),
            "pending_queue": [ev.to_summary_dict(consumers) for ev in pending],
            "available_consumers": list(consumers.values()),
        }


class NILMLastEventSensor(BaseNILMSensor):
    """Sensor showing the last detected appliance turn-ON event and ML status."""

    _attr_icon = "mdi:brain"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the last detected event sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_detected_event"
        self._attr_name = "Last Detected Event"

    @property
    def native_value(self) -> str:
        """Return a short summary of the last detected event."""
        ev = self.coordinator.last_event
        if ev is None:
            return "No events detected yet"
        delta_w = round(ev.features.delta_p)
        if ev.assigned_consumer_id:
            cname = self.coordinator.tracker.consumers.get(
                ev.assigned_consumer_id, ev.assigned_consumer_id
            )
            return f"{cname} (+{delta_w}W)"
        return f"Awaiting user input (+{delta_w}W)"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return ML model statistics and last event details."""
        ev = self.coordinator.last_event
        consumers = self.coordinator.tracker.consumers
        samples_per_consumer: dict[str, int] = {name: 0 for name in consumers.values()}
        for s in self.coordinator.classifier.samples:
            cname = consumers.get(s.consumer_id, s.consumer_id)
            samples_per_consumer[cname] = samples_per_consumer.get(cname, 0) + 1

        return {
            "last_event": ev.to_summary_dict(consumers) if ev else None,
            "total_learned_signatures": len(self.coordinator.classifier.samples),
            "signatures_per_consumer": samples_per_consumer,
        }
