"""Text platform for adding a new consumer when prompted by HomeWizard NILM."""

from __future__ import annotations

from homeassistant.components.text import TextEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_STATE_UPDATED
from .coordinator import HomeWizardNILMCoordinator
from .sensor import get_hub_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the New Consumer Name Text entity."""
    coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([NILMNewConsumerText(coordinator, entry)])


class NILMNewConsumerText(TextEntity):
    """Text input allowing the user to type a new consumer name when notified."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_icon = "mdi:plus-box-outline"
    _attr_native_min = 0
    _attr_native_max = 64

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the text entity."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_new_consumer_name"
        self._attr_name = "New Consumer Name"

    async def async_added_to_hass(self) -> None:
        """Subscribe to state updates."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}",
                self.async_write_ha_state,
            )
        )

    @property
    def native_value(self) -> str:
        """Return the currently staged new consumer name."""
        return self.coordinator.staged_new_consumer_name

    async def async_set_value(self, value: str) -> None:
        """Immediately create and assign the new consumer upon submission."""
        clean_name = value.strip()
        self.coordinator.staged_new_consumer_name = clean_name
        if clean_name:
            await self.coordinator.async_add_consumer_and_label(name=clean_name)
            self.coordinator.staged_new_consumer_name = ""
        self.async_write_ha_state()
