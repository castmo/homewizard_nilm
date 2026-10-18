"""Button platform for HomeWizard NILM quick actions (Add New Consumer & Mark Unknown)."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONSUMER_UNKNOWN_ID, DOMAIN
from .coordinator import HomeWizardNILMCoordinator
from .sensor import get_hub_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeWizard NILM action buttons."""
    coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            NILMConfirmNewConsumerButton(coordinator, entry),
            NILMMarkUnknownButton(coordinator, entry),
            NILMDismissAllPendingButton(coordinator, entry),
        ]
    )


class NILMConfirmNewConsumerButton(ButtonEntity):
    """Button that adds the consumer typed in 'New Consumer Name' and labels the pending event."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:ToyBrickPlus"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the button."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_confirm_new_consumer"
        self._attr_name = "Add New Consumer & Assign"
        self._attr_icon = "mdi:plus-circle"

    async def async_press(self) -> None:
        """Handle button press."""
        name = self.coordinator.staged_new_consumer_name.strip()
        if not name:
            raise HomeAssistantError(
                "Please enter a name in 'New Consumer Name' before pressing this button."
            )
        await self.coordinator.async_add_consumer_and_label(name=name)
        self.coordinator.staged_new_consumer_name = ""


class NILMMarkUnknownButton(ButtonEntity):
    """Button that marks the current pending appliance event as 'Unknown'."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:help-circle-outline"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the button."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_mark_unknown"
        self._attr_name = "Mark Pending Event as Unknown"

    async def async_press(self) -> None:
        """Handle button press."""
        await self.coordinator.async_label_event(consumer_id=CONSUMER_UNKNOWN_ID)


class NILMDismissAllPendingButton(ButtonEntity):
    """Button that dismisses all pending appliance identification events at once."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:notification-clear-all"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the button."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_dismiss_all_pending"
        self._attr_name = "Dismiss All Pending Prompts"

    async def async_press(self) -> None:
        """Handle button press to dismiss all pending events."""
        await self.coordinator.async_dismiss_all_pending()
