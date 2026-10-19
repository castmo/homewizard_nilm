"""The HomeWizard NILM Appliance Disaggregator integration."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    SERVICE_ADD_CONSUMER,
    SERVICE_DISMISS_ALL_PENDING,
    SERVICE_LABEL_EVENT,
    SERVICE_RESET_MODEL,
)
from .coordinator import HomeWizardNILMCoordinator

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.SELECT,
    Platform.TEXT,
    Platform.BUTTON,
    Platform.IMAGE,
]

SCHEMA_LABEL_EVENT = vol.Schema(
    {
        vol.Required("consumer"): cv.string,
        vol.Optional("event_id"): cv.string,
    }
)

SCHEMA_ADD_CONSUMER = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Optional("event_id"): cv.string,
        vol.Optional("assign_pending_event", default=True): cv.boolean,
    }
)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the HomeWizard NILM component."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HomeWizard NILM from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    coordinator = HomeWizardNILMCoordinator(hass, entry)
    await coordinator.async_setup()
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _async_register_services(hass)
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_unload()
    return unload_ok


def _async_register_services(hass: HomeAssistant) -> None:
    """Register domain services for labeling events, adding consumers, and resetting ML."""
    if hass.services.has_service(DOMAIN, SERVICE_LABEL_EVENT):
        return

    async def _handle_label_event(call: ServiceCall) -> None:
        consumer: str = call.data["consumer"]
        event_id: str | None = call.data.get("event_id")
        for coordinator in hass.data.get(DOMAIN, {}).values():
            await coordinator.async_label_event(
                consumer_id=consumer, event_id=event_id
            )

    async def _handle_add_consumer(call: ServiceCall) -> None:
        name: str = call.data["name"]
        event_id: str | None = call.data.get("event_id")
        assign_pending: bool = call.data.get("assign_pending_event", True)
        for coordinator in hass.data.get(DOMAIN, {}).values():
            if assign_pending:
                await coordinator.async_add_consumer_and_label(
                    name=name, event_id=event_id
                )
            else:
                from .const import SIGNAL_NEW_CONSUMER_ADDED, SIGNAL_STATE_UPDATED
                from homeassistant.helpers.dispatcher import async_dispatcher_send
                from .ml_engine import slugify_consumer_name

                cid = slugify_consumer_name(name)
                is_new = cid not in coordinator.tracker.consumers
                coordinator.tracker.register_consumer(name, cid)
                if is_new:
                    async_dispatcher_send(
                        hass,
                        f"{SIGNAL_NEW_CONSUMER_ADDED}_{coordinator.entry.entry_id}",
                        cid,
                    )
                await coordinator.async_save_storage()
                async_dispatcher_send(
                    hass, f"{SIGNAL_STATE_UPDATED}_{coordinator.entry.entry_id}"
                )

    async def _handle_reset_model(_call: ServiceCall) -> None:
        for coordinator in hass.data.get(DOMAIN, {}).values():
            await coordinator.async_reset_model()

    async def _handle_dismiss_all_pending(_call: ServiceCall) -> None:
        for coordinator in hass.data.get(DOMAIN, {}).values():
            await coordinator.async_dismiss_all_pending()

    hass.services.async_register(
        DOMAIN, SERVICE_LABEL_EVENT, _handle_label_event, schema=SCHEMA_LABEL_EVENT
    )
    hass.services.async_register(
        DOMAIN, SERVICE_ADD_CONSUMER, _handle_add_consumer, schema=SCHEMA_ADD_CONSUMER
    )
    hass.services.async_register(
        DOMAIN, SERVICE_RESET_MODEL, _handle_reset_model
    )
    hass.services.async_register(
        DOMAIN, SERVICE_DISMISS_ALL_PENDING, _handle_dismiss_all_pending
    )
