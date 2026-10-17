# HomeWizard NILM Appliance Disaggregator (`homewizard_nilm`)

A custom Home Assistant integration that uses **Non-Intrusive Load Monitoring (NILM)** and **Online Machine Learning** to identify individual appliance power signatures from your **HomeWizard P1 Meter** (or Energy Socket) and create individual **Power (`W`)** and **Energy (`kWh`)** sensors for each consumer.

---

## Pre-Configured Consumers

Out of the box, the integration creates live **Power (`W`)** and cumulative **Energy (`kWh`, Energy Dashboard compatible)** sensors for:

1. **Boiler** (`sensor.homewizard_nilm_disaggregator_boiler_power`, `sensor.homewizard_nilm_disaggregator_boiler_energy`)
2. **Stove** (`..._stove_power`, `..._stove_energy`)
3. **Oven** (`..._oven_power`, `..._oven_energy`)
4. **Washer** (`..._washer_power`, `..._washer_energy`)
5. **Dryer** (`..._dryer_power`, `..._dryer_energy`)
6. **Dishwasher** (`..._dishwasher_power`, `..._dishwasher_energy`)
7. **Infrared heating** (`..._infrared_heating_power`, `..._infrared_heating_energy`)
8. **Floor heating Bathroom 1** (`..._floor_heating_bathroom_1_power`, `..._floor_heating_bathroom_1_energy`)
9. **Floor heating Bathroom 2** (`..._floor_heating_bathroom_2_power`, `..._floor_heating_bathroom_2_energy`)
10. **Floor heating Elsa's bedroom** (`..._floor_heating_elsas_bedroom_power`, `..._floor_heating_elsas_bedroom_energy`)
11. **Floor heating Master bedroom** (`..._floor_heating_master_bedroom_power`, `..._floor_heating_master_bedroom_energy`)

Plus any **new consumers** you dynamically add when prompted!

---

## How the Machine Learning & User Prompting Works

1. **Step-Edge & Signature Extraction**:
   - The integration monitors your HomeWizard total active power sensor (and optional 3-phase `L1`, `L2`, `L3` power sensors).
   - When an appliance turns on ($\Delta P \ge \text{min\_power\_delta}$, default `80 W`), it extracts a multi-dimensional feature vector:
     - **Steady-State Power Step ($\Delta P$)** in Watts
     - **Transient Inrush / Peak Overshoot** in Watts
     - **Steady-State Ripple / Standard Deviation** (distinguishes resistive heaters like floor heating/boiler from motor/pulsing loads like washer/dryer/stove)
     - **3-Phase Distribution (`L1`, `L2`, `L3` shares)** (pinpoints which phase a specific bedroom/bathroom floor heating mat or appliance is wired to)
     - **Cyclical Time-of-Day Prior** ($\sin/\cos$ of hour)
2. **Online Classification & Cold-Start / Uncertainty Notification**:
   - An online Weighted $k$-NN + RBF Kernel Prototype classifier compares the extracted signature against stored labeled signatures in `.storage/homewizard_nilm.ml_store.<entry_id>`.
   - **Cold Start / Low Confidence ($< 75\%$)**:
     - The event is placed in the **Pending Identifications** queue (while its active power and accumulated kWh are tracked in a staging buffer so no energy is lost while waiting for your answer).
     - A **Home Assistant Persistent Notification** (and optional **Companion App Mobile Notification**) is sent to prompt you.
   - **Your 3 Ways to Answer a Prompt**:
     1. **Select a Known Consumer**: Pick from the dropdown (`select.homewizard_nilm_disaggregator_identify_consumer`) or tap the quick-action button on your phone notification. The ML model immediately learns the signature and credits the event's accumulated energy to that consumer.
     2. **Answer `"Unknown"`**: Select `Unknown` in the dropdown, press the **Mark Pending Event as Unknown** button (`button.homewizard_nilm_disaggregator_mark_pending_event_as_unknown`), or tap `Unknown` on the mobile notification. The event is dismissed without polluting ML training data.
     3. **Add a New Consumer on the Fly**: Type the new consumer's name into `text.homewizard_nilm_disaggregator_new_consumer_name` and press `button.homewizard_nilm_disaggregator_add_new_consumer_assign` (or tap **Other / Add New...** on the mobile notification and type the name). The integration dynamically creates `<New Consumer> Power` (`W`) and `<New Consumer> Energy` (`kWh`) sensors on the spot, trains the ML model with the event signature, and assigns the event's energy to the new consumer!

---

## Installation

1. Copy the `custom_components/homewizard_nilm` folder into your Home Assistant `config/custom_components/` directory:
   ```text
   <config_dir>/
   └── custom_components/
       └── homewizard_nilm/
           ├── __init__.py
           ├── button.py
           ├── config_flow.py
           ├── const.py
           ├── coordinator.py
           ├── manifest.json
           ├── ml_engine.py
           ├── select.py
           ├── sensor.py
           ├── services.yaml
           ├── strings.json
           ├── text.py
           └── translations/
               └── en.json
   ```
2. Restart Home Assistant.
3. Go to **Settings → Devices & Services → Add Integration** and search for **HomeWizard NILM Appliance Disaggregator**.
4. Select your HomeWizard Active Power sensor (e.g. `sensor.p1_meter_active_power`), optional 3-phase power sensors (`L1`, `L2`, `L3`), and optional Mobile App notify service (e.g. `notify.mobile_app_your_phone`).

---

## Suggested Lovelace Dashboard Card

Add this card to your Home Assistant dashboard to view pending appliance prompts, select a consumer, mark as Unknown, or add a new consumer:

```yaml
type: vertical-stack
cards:
  - type: entities
    title: ⚡ HomeWizard NILM - Appliance Identification
    entities:
      - entity: sensor.homewizard_nilm_disaggregator_pending_identifications
        name: Pending Prompts
      - entity: sensor.homewizard_nilm_disaggregator_last_detected_event
        name: Last Detected Signature
      - entity: select.homewizard_nilm_disaggregator_identify_consumer
        name: Select Consumer (or Unknown)
      - entity: button.homewizard_nilm_disaggregator_mark_pending_event_as_unknown
        name: Cannot Determine (Mark Unknown)
      - type: section
        label: Add Unlisted Consumer
      - entity: text.homewizard_nilm_disaggregator_new_consumer_name
        name: New Consumer Name
      - entity: button.homewizard_nilm_disaggregator_add_new_consumer_assign
        name: Add New Consumer & Assign Event

  - type: entities
    title: 🔌 Disaggregated Appliance Power & Energy
    entities:
      - sensor.homewizard_nilm_disaggregator_boiler_power
      - sensor.homewizard_nilm_disaggregator_boiler_energy
      - sensor.homewizard_nilm_disaggregator_stove_power
      - sensor.homewizard_nilm_disaggregator_stove_energy
      - sensor.homewizard_nilm_disaggregator_oven_power
      - sensor.homewizard_nilm_disaggregator_oven_energy
      - sensor.homewizard_nilm_disaggregator_washer_power
      - sensor.homewizard_nilm_disaggregator_washer_energy
      - sensor.homewizard_nilm_disaggregator_dryer_power
      - sensor.homewizard_nilm_disaggregator_dryer_energy
      - sensor.homewizard_nilm_disaggregator_dishwasher_power
      - sensor.homewizard_nilm_disaggregator_dishwasher_energy
      - sensor.homewizard_nilm_disaggregator_infrared_heating_power
      - sensor.homewizard_nilm_disaggregator_infrared_heating_energy
      - sensor.homewizard_nilm_disaggregator_floor_heating_bathroom_1_power
      - sensor.homewizard_nilm_disaggregator_floor_heating_bathroom_1_energy
      - sensor.homewizard_nilm_disaggregator_floor_heating_bathroom_2_power
      - sensor.homewizard_nilm_disaggregator_floor_heating_bathroom_2_energy
      - sensor.homewizard_nilm_disaggregator_floor_heating_elsas_bedroom_power
      - sensor.homewizard_nilm_disaggregator_floor_heating_elsas_bedroom_energy
      - sensor.homewizard_nilm_disaggregator_floor_heating_master_bedroom_power
      - sensor.homewizard_nilm_disaggregator_floor_heating_master_bedroom_energy
```
