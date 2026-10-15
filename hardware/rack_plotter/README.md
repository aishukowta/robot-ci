# Lab Rack-and-Pinion Pen Plotter

Standalone Arduino Uno firmware for the 3-axis rack-and-pinion pen plotter used in Robot CI physical experiments (E11).

## Hardware Setup
- **Actuators:** 3 x 28BYJ-48 stepper motors with ULN2003 driver boards.
- **Controller:** Arduino Uno.
- **Wiring:**
  - X axis: D2–D5
  - Y axis: D6–D9
  - PEN axis: D10–D13
  - Driver boards' `+` / `-` connected to 5 V / GND power supply.

## Stored Plans
The plans are pre-compiled directly inside the sketch:
- **Plan 1:** `HELLO` baseline (`4b69978ebb27afd1`)
- **Plan 2:** `HELLO` candidate (`9b7a63446cfded6b`)
- **Plan 3:** `square` baseline (`f9d9e3adaa619f3e`)
- **Plan 4:** `square` candidate (`2ab05ca5e98d11c6`)

## Operation
1. Open the Arduino IDE Serial Monitor at **115200 baud** with line ending set to **Newline**.
2. Send command `R n` (e.g. `R 1`, `R 2`, `R 3`, `R 4`) to draw the corresponding plan.
