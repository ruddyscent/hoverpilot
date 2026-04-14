# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python client to connect to RealFlight Link (TCP 18083), exchange RC commands, and expose a Gymnasium-compatible hover environment.

## Quickstart

```bash
pip install -e .
cp .env.example .env
python -m hoverpilot.main
```

## Gymnasium Environment

The project now exposes a Gymnasium-style environment:

```python
import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv

env = HoverPilotHoverEnv(
    host=HOST,
    port=PORT,
    max_episode_steps=250,
)
observation, info = env.reset()
action = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
```

The API mirrors Gymnasium:

- `reset(...) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`

Action format is a 4-element `float32` array:

- index 0: `aileron` in `[-1, 1]`
- index 1: `elevator` in `[-1, 1]`
- index 2: `throttle` in `[0, 1]`
- index 3: `rudder` in `[-1, 1]`

Observation format is a compact 12-element `float32` vector for hover training:

- position: `x`, `y`, `altitude_agl`
- attitude: `roll`, `inclination`, `azimuth`
- world velocity: `u`, `v`, `w`
- angular rates: `pitch_rate`, `roll_rate`, `yaw_rate`

Reward and termination are integrated from `hoverpilot.training.hover`:

- reward prefers staying near the target hover point and upright attitude
- boundary proximity adds a growing penalty before failure
- terminal failures include trainer boundary exit, altitude bounds, lost components, locked vehicle states, configured controller inactivity, configured engine stop, and post-start ground contact

`reset()` waits for a usable start state before returning. During this warmup the environment keeps sending a safe idle action and polls RealFlight until readiness is satisfied or a timeout is reached.

## Demo

Run:

```bash
python -m hoverpilot.main
```

The demo prints:

- observation shape
- scalar reward
- `terminated` / `truncated`
- termination reason when present
- current AGL altitude from `info["debug_state"]`
- a concise RealFlight state summary

The demo keeps running across episodes and only stops on `KeyboardInterrupt`. During reset wait periods it rate-limits the `waiting for trainer reset` log to avoid flooding the terminal.

## Episode Lifecycle

`Airplane Hover Trainer` does not always expose a clean explicit reset flag through RealFlight Link, so the environment manages episode lifecycle conservatively.

Episode start:

- `reset()` and reset-wait polling both use a safe idle action.
- A state is considered ready when it is not obviously uninitialized, not locked, and not already failed.
- Controller-active and engine-running checks are available as configurable readiness gates because these fields can behave differently across trainer modes.
- Ground contact is allowed during startup by default because some trainer resets spawn on or very near the ground.

Episode end:

- Hard terminal failures include:
  - `m_hasLostComponents > threshold`
  - boundary exit in `x` or `y`
  - altitude too low / too high
  - locked vehicle state
  - configured controller inactive / engine stopped conditions
  - post-start ground contact after the configured grace period
- `m_currentAircraftStatus` is currently treated as opaque. It is exposed in `info["debug_state"]`, and only becomes terminal if you explicitly configure known terminal status codes.

Reset-wait and restart:

- After termination, the environment keeps polling with a safe idle action until a new episode can be started.
- Restart signals are checked in this order:
  - reset button pressed
  - physics time rollback
  - lost-components recovery after a crash
  - observed crash signature recovery:
    `m_hasLostComponents > 0`, `m_anEngineIsRunning == 0`, `m_isTouchingGround == 1`
    followed by
    `m_hasLostComponents == 0`, `m_anEngineIsRunning == 1`, `m_isTouchingGround == 0`
  - controller or engine reactivation
  - trainer-driven reposition / teleport back near the hover target while the vehicle is nearly stationary

Useful tuning parameters on `HoverPilotHoverEnv`:

- readiness / warmup:
  - `max_reset_wait_seconds`
  - `reset_poll_interval_seconds`
  - `ready_controller_active_threshold`
  - `ready_running_threshold`
  - `ready_locked_threshold`
  - `allow_ground_contact_at_ready`
- teleport fallback:
  - `reposition_position_margin_ratio`
  - `reposition_altitude_margin_ratio`
  - `reposition_speed_threshold_mps`
  - `reset_teleport_distance_m`
- termination thresholds via `RewardConfig`:
  - `controller_active_threshold`
  - `terminate_on_engine_stopped`
  - `ground_contact_grace_seconds`
  - `known_terminal_aircraft_status_codes`

The environment prefers explicit RealFlight Link state flags first. Teleport / reposition detection is kept as a fallback because the Hover Trainer can reset by suddenly moving the aircraft back near the hover target without updating the more semantic lifecycle flags.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
