# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python client to connect to RealFlight Link (TCP 18083) and inspect raw data.

## Quickstart

```bash
pip install -e .
cp .env.example .env
python -m hoverpilot.main
```

## Sending Basic Actions

`hoverpilot.main` sends a simple fixed control-signal smoke test on each `ExchangeData` request while still reading simulator state back. It keeps the RealFlight Link controller injected once at startup and then reuses it across polling requests:

```python
from hoverpilot.rflink.models import RFControlAction

action = RFControlAction(
    throttle=0.55,
    aileron=0.0,
    elevator=0.0,
    rudder=0.0,
)
```

Logical controls use normalized values:

- `aileron`, `elevator`, `rudder`: `[-1.0, 1.0]`
- `throttle`: `[0.0, 1.0]`

They are mapped to the default 12-channel RealFlight payload as:

- channel 0: aileron
- channel 1: elevator
- channel 2: throttle
- channel 3: rudder

All remaining channels default to `0.0` unless explicitly overridden.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
