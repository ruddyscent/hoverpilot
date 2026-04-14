from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping


RF_CHANNEL_COUNT = 12
DEFAULT_CHANNEL_MAP = {
    "aileron": 0,
    "elevator": 1,
    "throttle": 2,
    "rudder": 3,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _validate_finite(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _normalize_bidir(value: float) -> float:
    bounded = _clamp(_validate_finite("bidirectional control", value), -1.0, 1.0)
    return (bounded + 1.0) * 0.5


def _normalize_throttle(value: float) -> float:
    return _clamp(_validate_finite("throttle", value), 0.0, 1.0)


def _normalize_channel_override(index: int, value: float) -> float:
    if not 0 <= index < RF_CHANNEL_COUNT:
        raise ValueError(f"channel index must be in [0, {RF_CHANNEL_COUNT - 1}]")
    return _clamp(_validate_finite(f"channel override {index}", value), 0.0, 1.0)


@dataclass(slots=True)
class RFControlAction:
    """Normalized RC action values for RealFlight Link.

    `aileron`, `elevator`, and `rudder` use [-1.0, 1.0].
    `throttle` uses [0.0, 1.0].
    `channel_overrides` can set any of the 12 outbound ExchangeData channels directly in [0.0, 1.0].
    """

    throttle: float = 0.0
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    channel_overrides: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.throttle = _normalize_throttle(self.throttle)
        self.aileron = _clamp(_validate_finite("aileron", self.aileron), -1.0, 1.0)
        self.elevator = _clamp(_validate_finite("elevator", self.elevator), -1.0, 1.0)
        self.rudder = _clamp(_validate_finite("rudder", self.rudder), -1.0, 1.0)
        self.channel_overrides = {
            index: _normalize_channel_override(index, value)
            for index, value in self.channel_overrides.items()
        }

    @classmethod
    def neutral(cls) -> "RFControlAction":
        return cls(throttle=0.0, aileron=0.0, elevator=0.0, rudder=0.0)

    @classmethod
    def safe_idle(cls) -> "RFControlAction":
        return cls(throttle=0.0, aileron=0.0, elevator=0.0, rudder=0.0)

    def to_channel_values(
        self,
        channel_map: Mapping[str, int] | None = None,
    ) -> list[float]:
        mapping = dict(DEFAULT_CHANNEL_MAP if channel_map is None else channel_map)
        _validate_channel_map(mapping)

        values = [0.0] * RF_CHANNEL_COUNT
        values[mapping["aileron"]] = _normalize_bidir(self.aileron)
        values[mapping["elevator"]] = _normalize_bidir(self.elevator)
        values[mapping["throttle"]] = _normalize_throttle(self.throttle)
        values[mapping["rudder"]] = _normalize_bidir(self.rudder)

        for index, value in self.channel_overrides.items():
            values[index] = value

        return values


def _validate_channel_map(channel_map: Mapping[str, int]) -> None:
    required_controls = set(DEFAULT_CHANNEL_MAP)
    missing = required_controls - set(channel_map)
    if missing:
        raise ValueError(f"channel_map is missing controls: {sorted(missing)}")

    indices = [channel_map[name] for name in required_controls]
    for name, index in channel_map.items():
        if name not in DEFAULT_CHANNEL_MAP:
            raise ValueError(f"unsupported logical control: {name}")
        if not 0 <= index < RF_CHANNEL_COUNT:
            raise ValueError(f"channel index for {name} must be in [0, {RF_CHANNEL_COUNT - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("channel_map must not assign multiple controls to the same channel")


@dataclass(slots=True)
class FlightAxisState:
    rcin: list[float] = field(default_factory=lambda: [0.0] * RF_CHANNEL_COUNT)
    m_airspeed_MPS: float = 0.0
    m_altitudeASL_MTR: float = 0.0
    m_altitudeAGL_MTR: float = 0.0
    m_groundspeed_MPS: float = 0.0
    m_pitchRate_DEGpSEC: float = 0.0
    m_rollRate_DEGpSEC: float = 0.0
    m_yawRate_DEGpSEC: float = 0.0
    m_azimuth_DEG: float = 0.0
    m_inclination_DEG: float = 0.0
    m_roll_DEG: float = 0.0
    m_aircraftPositionX_MTR: float = 0.0
    m_aircraftPositionY_MTR: float = 0.0
    m_velocityWorldU_MPS: float = 0.0
    m_velocityWorldV_MPS: float = 0.0
    m_velocityWorldW_MPS: float = 0.0
    m_velocityBodyU_MPS: float = 0.0
    m_velocityBodyV_MPS: float = 0.0
    m_velocityBodyW_MPS: float = 0.0
    m_accelerationWorldAX_MPS2: float = 0.0
    m_accelerationWorldAY_MPS2: float = 0.0
    m_accelerationWorldAZ_MPS2: float = 0.0
    m_accelerationBodyAX_MPS2: float = 0.0
    m_accelerationBodyAY_MPS2: float = 0.0
    m_accelerationBodyAZ_MPS2: float = 0.0
    m_windX_MPS: float = 0.0
    m_windY_MPS: float = 0.0
    m_windZ_MPS: float = 0.0
    m_propRPM: float = 0.0
    m_heliMainRotorRPM: float = 0.0
    m_batteryVoltage_VOLTS: float = 0.0
    m_batteryCurrentDraw_AMPS: float = 0.0
    m_batteryRemainingCapacity_MAH: float = 0.0
    m_fuelRemaining_OZ: float = 0.0
    m_isLocked: float = 0.0
    m_hasLostComponents: float = 0.0
    m_anEngineIsRunning: float = 0.0
    m_isTouchingGround: float = 0.0
    m_currentAircraftStatus: float = 0.0
    m_currentPhysicsTime_SEC: float = 0.0
    m_currentPhysicsSpeedMultiplier: float = 0.0
    m_orientationQuaternion_X: float = 0.0
    m_orientationQuaternion_Y: float = 0.0
    m_orientationQuaternion_Z: float = 0.0
    m_orientationQuaternion_W: float = 0.0
    m_flightAxisControllerIsActive: float = 0.0
    m_resetButtonHasBeenPressed: float = 0.0

    def summary(self) -> str:
        return (
            "time={time:.3f}s pos=({x:.2f}, {y:.2f})m alt_asl={asl:.2f}m "
            "gs={gs:.2f}m/s air={air:.2f}m/s att=(az={az:.1f}, inc={inc:.1f}, roll={roll:.1f})deg "
            "rates=(p={p:.1f}, r={r:.1f}, y={y_rate:.1f})deg/s "
            "quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
        ).format(
            time=self.m_currentPhysicsTime_SEC,
            x=self.m_aircraftPositionX_MTR,
            y=self.m_aircraftPositionY_MTR,
            asl=self.m_altitudeASL_MTR,
            gs=self.m_groundspeed_MPS,
            air=self.m_airspeed_MPS,
            az=self.m_azimuth_DEG,
            inc=self.m_inclination_DEG,
            roll=self.m_roll_DEG,
            p=self.m_pitchRate_DEGpSEC,
            r=self.m_rollRate_DEGpSEC,
            y_rate=self.m_yawRate_DEGpSEC,
            qx=self.m_orientationQuaternion_X,
            qy=self.m_orientationQuaternion_Y,
            qz=self.m_orientationQuaternion_Z,
            qw=self.m_orientationQuaternion_W,
        )
