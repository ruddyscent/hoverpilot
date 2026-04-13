from dataclasses import dataclass, field


@dataclass(slots=True)
class FlightAxisState:
    rcin: list[float] = field(default_factory=lambda: [0.0] * 12)
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
