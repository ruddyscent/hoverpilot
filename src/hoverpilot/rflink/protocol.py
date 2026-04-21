from xml.etree import ElementTree as ET
from typing import List, Optional

from hoverpilot.rflink.models import FlightAxisState, RF_CHANNEL_COUNT


SOAP_ENVELOPE_PREFIX = """<?xml version='1.0' encoding='UTF-8'?>
<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' xmlns:xsd='http://www.w3.org/2001/XMLSchema' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <soap:Body>
"""

SOAP_ENVELOPE_SUFFIX = """
  </soap:Body>
</soap:Envelope>"""

STATE_FIELD_MAP = {
    "m-airspeed-MPS": "m_airspeed_MPS",
    "m-altitudeASL-MTR": "m_altitudeASL_MTR",
    "m-altitudeAGL-MTR": "m_altitudeAGL_MTR",
    "m-groundspeed-MPS": "m_groundspeed_MPS",
    "m-pitchRate-DEGpSEC": "m_pitchRate_DEGpSEC",
    "m-rollRate-DEGpSEC": "m_rollRate_DEGpSEC",
    "m-yawRate-DEGpSEC": "m_yawRate_DEGpSEC",
    "m-azimuth-DEG": "m_azimuth_DEG",
    "m-inclination-DEG": "m_inclination_DEG",
    "m-roll-DEG": "m_roll_DEG",
    "m-aircraftPositionX-MTR": "m_aircraftPositionX_MTR",
    "m-aircraftPositionY-MTR": "m_aircraftPositionY_MTR",
    "m-velocityWorldU-MPS": "m_velocityWorldU_MPS",
    "m-velocityWorldV-MPS": "m_velocityWorldV_MPS",
    "m-velocityWorldW-MPS": "m_velocityWorldW_MPS",
    "m-velocityBodyU-MPS": "m_velocityBodyU_MPS",
    "m-velocityBodyV-MPS": "m_velocityBodyV_MPS",
    "m-velocityBodyW-MPS": "m_velocityBodyW_MPS",
    "m-accelerationWorldAX-MPS2": "m_accelerationWorldAX_MPS2",
    "m-accelerationWorldAY-MPS2": "m_accelerationWorldAY_MPS2",
    "m-accelerationWorldAZ-MPS2": "m_accelerationWorldAZ_MPS2",
    "m-accelerationBodyAX-MPS2": "m_accelerationBodyAX_MPS2",
    "m-accelerationBodyAY-MPS2": "m_accelerationBodyAY_MPS2",
    "m-accelerationBodyAZ-MPS2": "m_accelerationBodyAZ_MPS2",
    "m-windX-MPS": "m_windX_MPS",
    "m-windY-MPS": "m_windY_MPS",
    "m-windZ-MPS": "m_windZ_MPS",
    "m-propRPM": "m_propRPM",
    "m-heliMainRotorRPM": "m_heliMainRotorRPM",
    "m-batteryVoltage-VOLTS": "m_batteryVoltage_VOLTS",
    "m-batteryCurrentDraw-AMPS": "m_batteryCurrentDraw_AMPS",
    "m-batteryRemainingCapacity-MAH": "m_batteryRemainingCapacity_MAH",
    "m-fuelRemaining-OZ": "m_fuelRemaining_OZ",
    "m-isLocked": "m_isLocked",
    "m-hasLostComponents": "m_hasLostComponents",
    "m-anEngineIsRunning": "m_anEngineIsRunning",
    "m-isTouchingGround": "m_isTouchingGround",
    "m-currentAircraftStatus": "m_currentAircraftStatus",
    "m-currentPhysicsTime-SEC": "m_currentPhysicsTime_SEC",
    "m-currentPhysicsSpeedMultiplier": "m_currentPhysicsSpeedMultiplier",
    "m-orientationQuaternion-X": "m_orientationQuaternion_X",
    "m-orientationQuaternion-Y": "m_orientationQuaternion_Y",
    "m-orientationQuaternion-Z": "m_orientationQuaternion_Z",
    "m-orientationQuaternion-W": "m_orientationQuaternion_W",
    "m-flightAxisControllerIsActive": "m_flightAxisControllerIsActive",
    "m-resetButtonHasBeenPressed": "m_resetButtonHasBeenPressed",
}


STATE_ZERO_FIELDS = (
    "m_currentPhysicsTime_SEC",
    "m_altitudeASL_MTR",
    "m_groundspeed_MPS",
    "m_airspeed_MPS",
    "m_orientationQuaternion_W",
    "m_flightAxisControllerIsActive",
)



def build_simple_request(host: str, action: str, body_inner_xml: str) -> bytes:
    body = f"{SOAP_ENVELOPE_PREFIX}{body_inner_xml}{SOAP_ENVELOPE_SUFFIX}"
    return build_soap_request(host=host, action=action, body=body)



def build_exchange_data_request(
    host: str,
    channel_values: Optional[List[float]] = None,
    selected_channels: int = 4095,
) -> bytes:
    values = channel_values or [0.0] * RF_CHANNEL_COUNT
    if len(values) != RF_CHANNEL_COUNT:
        raise ValueError(f"channel_values must contain {RF_CHANNEL_COUNT} items")

    channel_items = "\n".join(f"          <item>{value:.4f}</item>" for value in values)
    body = f"""<?xml version='1.0' encoding='UTF-8'?>
<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' xmlns:xsd='http://www.w3.org/2001/XMLSchema' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <soap:Body>
    <ExchangeData>
      <pControlInputs>
        <m-selectedChannels>{selected_channels}</m-selectedChannels>
        <m-channelValues-0to1>
{channel_items}
        </m-channelValues-0to1>
      </pControlInputs>
    </ExchangeData>
  </soap:Body>
</soap:Envelope>"""
    return build_soap_request(host=host, action="ExchangeData", body=body)



def build_soap_request(host: str, action: str, body: str) -> bytes:
    payload = body.encode("utf-8")
    headers = [
        b"POST / HTTP/1.1",
        f"Host: {host}".encode("utf-8"),
        f"soapaction: '{action}'".encode("utf-8"),
        f"content-length: {len(payload)}".encode("ascii"),
        b"content-type: text/xml;charset='UTF-8'",
        b"Connection: Keep-Alive",
        b"",
        b"",
    ]
    return b"\r\n".join(headers) + payload



def parse_http_body(data: bytes) -> str:
    marker = b"\r\n\r\n"
    if marker not in data:
        raise ValueError("invalid HTTP response: missing header separator")
    return data.split(marker, 1)[1].decode("utf-8", errors="replace")



def parse_state(xml_text: str) -> FlightAxisState:
    root = ET.fromstring(xml_text)
    state = FlightAxisState()
    item_values: List[float] = []

    for element in root.iter():
        tag = _strip_namespace(element.tag)
        text = (element.text or "").strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue

        if tag == "item":
            item_values.append(value)
            continue

        field_name = STATE_FIELD_MAP.get(tag)
        if field_name is not None:
            setattr(state, field_name, value)

    if item_values:
        state.rcin = (item_values + [0.0] * RF_CHANNEL_COUNT)[:RF_CHANNEL_COUNT]

    return state



def state_looks_uninitialized(state: FlightAxisState) -> bool:
    return all(abs(getattr(state, field_name)) < 1.0e-9 for field_name in STATE_ZERO_FIELDS)



def debug_print(data: bytes):
    print(f"[RAW] {len(data)} bytes")
    print(data[:64])



def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]
