import socket
from typing import Mapping

from hoverpilot.rflink.models import DEFAULT_CHANNEL_MAP, FlightAxisState, RFControlAction
from hoverpilot.rflink.protocol import (
    build_exchange_data_request,
    build_simple_request,
    parse_http_body,
    parse_state,
    state_looks_uninitialized,
)


class RFLinkClient:
    def __init__(
        self,
        host: str,
        port: int,
        channel_map: Mapping[str, int] | None = None,
    ):
        self.host = host
        self.port = port
        self.channel_map = dict(DEFAULT_CHANNEL_MAP if channel_map is None else channel_map)
        self.sock = None
        self._buffer = b""
        self._controller_started = False
        self._printed_zero_state_debug = False

    def connect(self):
        self._open_socket(log=True)
        self._start_controller()

    def request_state(self, action: RFControlAction | None = None) -> FlightAxisState:
        try:
            self._ensure_controller_ready()
            self._send_exchange_request(action)
            response = self._receive_http_response()
        except (ConnectionError, OSError):
            self._reset_connection()
            self._ensure_controller_ready()
            self._send_exchange_request(action)
            response = self._receive_http_response()

        body = parse_http_body(response)
        state = parse_state(body)
        if state_looks_uninitialized(state) and not self._printed_zero_state_debug:
            self._printed_zero_state_debug = True
            print("[RFLINK] Received zeroed state. First SOAP body follows:")
            print(body[:2000])
        return state

    def step(self, action: RFControlAction | None = None) -> FlightAxisState:
        return self.request_state(action=action)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        self._buffer = b""

    def _ensure_controller_ready(self):
        if self.sock is None:
            self._open_socket(log=True)
        if not self._controller_started:
            self._start_controller()

    def _open_socket(self, log: bool):
        self.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        if log:
            print(f"[RFLINK] Connected to {self.host}:{self.port}")

    def _reset_connection(self):
        self.close()

    def _start_controller(self):
        self._call_simple_action(
            "RestoreOriginalControllerDevice",
            "<RestoreOriginalControllerDevice><a>1</a><b>2</b></RestoreOriginalControllerDevice>",
        )
        self._call_simple_action(
            "InjectUAVControllerInterface",
            "<InjectUAVControllerInterface><a>1</a><b>2</b></InjectUAVControllerInterface>",
        )
        self._controller_started = True

    def _call_simple_action(self, action: str, body_inner_xml: str):
        self._ensure_socket()
        self.sock.sendall(build_simple_request(self.host, action, body_inner_xml))
        self._receive_http_response(close_after_read=True)
        self._open_socket(log=False)

    def _send_exchange_request(self, action: RFControlAction | None = None):
        self._ensure_socket()
        channel_values = None if action is None else action.to_channel_values(self.channel_map)
        self.sock.sendall(build_exchange_data_request(self.host, channel_values=channel_values))

    def _receive_http_response(self, close_after_read: bool = True) -> bytes:
        self._ensure_socket()

        while b"\r\n\r\n" not in self._buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed while receiving headers")
            self._buffer += chunk

        headers, remainder = self._buffer.split(b"\r\n\r\n", 1)
        content_length = _parse_content_length(headers)

        while len(remainder) < content_length:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed while receiving body")
            remainder += chunk

        body = remainder[:content_length]
        self._buffer = remainder[content_length:]
        if close_after_read:
            self._reset_connection()
        return headers + b"\r\n\r\n" + body

    def _ensure_socket(self):
        if not self.sock:
            raise RuntimeError("socket is not connected")



def _parse_content_length(headers: bytes) -> int:
    for line in headers.decode("iso-8859-1").split("\r\n"):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("missing Content-Length header")
