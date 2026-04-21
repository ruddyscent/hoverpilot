import os
import socket
import time
from typing import Mapping, Optional, Tuple

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
        channel_map: Optional[Mapping[str, int]] = None,
        socket_timeout_s: float = 1.0,
        debug_state_flags: Optional[bool] = None,
    ):
        self.host = host
        self.port = port
        self.channel_map = dict(DEFAULT_CHANNEL_MAP if channel_map is None else channel_map)
        self.socket_timeout_s = socket_timeout_s
        self.debug_state_flags = (
            _env_flag_enabled("RFLINK_DEBUG_STATE_FLAGS") if debug_state_flags is None else debug_state_flags
        )
        self.sock = None
        self._buffer = b""
        self._controller_started = False
        self._printed_zero_state_debug = False
        self._last_flag_debug_tuple: Optional[Tuple[float, float, float]] = None

    def connect(self):
        self._open_socket(log=True)
        self._start_controller()

    def request_state(self, action: Optional[RFControlAction] = None) -> FlightAxisState:
        try:
            self._ensure_controller_ready()
            self._send_exchange_request(action)
            response = self._receive_http_response()
        except (ConnectionError, OSError, TimeoutError, socket.timeout):
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
        self._maybe_print_flag_debug(state)
        return state

    def step(self, action: Optional[RFControlAction] = None) -> FlightAxisState:
        return self.request_state(action=action)

    def close(self, restore_controller: bool = True):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None
        self._buffer = b""

        if restore_controller:
            self._restore_original_controller()
        self._controller_started = False

    def _ensure_controller_ready(self):
        if self.sock is None:
            self._open_socket(log=False)
        if not self._controller_started:
            self._start_controller()

    def _open_socket(self, log: bool):
        self.close(restore_controller=False)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.socket_timeout_s)
        self.sock.connect((self.host, self.port))
        if log:
            print(f"[RFLINK] Connected to {self.host}:{self.port}")

    def _reset_connection(self):
        self.close(restore_controller=False)

    def _start_controller(self):
        self._call_simple_action(
            "InjectUAVControllerInterface",
            "<InjectUAVControllerInterface><a>1</a><b>2</b></InjectUAVControllerInterface>",
        )
        if self.sock is None:
            self._open_socket(log=False)
        self._controller_started = True

    def _restore_original_controller(self):
        request_body = (
            "<RestoreOriginalControllerDevice><a>1</a><b>2</b></RestoreOriginalControllerDevice>"
        )
        for attempt in range(1, 4):
            restore_sock = None
            try:
                # Always restore on a fresh short-lived connection. By shutdown time the
                # long-lived ExchangeData socket may already be stale, and using it can
                # leave RealFlight's original InterLink controller un-restored.
                restore_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                restore_sock.settimeout(self.socket_timeout_s)
                restore_sock.connect((self.host, self.port))
                restore_sock.sendall(
                    build_simple_request(
                        self.host,
                        "RestoreOriginalControllerDevice",
                        request_body,
                    )
                )
                _receive_single_http_response(restore_sock)
                print(f"[RFLINK] RestoreOriginalControllerDevice succeeded on attempt {attempt}")
                return
            except (ConnectionError, OSError, TimeoutError, socket.timeout) as exc:
                if attempt == 3:
                    print(f"[RFLINK] RestoreOriginalControllerDevice failed: {exc}")
                else:
                    time.sleep(0.1)
            finally:
                if restore_sock is not None:
                    try:
                        restore_sock.close()
                    except Exception:
                        pass

    def _call_simple_action(self, action: str, body_inner_xml: str):
        if self.sock is None:
            self._open_socket(log=False)
        self.sock.sendall(build_simple_request(self.host, action, body_inner_xml))
        self._receive_http_response(close_after_read=True)

    def _send_exchange_request(self, action: Optional[RFControlAction] = None):
        self._ensure_socket()
        channel_values = None if action is None else action.to_channel_values(self.channel_map)
        self.sock.sendall(build_exchange_data_request(self.host, channel_values=channel_values))

    def _receive_http_response(self, close_after_read: bool = False) -> bytes:
        self._ensure_socket()

        while b"\r\n\r\n" not in self._buffer:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError("timed out while receiving headers") from exc
            if not chunk:
                raise ConnectionError("connection closed while receiving headers")
            self._buffer += chunk

        headers, remainder = self._buffer.split(b"\r\n\r\n", 1)
        content_length = _parse_content_length(headers)

        while len(remainder) < content_length:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError("timed out while receiving body") from exc
            if not chunk:
                raise ConnectionError("connection closed while receiving body")
            remainder += chunk

        body = remainder[:content_length]
        self._buffer = remainder[content_length:]
        if close_after_read:
            self._reset_connection()
        return headers + b"\r\n\r\n" + body


    def _maybe_print_flag_debug(self, state: FlightAxisState):
        if not self.debug_state_flags:
            return

        flag_tuple = (
            float(state.m_hasLostComponents),
            float(state.m_anEngineIsRunning),
            float(state.m_isTouchingGround),
        )
        if flag_tuple == self._last_flag_debug_tuple:
            return

        self._last_flag_debug_tuple = flag_tuple
        print(
            "[RFLINK:flags] "
            f"lost={flag_tuple[0]:0.1f} "
            f"engine={flag_tuple[1]:0.1f} "
            f"ground={flag_tuple[2]:0.1f}"
        )

    def _ensure_socket(self):
        if not self.sock:
            raise RuntimeError("socket is not connected")



def _parse_content_length(headers: bytes) -> int:
    for line in headers.decode("iso-8859-1").split("\r\n"):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("missing Content-Length header")


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _receive_single_http_response(sock: socket.socket) -> bytes:
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        try:
            chunk = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutError("timed out while receiving headers") from exc
        if not chunk:
            raise ConnectionError("connection closed while receiving headers")
        buffer += chunk

    headers, remainder = buffer.split(b"\r\n\r\n", 1)
    content_length = _parse_content_length(headers)

    while len(remainder) < content_length:
        try:
            chunk = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutError("timed out while receiving body") from exc
        if not chunk:
            raise ConnectionError("connection closed while receiving body")
        remainder += chunk

    body = remainder[:content_length]
    return headers + b"\r\n\r\n" + body
