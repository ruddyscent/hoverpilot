import socket
import unittest
from unittest.mock import Mock, patch

from hoverpilot.rflink.client import RFLinkClient, RFLinkConnectionError


class FailingSocket:
    def __init__(self, exc):
        self.exc = exc
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        raise self.exc

    def close(self):
        self.closed = True


class RFLinkClientTests(unittest.TestCase):
    def test_connect_timeout_raises_clear_connection_error_and_closes_socket(self):
        fake_socket = FailingSocket(socket.timeout("timed out"))
        client = RFLinkClient("10.0.0.2", 18083, socket_timeout_s=0.2)

        with patch("hoverpilot.rflink.client.socket.socket", return_value=fake_socket):
            with self.assertRaises(RFLinkConnectionError) as context:
                client.connect()

        self.assertIn("10.0.0.2:18083", str(context.exception))
        self.assertIn("0.2s", str(context.exception))
        self.assertTrue(fake_socket.closed)
        self.assertIsNone(client.sock)

    def test_close_only_restores_controller_after_successful_injection(self):
        client = RFLinkClient("127.0.0.1", 18083)
        client._restore_original_controller = Mock()

        client.close()

        client._restore_original_controller.assert_not_called()

        client._controller_started = True
        client.close()

        client._restore_original_controller.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
