import socket

class RFLinkClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"[RFLINK] Connected to {self.host}:{self.port}")

    def receive(self, bufsize=4096):
        data = self.sock.recv(bufsize)
        return data

    def close(self):
        if self.sock:
            self.sock.close()
