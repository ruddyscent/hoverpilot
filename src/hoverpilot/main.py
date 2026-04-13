from hoverpilot.config import HOST, PORT
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.rflink.protocol import debug_print

def main():
    client = RFLinkClient(HOST, PORT)
    client.connect()

    try:
        while True:
            data = client.receive()
            if not data:
                break
            debug_print(data)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.close()

if __name__ == "__main__":
    main()
