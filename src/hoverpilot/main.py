from hoverpilot.config import HOST, PORT
from hoverpilot.rflink.client import RFLinkClient


def main():
    client = RFLinkClient(HOST, PORT)
    client.connect()

    try:
        while True:
            state = client.request_state()
            print(state.summary())
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.close()


if __name__ == "__main__":
    main()
