import time

from hoverpilot.config import HOST, PORT
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.rflink.models import RFControlAction


def main():
    client = RFLinkClient(HOST, PORT)
    client.connect()
    # Smoke-test action only: light throttle with centered control surfaces.
    # Edit these values while validating outbound RC control delivery.
    demo_action = RFControlAction(
        throttle=0.55,
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
    )

    try:
        while True:
            state = client.step(demo_action)
            print(state.summary())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.close()


if __name__ == "__main__":
    main()
