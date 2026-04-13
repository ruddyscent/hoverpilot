from dotenv import load_dotenv
import os

load_dotenv()

HOST = os.getenv("RFLINK_HOST", "127.0.0.1")
PORT = int(os.getenv("RFLINK_PORT", 18083))
