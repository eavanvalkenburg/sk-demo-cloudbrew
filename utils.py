import socket
import logging
import os

logger = logging.getLogger(__name__)


def internet(host="1.1.1.1", port=53, timeout=3):
    """
    Host: 1.1.1.1
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    if mode := os.getenv("MODE", None):
        if mode in ("offline", "online"):
            return not (mode == "offline")
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        print("I'm online!")
        return True
    except socket.error:
        print("I'm offline!")
        # print(ex)
        return False
