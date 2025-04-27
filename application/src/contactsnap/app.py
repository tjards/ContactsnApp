import threading
import time
import toga
import logging

from contactsnap.backend.app import app as flask_app
from contactsnap.frontend.app import ContactSnap

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def start_backend():
    logging.info("Starting backend Flask server...")
    flask_app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)

def main():
    # Start Flask backend in a background thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()

    # Give backend a moment to spin up
    time.sleep(3)

    # Launch the Toga frontend
    return ContactSnap()
