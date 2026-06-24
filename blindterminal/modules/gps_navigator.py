"""
gps_navigator.py — BlindAssist Project
========================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : GPS Navigation Assistant

Reads GPS coordinates and provides:
  - Current location (reverse geocoded to street address)
  - Walking directions to a destination via OSRM

LAPTOP SIMULATION — uses hardcoded Indore, India coordinates.
When Pi hardware arrives, flip USE_GPIO=True to read UART serial
from NEO-8M GPS module.
"""

import sys
import signal
import logging
import json
import time

from pathlib import Path
from typing import Optional, Dict

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False     # True = read real GPS from NEO-8M via UART

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "gps.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GPSModule")

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Using defaults: {e}")
        return {}

_settings = _load_settings()

# LAPTOP SIMULATION — Hardcoded test coordinates (Indore, India)
GPS_TEST_LAT = _settings.get("gps_test_lat", 22.7196)
GPS_TEST_LON = _settings.get("gps_test_lon", 75.8577)
GPS_BAUD = _settings.get("gps_baud", 9600)
GPS_PORT = _settings.get("gps_port", "/dev/ttyS0")

# Nominatim API (free, no key needed)
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OSRM_URL = "http://router.project-osrm.org"

# ──────────────────────────────────────────────────────────────
# GPS READER
# ──────────────────────────────────────────────────────────────
def _read_gps_hardware() -> Optional[Dict]:
    """Read real GPS coordinates from NEO-8M via UART serial."""
    try:
        import serial
        import pynmea2

        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=5)
        logger.info(f"GPS serial opened on {GPS_PORT} @ {GPS_BAUD} baud")

        # Try to get a valid fix (timeout after 60 seconds)
        start = time.time()
        while time.time() - start < 60:
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPRMC') or line.startswith('$GPGGA'):
                try:
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                        if msg.latitude != 0.0 and msg.longitude != 0.0:
                            logger.info(f"GPS fix: {msg.latitude}, {msg.longitude}")
                            ser.close()
                            return {
                                'lat': msg.latitude,
                                'lon': msg.longitude
                            }
                except pynmea2.ParseError:
                    continue

        ser.close()
        logger.warning("GPS timeout — no satellite fix obtained.")
        return None

    except ImportError:
        logger.error("pyserial/pynmea2 not installed.")
        return None
    except Exception as e:
        logger.error(f"GPS hardware error: {e}")
        return None


def _read_gps_simulation() -> Dict:
    """Return hardcoded test coordinates for laptop simulation."""
    # LAPTOP SIMULATION — replace with GPIO on Pi
    logger.info(
        f"[SIMULATION] Using test coordinates: "
        f"{GPS_TEST_LAT}, {GPS_TEST_LON} (Indore, India)"
    )
    return {
        'lat': GPS_TEST_LAT,
        'lon': GPS_TEST_LON
    }


# ──────────────────────────────────────────────────────────────
# REVERSE GEOCODING (coordinates → address)
# ──────────────────────────────────────────────────────────────
def _reverse_geocode(lat: float, lon: float) -> str:
    """Convert GPS coordinates to a street address using Nominatim."""
    try:
        import requests

        url = f"{NOMINATIM_URL}/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {'User-Agent': 'BlindAssist/1.0'}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        address = data.get('display_name', 'Unknown location')
        logger.info(f"Geocoded: {address}")
        return address

    except ImportError:
        logger.error("requests library not installed.")
        return f"Coordinates: {lat:.4f}, {lon:.4f}"
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        return f"Coordinates: {lat:.4f}, {lon:.4f}"


# ──────────────────────────────────────────────────────────────
# FORWARD GEOCODING (place name → coordinates)
# ──────────────────────────────────────────────────────────────
def _forward_geocode(place_name: str) -> Optional[Dict]:
    """Convert a place name to GPS coordinates using Nominatim."""
    try:
        import requests

        url = f"{NOMINATIM_URL}/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'BlindAssist/1.0'}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()

        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            logger.info(f"Forward geocode '{place_name}' → {lat}, {lon}")
            return {'lat': lat, 'lon': lon}

        logger.warning(f"Place not found: {place_name}")
        return None

    except Exception as e:
        logger.error(f"Forward geocoding failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# ROUTE DIRECTIONS (OSRM)
# ──────────────────────────────────────────────────────────────
def _get_route(from_lat: float, from_lon: float,
               to_lat: float, to_lon: float) -> Optional[str]:
    """Get walking directions using OSRM free routing API."""
    try:
        import requests

        url = (
            f"{OSRM_URL}/route/v1/foot/"
            f"{from_lon},{from_lat};{to_lon},{to_lat}"
        )
        params = {
            'overview': 'false',
            'steps': 'true'
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('code') != 'Ok' or not data.get('routes'):
            return None

        route = data['routes'][0]
        legs = route.get('legs', [])

        # Build spoken directions
        directions = []
        total_distance = route.get('distance', 0)
        total_duration = route.get('duration', 0)

        directions.append(
            f"Total distance is about {int(total_distance)} meters, "
            f"approximately {int(total_duration / 60)} minutes walking."
        )

        for leg in legs:
            for step in leg.get('steps', []):
                maneuver = step.get('maneuver', {})
                modifier = maneuver.get('modifier', '')
                step_type = maneuver.get('type', '')
                distance = int(step.get('distance', 0))
                name = step.get('name', '')

                if step_type == 'depart':
                    direction = f"Start walking"
                    if name:
                        direction += f" on {name}"
                elif step_type == 'arrive':
                    direction = "You have arrived at your destination"
                elif step_type == 'turn':
                    direction = f"Turn {modifier}"
                    if name:
                        direction += f" onto {name}"
                elif step_type == 'continue':
                    direction = f"Continue straight"
                    if name:
                        direction += f" on {name}"
                else:
                    direction = f"Go {modifier}" if modifier else "Continue"
                    if name:
                        direction += f" on {name}"

                if distance > 0 and step_type not in ('arrive',):
                    direction += f" for {distance} meters"

                directions.append(direction + ".")

        result = " ".join(directions)
        logger.info(f"Route: {result[:100]}...")
        return result

    except Exception as e:
        logger.error(f"Routing failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# PUBLIC API — get_location()
# ──────────────────────────────────────────────────────────────
def get_location() -> Dict:
    """
    Get current GPS location with address.

    Returns:
        Dict with 'lat', 'lon', 'address' keys.
    """
    if USE_GPIO:
        coords = _read_gps_hardware()
        if not coords:
            coords = _read_gps_simulation()
    else:
        coords = _read_gps_simulation()

    address = _reverse_geocode(coords['lat'], coords['lon'])
    coords['address'] = address

    return coords


# ──────────────────────────────────────────────────────────────
# PUBLIC API — get_directions()
# ──────────────────────────────────────────────────────────────
def get_directions(destination: str) -> str:
    """
    Get walking directions from current location to a destination.

    Args:
        destination: Place name (e.g., "hospital", "railway station")

    Returns:
        Spoken navigation text.
    """
    if not destination or not destination.strip():
        return "Please tell me where you want to go."

    # Get current position
    current = get_location()
    logger.info(f"Navigation from {current['lat']},{current['lon']} to '{destination}'")

    # Geocode destination
    dest_coords = _forward_geocode(destination)
    if not dest_coords:
        return f"I could not find {destination}. Please try a more specific name."

    # Get route
    route_text = _get_route(
        current['lat'], current['lon'],
        dest_coords['lat'], dest_coords['lon']
    )

    if route_text:
        return route_text
    else:
        return (
            f"I found {destination} but could not calculate a walking route. "
            f"It is located at coordinates {dest_coords['lat']:.4f}, {dest_coords['lon']:.4f}."
        )


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down GPS module.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist GPS Navigator — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("[SIMULATION] Using Indore, India coordinates")
    print("Commands:")
    print("  W  → Where am I? (current location)")
    print("  N  → Navigate to a destination")
    print("  Q  → Quit")
    print("=" * 50 + "\n")

    while True:
        try:
            cmd = input("Command (W/N/Q): ").strip().upper()

            if cmd == 'Q':
                break

            elif cmd == 'W':
                print("\nFinding your location...")
                location = get_location()
                print(f"\n>>> You are at: {location['address']}")
                print(f"    Coordinates: {location['lat']:.4f}, {location['lon']:.4f}\n")

            elif cmd == 'N':
                dest = input("Where do you want to go? ").strip()
                if dest:
                    print(f"\nCalculating route to '{dest}'...")
                    directions = get_directions(dest)
                    print(f"\n>>> {directions}\n")

            else:
                print("Unknown command. Use W, N, or Q.")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting GPS Test...")
            break

    print("GPS Navigator Closed.")
