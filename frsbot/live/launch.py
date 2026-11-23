"""One-command launcher: opens (or reuses) a Chrome window with a debug port, then runs ImAI solo in it.

Usage: python -m frsbot.live.launch [--team 254] [--alliance blue] [bridge options...]
Extra options are passed straight to frsbot.live.bridge.
"""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PORT = 9222
CDP = f"http://127.0.0.1:{PORT}"
PROFILE = Path.home() / ".frsbot" / "chrome-profile"
SITE = "https://rebuilt.frsim.com/"


def chrome_ready() -> bool:
    try:
        urllib.request.urlopen(f"{CDP}/json/version", timeout=1)
        return True
    except OSError:
        return False


def start_chrome():
    if chrome_ready():
        print("reusing Chrome already on port", PORT)
        return
    if not CHROME.exists():
        sys.exit(f"Google Chrome not found at {CHROME}")
    subprocess.Popen([str(CHROME), f"--remote-debugging-port={PORT}", f"--user-data-dir={PROFILE}",
                      "--no-first-run", "--no-default-browser-check", SITE],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if chrome_ready():
            print("started Chrome on port", PORT)
            return
        time.sleep(0.5)
    sys.exit("Chrome did not open its debug port")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, default=254)
    parser.add_argument("--alliance", choices=("blue", "red"), default="blue")
    args, bridge_args = parser.parse_known_args()
    start_chrome()
    command = [sys.executable, "-m", "frsbot.live.bridge", "--team", str(args.team), "--alliance", args.alliance,
               "--cdp", CDP, "--driver", "python", "--latency", "auto", "--calibrate", "6", *bridge_args]
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
