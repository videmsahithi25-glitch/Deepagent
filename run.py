import subprocess
import sys
import os
import time

def main():
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "imatb.py"])
    try:
        while proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Force killing Streamlit and all child threads/processes...]")
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
        else:
            proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()