import requests
import time

URL = "http://localhost:8000/telemetry"

def send_telemetry(payload, description=""):
    try:
        response = requests.post(URL, json=payload)
        res = response.json()
        print(f"[{description}]")
        print(f"Risk Score: {res.get('risk_score')} | State: {res.get('status')}")
        print(f"Reasoning:  {res.get('xai_reasoning')}")
        print("-" * 60)
    except Exception as e:
        print(f"Failed to connect. Is the Uvicorn server running? Error: {e}")

print("\n=== STARTING S.T.R.I.D.E 4.0 PENETRATION TESTS ===\n")

# ---------------------------------------------------------
# SCENARIO 1: The "Golden Run" (Normal Human Calibration)
# ---------------------------------------------------------
print("SCENARIO 1: Calibrating Normal Human Operator...")
session_1 = "demo_user_alpha"
base_payload = {
    "session_id": session_1,
    "flight_times": [180.0],
    "hold_times": [90.0],
    "mouse_trajectory": [150.0],
    "error_rates": [1.0],
    "mouse_acceleration": [2.0],
    "context_switch_latency": [300.0],
    "screen_width": 1920,
    "hardware_concurrency": 8,
    "gpu_hash": "NVIDIA_RTX_3080",
    "override_ip": "127.0.0.1" # NY
}

# Send 16 frames to fill the 15-frame buffer and trigger CNN Calibration
for i in range(16):
    send_telemetry(base_payload, f"Normal Operator - Buffer Frame {i+1}/16")
    time.sleep(0.05)

# ---------------------------------------------------------
# SCENARIO 2: The Script Injection (Bot Attack)
# ---------------------------------------------------------
print("\nSCENARIO 2: Injecting Automated Script (0ms unhuman typing)...")
# Modifying the payload to mimic a Python macro/bot hijacking the keyboard
bot_payload = base_payload.copy()
bot_payload["flight_times"] = [1.0] # Physically impossible for humans
bot_payload["hold_times"] = [1.0]
bot_payload["mouse_trajectory"] = [5000.0] # Jerked the mouse across the whole screen instantly

for i in range(3):
    send_telemetry(bot_payload, f"Bot Injection - Frame {i+1}")
    time.sleep(0.05)

# ---------------------------------------------------------
# SCENARIO 3: Session Cookie Hijack (Hardware Mismatch)
# ---------------------------------------------------------
print("\nSCENARIO 3: Session Cookie Stolen & Used on Hacker's Machine...")
session_2 = "demo_user_beta"
hijack_payload = base_payload.copy()
hijack_payload["session_id"] = session_2

# Establish a baseline for the new session
send_telemetry(hijack_payload, "Session Beta - Initial Baseline Payload")

# Uh oh, the GPU hash just instantly changed mid-session!
hijack_payload["gpu_hash"] = "UNKNOWN_AMD_RADEON"
send_telemetry(hijack_payload, "Session Beta - Hardware Invariant Mismatch")

# ---------------------------------------------------------
# SCENARIO 4: Geovelocity Anomaly (Impossible Travel)
# ---------------------------------------------------------
print("\nSCENARIO 4: Impossible Travel Speed (NY to Tokyo in 1 second)...")
session_3 = "demo_user_gamma"
geo_payload = base_payload.copy()
geo_payload["session_id"] = session_3
geo_payload["override_ip"] = "127.0.0.1" # Starts in New York

# Establish baseline geolocation
send_telemetry(geo_payload, "Session Gamma - Baseline Location (NY)")

# The next API ping comes from an IP in Tokyo!
geo_payload["override_ip"] = "4.4.4.4" # Tokyo IP Address
send_telemetry(geo_payload, "Session Gamma - Teleportation to Tokyo")

print("\n=== TESTS COMPLETE ===")
