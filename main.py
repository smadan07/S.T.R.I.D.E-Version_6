from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import math
import numpy as np
import datetime
import tensorflow as tf
import requests

IP_CACHE = {}

# --- S.T.R.I.D.E. 4.0: Dense MLP Autoencoder ---
# Architecture: Zero-Trust Continuous Authentication Platform
# Powered by a 1D Convolutional Neural Network processing time-series tensors.

app = FastAPI(title="S.T.R.I.D.E. 4.0 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables loaded via try/except loop
try:
    print("Loading Dense Autoencoder and stride_config.pkl...")
    GLOBAL_MODEL = tf.keras.models.load_model("stride_cnn.keras")
    
    config = joblib.load("stride_config.pkl")
    GLOBAL_SCALER = config['scaler']
    ANOMALY_THRESHOLD = config['anomaly_threshold']
    print(f"Deep Engine Online. Anomaly Limit: {ANOMALY_THRESHOLD:.4f}")
except Exception as e:
    print(f"CRITICAL WARNING: Base model failed to load. Ensure train.py has run. Error: {e}")
    GLOBAL_MODEL = None
    GLOBAL_SCALER = None
    ANOMALY_THRESHOLD = 999.0

# Global sessions database
sessions_db = {}

# --- DTOs ---
from typing import Optional

class TelemetryData(BaseModel):
    session_id: str = "default_session"
    flight_times: Optional[list[float]] = []
    hold_times: Optional[list[float]] = []
    mouse_trajectory: Optional[list[float]] = []
    error_rates: Optional[list[float]] = []
    mouse_acceleration: Optional[list[float]] = []
    context_switch_latency: Optional[list[float]] = []
    screen_width: Optional[int] = 1920
    hardware_concurrency: Optional[int] = 2
    gpu_hash: Optional[str] = "unknown"

class Invariants(BaseModel):
    screen_width: Optional[int] = 1920
    hardware_concurrency: Optional[int] = 2
    gpu_hash: Optional[str] = "unknown"

    def __eq__(self, other):
        return (self.screen_width == other.screen_width and
                self.hardware_concurrency == other.hardware_concurrency and
                self.gpu_hash == other.gpu_hash)

# --- UTILS ---
def get_mock_coordinates(ip: str):
    import hashlib
    h = hashlib.md5(ip.encode()).hexdigest()
    lat = -90 + (int(h[:8], 16) / 0xffffffff) * 180
    lon = -180 + (int(h[8:16], 16) / 0xffffffff) * 360
    if ip == "127.0.0.1" or ip.startswith("10.") or ip.startswith("192.168."): return (40.7128, -74.0060) # NY
    if ip == "8.8.8.8": return (51.5074, -0.1278) # London (VPN Scenario)
    if ip == "4.4.4.4": return (35.6762, 139.6503) # Tokyo (Teleport Scenario)
    return (lat, lon)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def is_vpn(ip: str):
    return ip == "8.8.8.8"

# 1. The Setup & Cache
def get_ip_reputation(ip_address: str) -> dict:
    # Whitelist localhost to prevent pointless loopback lookups
    if ip_address in ("127.0.0.1", "localhost", "0.0.0.0") or ip_address.startswith("192.") or ip_address.startswith("10."):
        return {"isp": "Local Network", "proxy": False, "hosting": False}
        
    # The In-Memory Cache (0ms latency)
    if ip_address in IP_CACHE:
        return IP_CACHE[ip_address]
        
    try:
        url = f"http://ip-api.com/json/{ip_address}?fields=isp,proxy,hosting"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            context = {
                "isp": data.get("isp", "Unknown ISP"),
                "proxy": data.get("proxy", False),
                "hosting": data.get("hosting", False)
            }
            IP_CACHE[ip_address] = context
            return context
    except Exception:
        # Fails open: If the API crashes during the hackathon demo, we simply ignore it rather than killing the server.
        pass
        
    return {"isp": "Unknown", "proxy": False, "hosting": False}

# --- CORE ENDPOINTS ---

@app.get("/")
async def serve_demo_ui():
    return FileResponse("index.html")

@app.get("/sensor.js")
async def serve_stealth_agent():
    return FileResponse("sensor.js")

@app.post("/telemetry")
async def process_telemetry(payload: TelemetryData, request: Request):
    if GLOBAL_MODEL is None or GLOBAL_SCALER is None:
        return {"error": "Dense Autoencoder offline - run train.py"}

    sid = payload.session_id
    current_ip = request.client.host
    now = datetime.datetime.now()
    
    current_invariants = Invariants(
        screen_width=payload.screen_width,
        hardware_concurrency=payload.hardware_concurrency,
        gpu_hash=payload.gpu_hash
    )
    
    # 1. State Management - Initialization
    if sid not in sessions_db:
        # Clone the Dense Autoencoder for per-session personalization (Zero-Trust isolation)
        personal_model = tf.keras.models.clone_model(GLOBAL_MODEL)
        personal_model.set_weights(GLOBAL_MODEL.get_weights())
        # Re-compile to allow online fine-tuning
        personal_model.compile(optimizer='adam', loss='mse')

        sessions_db[sid] = {
            "state": "active",
            "personal_model": personal_model,
            "last_ip": current_ip,
            "last_timestamp": now,
            "calibration_samples": 0,
            "personal_threshold": ANOMALY_THRESHOLD,
            "baseline_invariants": current_invariants,
            "risk_score": 0.0,
            "xai_reasoning": "Dense Autoencoder initialized. Calibrating to your behavioral baseline..."
        }
        
    sess = sessions_db[sid]
    
    # The Sandbox Latch Fast Reject
    if sess["state"] == "sandboxed":
        sess["risk_score"] = 100.0
        return {"status": "sandboxed", "risk_score": 100.0, "xai_reasoning": "Session permanently locked by Neural Net."}
        
    # Feature Engineering — Rolling Window Statistics
    # The rolling arrays from sensor.js (last 30 keystrokes) allow us to compute
    # both mean AND standard deviation, giving a richer behavioral fingerprint.
    h     = np.mean(payload.hold_times)   if payload.hold_times   else 0.0
    std_h = np.std(payload.hold_times)    if len(payload.hold_times) > 1   else 3.0
    f     = np.mean(payload.flight_times) if payload.flight_times else 0.0
    std_f = np.std(payload.flight_times)  if len(payload.flight_times) > 1 else 5.0
    m     = sum(payload.mouse_trajectory) if payload.mouse_trajectory else 0.0
    er    = np.mean(payload.error_rates)  if payload.error_rates  else 0.0
    
    # 2. Structural Invariant Verification (WebGL + Hardware)
    if current_invariants != sess["baseline_invariants"]:
        # The user's GPU hardware signature or screen dimensions suddenly changed mid-session.
        # This is mathematically impossible unless a hacker stole the session cookie on another machine!
        sess["risk_score"] += 100.0
        sess["xai_reasoning"] = "CRITICAL: WebGL GPU Hardware Fingerprint Mismatch. Confirmed Session Hijack."
        sess["baseline_invariants"] = current_invariants # Prevent infinitely spamming the score
        
    # 3. Geovelocity Engine & Location Anomalies
    time_diff_hours = (now - sess["last_timestamp"]).total_seconds() / 3600.0
    if time_diff_hours > 0 and current_ip != sess["last_ip"]:
        
        # Datacenter Trap / ASN Reputation Matrix
        ip_context = get_ip_reputation(current_ip)
        if ip_context["hosting"] or ip_context["proxy"]:
            sess["risk_score"] += 100.0
            sess["xai_reasoning"] = f"CRITICAL ASN Anomaly: Traffic routed through Commercial Datacenter/Proxy ({ip_context['isp']})."
        else:
            lat1, lon1 = get_mock_coordinates(sess["last_ip"])
            lat2, lon2 = get_mock_coordinates(current_ip)
            speed = haversine(lat1, lon1, lat2, lon2) / time_diff_hours
            
            if speed > 1000:
                vpn_flag = is_vpn(current_ip)
                if not vpn_flag:
                    sess["risk_score"] += 100.0
                    sess["xai_reasoning"] = f"CRITICAL Geovelocity anomaly. Impossible Travel Speed: {speed:.0f}km/h."
                
    sess["last_ip"] = current_ip
    sess["last_timestamp"] = now
    
    # 3. Dense Autoencoder Inference (instant — no sequence queue needed)
    # Feature vector: [mean_hold, std_hold, mean_flight, std_flight, mouse_dist, error_rate]
    # std_hold and std_flight capture RHYTHM CONSISTENCY — a key biometric signal.
    if h > 0.0 or f > 0.0 or er > 0.0:
        if m == 0.0: m = 380.0  # Allow pure typing without artificial mouse penalties

        feature_vector = np.array([[h, std_h, f, std_f, m, er]])
        scaled_vector  = GLOBAL_SCALER.transform(feature_vector)  # shape (1, 6)

        # Autoencoder Reconstruction:
        # If the rhythm is unrecognized, it will fail to rebuild the output vector.
        reconstructed = sess["personal_model"].predict(scaled_vector, verbose=0)

        # Reconstruction Error (MSE) — the anomaly signal
        mse = np.mean(np.square(scaled_vector - reconstructed))

        is_calibrating = sess.get("calibration_samples", 0) < 5

        if is_calibrating:
            # Fine-tune the model weights to this specific user's rhythm.
            # epochs=2 avoids overfitting that would cause it to reject natural variation.
            sess["personal_model"].fit(scaled_vector, scaled_vector, epochs=2, verbose=0)
            sess["calibration_samples"] += 1

            # Expand threshold elasticity to the Goldilocks Zone (1.5x max calibration noise).
            if mse > sess["personal_threshold"]:
                sess["personal_threshold"] = max(sess["personal_threshold"], mse * 1.5)

            sess["risk_score"] = 0.0
            sess["xai_reasoning"] = f"Calibrating behavioral baseline... ({5 - sess['calibration_samples']} cycles remain)"
        else:
            pt = sess["personal_threshold"]
            if mse > pt:
                # Anomaly: proportional penalty capped at 15/tick for smooth UI escalation
                drift   = abs(mse - pt)
                penalty = min(15.0, (drift * 10.0) + 5.0)
                sess["risk_score"] += penalty
                sess["xai_reasoning"] = f"Behavioral Drift Detected. MSE {mse:.4f} > {pt:.4f}. (+{penalty:.1f} Risk)"
            else:
                # Clean rhythm — recover points and optionally fine-tune
                sess["risk_score"] = max(0.0, sess["risk_score"] - 6.0)
                if mse < (pt * 0.5):
                    # Safe zone: online learning to adapt to natural drift over time
                    sess["personal_model"].fit(scaled_vector, scaled_vector, epochs=1, verbose=0)
                    if sess["risk_score"] < 5.0:
                        sess["xai_reasoning"] = f"Rhythm verified. Continuous auth active. (MSE: {mse:.4f})"
                    else:
                        sess["xai_reasoning"] = f"Recovering. Neural lock active against poisoning. (MSE: {mse:.4f})"
                else:
                    sess["xai_reasoning"] = f"Borderline rhythm. Neural lock active. (MSE: {mse:.4f})"
            
    # 4. The Sandbox Latch
    risk = sess["risk_score"]
    
    if risk < 36:
        sess["state"] = "seamless"
    elif risk < 70:
        sess["state"] = "mfa_challenge"
    else:
        sess["state"] = "sandboxed"
        sess["risk_score"] = 100.0 
        
    return {
        "status": sess["state"],
        "risk_score": sess["risk_score"],
        "xai_reasoning": sess["xai_reasoning"]
    }

@app.get("/risk-status/{session_id}")
async def fetch_risk_status(session_id: str):
    if session_id not in sessions_db:
        return {"error": "Session Not Found", "status": "unknown"}
    sess = sessions_db[session_id]
    return {
        "status": sess["state"],
        "risk_score": sess["risk_score"],
        "xai_reasoning": sess["xai_reasoning"]
    }

# --- HACKATHON DEMO ATTACK SIMULATORS ---
# These endpoints let you trigger specific attack scenarios on demand
# without needing a second machine or VPN. Pure controlled demonstration.

@app.get("/demo/hardware-hijack/{session_id}")
async def demo_hardware_hijack(session_id: str):
    """
    Simulates a session hijack where an attacker stole the session cookie
    and is now connecting from a DIFFERENT physical machine with a different GPU.
    Corrupts the baseline GPU hash so the next telemetry tick detects a mismatch.
    """
    if session_id not in sessions_db:
        return {"error": "Session not found. Make sure you're typing in the browser first."}
    
    sess = sessions_db[session_id]
    original_hash = sess["baseline_invariants"].gpu_hash
    
    # Plant a fake GPU hash as the new "attacker's machine" baseline.
    # The next real telemetry tick from the real browser will send the REAL hash,
    # which will no longer match — triggering the hardware invariant alarm.
    sess["baseline_invariants"] = Invariants(
        screen_width=2560,            # attacker has a different resolution
        hardware_concurrency=32,      # attacker has a different CPU
        gpu_hash="gpu_attacker_machine_f4k3d"  # attacker's GPU hash
    )
    
    return {
        "attack": "Hardware Invariant Corruption",
        "original_gpu": original_hash,
        "injected_gpu": "gpu_attacker_machine_f4k3d",
        "message": "Baseline corrupted. Next telemetry tick will detect hardware mismatch and fire +100 risk.",
        "what_happens_next": "CRITICAL: WebGL GPU Hardware Fingerprint Mismatch. Confirmed Session Hijack."
    }

@app.get("/demo/geovelocity/{session_id}")
async def demo_geovelocity(session_id: str):
    """
    Simulates impossible travel: the user was in New York (127.0.0.1 → NY),
    and their IP suddenly jumps to Tokyo (4.4.4.4). 
    Travel distance: ~10,800 km. Time: <1 second. Speed: millions of km/h.
    """
    if session_id not in sessions_db:
        return {"error": "Session not found. Make sure you're typing in the browser first."}
    
    sess = sessions_db[session_id]
    original_ip = sess["last_ip"]
    
    # Plant the "previous location" as Tokyo.
    # The real browser IP (127.0.0.1 → mapped to New York) will be compared
    # against Tokyo on the next telemetry tick, yielding impossible travel speed.
    sess["last_ip"] = "4.4.4.4"  # 4.4.4.4 → mapped to Tokyo in get_mock_coordinates()
    sess["last_timestamp"] = sess["last_timestamp"] - __import__("datetime").timedelta(seconds=10)
    
    lat1, lon1 = get_mock_coordinates("4.4.4.4")     # Tokyo
    lat2, lon2 = get_mock_coordinates("127.0.0.1")   # New York
    distance_km = haversine(lat1, lon1, lat2, lon2)
    
    return {
        "attack": "Geovelocity Teleportation",
        "previous_location": "Tokyo, JP  (4.4.4.4)",
        "current_location":  "New York, US  (127.0.0.1)",
        "distance_km": round(distance_km, 0),
        "time_allowed_seconds": 10,
        "implied_speed_kmh": round(distance_km / (10 / 3600), 0),
        "message": "Location poisoned. Next telemetry tick will calculate impossible travel speed and fire +100 risk.",
        "what_happens_next": "CRITICAL Geovelocity anomaly. Impossible Travel Speed detected."
    }

@app.get("/demo/status/{session_id}")
async def demo_status(session_id: str):
    """Returns current session state for console verification."""
    if session_id not in sessions_db:
        return {"error": "Session not found"}
    sess = sessions_db[session_id]
    return {
        "session_id": session_id,
        "state": sess["state"],
        "risk_score": sess["risk_score"],
        "calibration_samples": sess["calibration_samples"],
        "gpu_baseline": sess["baseline_invariants"].gpu_hash,
        "last_ip": sess["last_ip"],
        "xai_reasoning": sess["xai_reasoning"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
