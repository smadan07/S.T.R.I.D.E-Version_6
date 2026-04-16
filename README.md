# S.T.R.I.D.E. 🛡️
**Security Tracking & Real-time Identity Defense Engine**

An advanced, Zero-Trust cybersecurity MVP that continuously authenticates users through multi-modal behavioral biometrics and deep learning, eliminating the reliance on static and easily compromised passwords.

---

## 📖 Overview
S.T.R.I.D.E. transitions security from "point-in-time" authentication to continuous, background verification. By leveraging a **Dense MLP Autoencoder** trained on real-world datasets (like the *CMU Keystroke Dynamics Benchmark*), the system continuously computes a reconstruction loss for user interactions. High loss translates directly to a high anomaly score—instantly flagging session hijacking, credential stuffing, or synthetic bot interactions.

## ✨ Core Features
* 🧠 **Deep Learning Anomaly Engine:** Utilizes an Autoencoder design to model behavioral baselines dynamically based on keystroke flight times, dwell times, and cadence.
* 💻 **Hardware Invariant Fingerprinting:** Collects underlying hardware configurations (WebGL, Canvas, screen telemetry) hashed irreversibly to prevent device spoofing.
* 🌍 **Geovelocity Tracking:** Monitors IP/ASN changes and distance/speed vectors over time to flag impossible physical travel between distinct login attempts.
* 🔒 **Continuous & Zero-Trust:** Operates entirely in the background, analyzing telemetry with zero UX friction and strict local-first processing protocols.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch / TensorFlow, Scikit-learn, Pandas, NumPy
* **Frontend:** Vanilla JavaScript, HTML5, CSS3

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/STRIDE.git
   cd STRIDE/v6
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: ensure FastAPI, Uvicorn, Pandas, and PyTorch/Sklearn are installed)*

3. Process the dataset (Optional - if required by the pipeline):
   ```bash
   python prepare_real_dataset.py
   ```

4. Train the models:
   ```bash
   python train.py
   ```

5. Start the backend API:
   ```bash
   uvicorn main:app --reload
   ```

6. **Serve the Frontend:** Open `index.html` in your browser (or use a local server like Live Server).

## 📊 How It Works
1. **Telemetry Ingestion:** The `sensor.js` module invisibly tracks mouse movements, hardware metrics, and keystroke dynamics.
2. **Scoring:** The FastAPI backend receives the telemetry payload and passes it through the autoencoder.
3. **Thresholding:** Normal behavior yields low reconstruction loss. If a threat actor hijacks the session, their differing biomechanics will cause a spike in loss, crossing the threat threshold and freezing the session.

## 🔬 Penetration & Adversarial Testing
This platform has been stress-tested across a multitude of vectors including:
* Session Hijacking
* WebGL / User-Agent Spoofing
* Replay Attacks
* Geovelocity manipulation

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](#) if you want to contribute.

## 📜 License
[MIT](https://choosealicense.com/licenses/mit/)
