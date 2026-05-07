# FootAgent: Agentic Intelligence for Real-Time Football Analytics

FootAgent is a research-grade agentic pipeline designed to extract tactical intelligence from broadcast football video. It utilizes a multi-model architecture orchestrated via **LangGraph** to perform high-precision player tracking, strategic off-ball contribution analysis, and automated incident classification.

![FootAgent Dashboard Mockup](https://raw.githubusercontent.com/WalidMahmood/footagent_cse490d/main/docs/dashboard_preview.png)

## 🚀 Features

-   **Agentic Orchestration**: Uses LangGraph to dynamically manage model inference (YOLO, GAT, ViT) based on real-time game state.
-   **Temporal GAT Intelligence**: A Graph Attention Network that calculates per-player Off-Ball Contribution (OBC) and strategic threat levels.
-   **Robust ByteTrack Tracking**: Enhanced person tracking with Kalman Filter stabilization for consistent ID maintenance across occlusions.
-   **Real-Time 3D Dashboard**: Live telemetry visualization using Three.js, featuring dynamic threat glows and pitch-perfect coordinate mapping.
-   **Optimized VRAM Management**: Custom lazy-loading scheduler designed for high-performance inference on consumer GPUs (RTX 5050/3050).

## 🏗️ System Architecture

1.  **Tracking Node**: Person detection (YOLOv8) + ID association (ByteTrack).
2.  **Off-Ball Node**: Strategic analysis via Temporal GAT.
3.  **Trigger Node**: Event detection for potential foul incidents.
4.  **VAR Node**: Video classification (ViT) for incident validation.
5.  **Memory Node**: Thread-safe, atomic state persistence for live streaming.

## 🛠️ Tech Stack

-   **Backend**: FastAPI, LangGraph, PyTorch, PyTorch Geometric, Ultralytics.
-   **Frontend**: Three.js, Vanilla JS (ES6+), Glassmorphic CSS.
-   **Hardware**: Optimized for NVIDIA CUDA-accelerated environments.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/WalidMahmood/footagent_cse490d.git
cd footagent_cse490d

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Running the Pipeline

1.  **Start the Backend**:
    ```bash
    python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
    ```
2.  **Launch the Dashboard**:
    Open `dashboard/index.html` in your browser.
3.  **Upload Video**: Use the dashboard to upload a `.mp4` clip and watch the real-time tactical analysis.

## 📄 Research & Results

Detailed analysis, including hyperparameters and ablation studies, can be found in [FootAgent_Research_Paper.md](./FootAgent_Research_Paper.md).

-   **GAT Strategic Alignment**: 0.892
-   **Tracking MOTA**: 0.92
-   **Pipeline Throughput**: ~42 FPS (Full processing)

## 🤝 Authors

-   **Antigravity AI**
-   **Walid Mahmood**

---
*Developed for CSE490D Research Project.*
