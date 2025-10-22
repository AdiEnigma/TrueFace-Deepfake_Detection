<div align="center">
  <h1>🖼️ TrueFace</h1>
  <h3>Advanced Real-time Deepfake Detection System</h3>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![Open Issues](https://img.shields.io/github/issues/AdiEnigma/TrueFace-Deepfake_Detection)](https://github.com/AdiEnigma/TrueFace-Deepfake_Detection/issues)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

  <p align="center">
    A cutting-edge solution for detecting deepfake media in real-time with advanced machine learning models and WebSocket communication.
  </p>
</div>

## 🌟 Key Features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [Research Papers](#) - Links to relevant research papers
- [Dataset Collection](#) - Information about training data
- [Model Architecture](#) - Technical details about the models

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special thanks to the open-source community for valuable resources and tools
- Inspired by the latest research in deepfake detection

## 📧 Contact

For any questions or suggestions, please reach out to [Your Email] or open an issue on GitHub.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Report Bugs**: File an issue if you find any bugs or have suggestions.
2. **Feature Requests**: Suggest new features or improvements.
3. **Code Contributions**: Submit pull requests for bug fixes or new features.

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/TrueFace-Deepfake_Detection.git
   cd TrueFace-Deepfake_Detection
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for better code clarity
- Write docstrings for all public functions and classes
- Include tests for new features

## 🏗️ Project Structure

```
TrueFace/
├── TrueFace-Backend/    # Backend server implementation
│   ├── api/             # API endpoints and routes
│   ├── core/            # Core business logic
│   ├── models/          # Database models
│   └── services/        # Business services
├── browser-extension/   # Chrome extension source
│   ├── src/             # Extension source code
│   ├── assets/          # Images, icons, etc.
│   └── manifest.json    # Extension configuration
├── models/              # Pre-trained ML models
│   ├── video/           # Video analysis models
│   └── audio/           # Audio analysis models
├── src/                 # Core Python modules
│   ├── __init__.py
│   ├── deepfake_detector.py  # Main detection logic
│   ├── mesonet_model.py      # MesoNet implementation
│   ├── buffer_manager.py     # Data buffering
│   ├── database_manager.py   # Database operations
│   └── utils.py              # Utility functions
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── config.py            # Configuration settings
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🌟 Key Features

TrueFace is an advanced deepfake detection system that leverages state-of-the-art machine learning models to identify manipulated media in real-time. The system is designed with a modular architecture, supporting both video and audio analysis through WebSocket communication for seamless integration with various frontend applications.

### Core Functionality
- **Real-time WebSocket Communication**: Bi-directional data streaming between backend and frontend
- **Advanced Deepfake Detection**: Video and audio analysis using ONNX-optimized models
- **Intelligent Buffer Management**: Efficient frame/audio processing with asyncio queues
- **Session Management**: Complete user session lifecycle with analytics
- **Database Logging**: SQLite-based storage for analysis results and system logs

### Performance Optimizations
- **GPU/CPU Fallback**: Automatic hardware detection and graceful fallback
- **Frame Sampling**: Configurable frame skip rates to reduce computational load
- **Batch Processing**: Efficient batch analysis of video frames and audio chunks
- **ONNX Model Support**: Optimized inference with ONNX runtime
- **Async Architecture**: Non-blocking operations throughout the system

### Analytics & Monitoring
- **Real-time Analytics**: Live session statistics and detection metrics
- **Historical Data**: Comprehensive logging and trend analysis
- **Health Monitoring**: System health checks and performance metrics
- **Configurable Alerts**: Real-time deepfake detection notifications

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core programming language
- **FastAPI & Uvicorn** - High-performance web framework
- **ONNX Runtime** - Optimized model inference
- **OpenCV & NumPy** - Image/video processing
- **SQLite/PostgreSQL** - Data storage
- **WebSockets** - Real-time communication
- **Asyncio** - Asynchronous task management

### Machine Learning
- **Deep Learning Models**: Custom-trained CNNs for deepfake detection
- **Transfer Learning**: Leveraging pre-trained models
- **ONNX Optimization**: For production-grade performance
- **Data Augmentation**: Enhanced model robustness

### Frontend (Browser Extension)
- **HTML5/CSS3/JavaScript** - Core web technologies
- **WebSockets** - Real-time communication
- **Chrome Extension API** - Browser integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional, CUDA-compatible)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdiEnigma/TrueFace-Deepfake_Detection.git
   cd TrueFace-Deepfake_Detection
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python -c "from src.database_manager import init_db; init_db()"
   ```

5. **Start the server**
   ```bash
   python main.py
   ```
   The server will start on `http://localhost:8000` by default.

### Browser Extension Setup

1. Open Chrome/Edge and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked" and select the `browser-extension` directory
4. Pin the extension to your toolbar for easy access

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd TrueFace
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Configuration
```bash
# Copy and modify configuration (optional)
cp config.py.example config.py
```

### 5. Initialize Database
The database will be automatically initialized on first run.

## 🚀 Quick Start

### Start the Server
```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

### WebSocket Connection
Connect to the WebSocket endpoint at `ws://localhost:8000/ws`

### API Endpoints
- `GET /` - Health check
- `GET /health` - Detailed system status
- `GET /logs/{session_id}` - Session analysis logs
- `GET /sessions` - Active sessions list
- `WS /ws` - Main WebSocket endpoint

## 📡 WebSocket API

### Connection Flow
1. Connect to `/ws` endpoint
2. Receive connection confirmation
3. Send stream configuration
4. Stream video/audio data
5. Receive real-time analysis results

### Message Types

#### Client → Server

**Start Stream**
```json
{
  "type": "start_stream",
  "config": {
    "video_enabled": true,
    "audio_enabled": true,
    "quality": "medium",
    "detection_sensitivity": "high"
  }
}
```

**Video Frame**
```json
{
  "type": "video_frame",
  "data": "base64_encoded_frame",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Audio Chunk**
```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Stop Stream**
```json
{
  "type": "stop_stream"
}
```

#### Server → Client

**Analysis Result**
```json
{
  "type": "video_analysis_result",
  "session_id": "session_123",
  "results": {
    "overall_score": 0.85,
    "confidence": 0.92,
    "is_deepfake": true,
    "detections": [...],
    "processing_time": 0.15
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# Server Configuration
TRUEFACE_HOST=0.0.0.0
TRUEFACE_PORT=8000
TRUEFACE_LOG_LEVEL=info

# Model Configuration
TRUEFACE_MODELS_DIR=models
TRUEFACE_USE_GPU=true
TRUEFACE_VIDEO_THRESHOLD=0.7
TRUEFACE_AUDIO_THRESHOLD=0.6

# Buffer Configuration
TRUEFACE_VIDEO_BATCH_SIZE=10
TRUEFACE_AUDIO_BATCH_SIZE=5

# Session Configuration
TRUEFACE_MAX_SESSION_DURATION=86400
TRUEFACE_INACTIVE_TIMEOUT=1800

# Database Configuration
TRUEFACE_DB_PATH=data/trueface.db
TRUEFACE_RETENTION_DAYS=30
```

## 🚀 Quick Start Guide

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start server**: `python main.py`
3. **Connect WebSocket**: `ws://localhost:8000/ws`
4. **Send video/audio data**: Use the WebSocket API
5. **Receive results**: Real-time deepfake detection results#   T r u e F a c e - D e e p f a k e - d e t e c i o n - 
 
 
