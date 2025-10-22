# TrueFace Browser Extension

Real-time deepfake detection for video calls using advanced AI technology.

## 🚀 Features

- **Real-time Detection**: Analyze video streams during calls
- **Multi-platform Support**: Works on Google Meet, Zoom, Teams, WebEx
- **Trust Scores**: Visual indicators showing authenticity percentage
- **Privacy First**: All processing done locally via your backend
- **Customizable**: Adjustable sensitivity and trust thresholds
- **Smart Notifications**: Alerts for suspicious activity

## 📋 Prerequisites

1. **TrueFace Backend Running**:
   ```bash
   cd "c:\Users\ADITYA\Desktop\TrueFace\TrueFace-Backend"
   python main.py
   ```
   Backend should be accessible at `http://localhost:8000`

2. **Chrome/Edge Browser** (Chromium-based)

## 🔧 Installation

### Method 1: Load as Unpacked Extension (Development)

1. **Open Chrome Extensions**:
   - Go to `chrome://extensions/`
   - Enable "Developer mode" (top right toggle)

2. **Load Extension**:
   - Click "Load unpacked"
   - Select the `browser-extension` folder
   - Extension should appear in your extensions list

3. **Pin Extension**:
   - Click the extensions icon (puzzle piece) in toolbar
   - Pin TrueFace extension for easy access

### Method 2: Package for Distribution

1. **Create Extension Package**:
   ```bash
   # Zip the browser-extension folder
   # Upload to Chrome Web Store (requires developer account)
   ```

## 🎯 Usage

### 1. Start Your Backend
```bash
cd "c:\Users\ADITYA\Desktop\TrueFace\TrueFace-Backend"
python main.py
```

### 2. Configure Extension
- Click the TrueFace icon in your browser toolbar
- Verify backend connection (should show "Backend Online")
- Adjust settings as needed:
  - **Detection Sensitivity**: Low/Medium/High
  - **Trust Threshold**: 30-95% (default: 70%)
  - **Show Overlay**: Enable/disable visual indicators
  - **Notifications**: Enable/disable alerts

### 3. Join Video Call
- Go to supported platforms:
  - Google Meet: `meet.google.com`
  - Zoom: `zoom.us`
  - Microsoft Teams: `teams.microsoft.com`
  - WebEx: `webex.com`

### 4. Monitor Trust Scores
- Extension automatically detects participant videos
- Trust scores appear as overlays on each video
- Color coding:
  - 🟢 **Green (80-100%)**: High trust
  - 🟡 **Yellow (60-79%)**: Medium trust
  - 🔴 **Red (0-59%)**: Low trust / Suspicious

## ⚙️ Settings

### Backend Configuration
- **Backend URL**: `ws://localhost:8000/ws` (default)
- **Connection Test**: Verify backend connectivity

### Detection Settings
- **Sensitivity**:
  - **Low**: Fewer false positives, may miss subtle deepfakes
  - **Medium**: Balanced detection (recommended)
  - **High**: More sensitive, may have more false positives

- **Trust Threshold**: Percentage below which participants are flagged as suspicious

### Display Options
- **Show Overlay**: Toggle visual trust score indicators
- **Notifications**: Enable browser notifications for low trust scores

## 🔧 Troubleshooting

### Extension Not Working
1. **Check Backend**: Ensure TrueFace backend is running on `localhost:8000`
2. **Test Connection**: Use "Test Connection" button in extension popup
3. **Reload Extension**: Go to `chrome://extensions/` and reload TrueFace
4. **Check Console**: Open DevTools (F12) and check for errors

### No Trust Scores Showing
1. **Enable Overlay**: Check that "Show Overlay" is enabled
2. **Supported Site**: Verify you're on a supported video call platform
3. **Video Detection**: Extension needs to detect participant videos
4. **Backend Connection**: Ensure WebSocket connection is active

### Poor Detection Accuracy
1. **Fine-tune Model**: Run the advanced fine-tuning script
2. **Adjust Sensitivity**: Try different sensitivity settings
3. **Check Video Quality**: Higher quality videos give better results
4. **Lighting Conditions**: Ensure good lighting in video calls

## 🛡️ Privacy & Security

- **Local Processing**: All analysis done via your local backend
- **No Data Collection**: Extension doesn't collect or store personal data
- **Secure Communication**: WebSocket connection to localhost only
- **User Control**: Full control over when detection is active

## 📁 File Structure

```
browser-extension/
├── manifest.json          # Extension configuration
├── background.js           # Extension background script
├── content.js             # Content script (injected into pages)
├── popup.html             # Extension popup interface
├── popup.js               # Popup functionality
├── popup.css              # Popup styling
├── overlay.css            # Video overlay styling
├── icons/                 # Extension icons
└── README.md              # This file
```

## 🔄 Updates

To update the extension:
1. Pull latest changes from repository
2. Go to `chrome://extensions/`
3. Click reload button on TrueFace extension
4. Restart browser if needed

## 🐛 Known Issues

1. **Video Detection Delay**: May take a few seconds to detect new participants
2. **Platform Updates**: Video call platforms may change their DOM structure
3. **Performance**: High CPU usage during active analysis

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify backend is running and accessible
3. Check browser console for error messages
4. Ensure you're using a supported video call platform

## 🚀 Advanced Usage

### Custom Backend URL
If running backend on different port or server:
1. Open extension popup
2. Change "Backend Server" URL
3. Test connection
4. Save settings

### Multiple Participants
Extension automatically handles multiple participants:
- Each participant gets their own trust score overlay
- Scores update independently
- Notifications triggered per participant

### Performance Optimization
- Extension analyzes frames every 2 seconds by default
- Adjust `FRAME_INTERVAL_MS` in content.js for different intervals
- Higher intervals = better performance, less frequent updates

## 🎉 Success!

Your TrueFace browser extension is now ready to protect you from deepfakes in real-time during video calls!
