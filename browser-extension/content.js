/**
 * TrueFace Browser Extension - Content Script
 * Injects TrueFace detection into video call pages
 */

(function() {
  'use strict';
  
  // Prevent multiple injections
  if (window.__TRUEFACE_EXTENSION_LOADED__) return;
  window.__TRUEFACE_EXTENSION_LOADED__ = true;
  
  console.log('TrueFace: Content script loaded on', window.location.hostname);
  
  let extensionEnabled = true;
  let settings = {};
  
  // Load extension settings
  chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (response) => {
    settings = response || {};
    extensionEnabled = settings.enabled !== false;
    
    if (extensionEnabled) {
      initializeTrueFace();
    }
  });
  
  function initializeTrueFace() {
    console.log('TrueFace: Initializing detection system...');
    
    // Wait for page to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', injectTrueFace);
    } else {
      injectTrueFace();
    }
  }
  
  function injectTrueFace() {
    // Create and inject the main TrueFace script
    const script = document.createElement('script');
    script.textContent = getTrueFaceScript();
    
    // Pass settings to the injected script
    const configScript = document.createElement('script');
    configScript.textContent = `
      window.TRUEFACE_EXTENSION_CONFIG = ${JSON.stringify({
        backendUrl: settings.backendUrl || 'ws://localhost:8000/ws',
        sensitivity: settings.sensitivity || 'medium',
        showOverlay: settings.showOverlay !== false,
        trustThreshold: settings.trustThreshold || 70,
        frameInterval: 2000,
        debug: true
      })};
    `;
    
    // Inject configuration first, then main script
    document.head.appendChild(configScript);
    document.head.appendChild(script);
    
    console.log('TrueFace: Detection system injected');
    
    // Monitor for backend connection status
    monitorBackendStatus();
  }
  
  function getTrueFaceScript() {
    // This is your existing inject.js content, adapted for extension
    return `
    (function(){
      'use strict';
      
      if(window.__TRUEFACE_INJECTED__) return;
      window.__TRUEFACE_INJECTED__ = true;
      
      const config = window.TRUEFACE_EXTENSION_CONFIG || {};
      const DEFAULT_CONFIG = {
        WS_URL: config.backendUrl || 'ws://localhost:8000/ws',
        RECONNECT_INTERVAL_MS: 5000,
        DEBUG: config.debug || true,
        FRAME_INTERVAL_MS: config.frameInterval || 2000,
        JPEG_QUALITY: 0.8,
        MAX_WIDTH: 640,
        SHOW_OVERLAY: config.showOverlay !== false,
        TRUST_THRESHOLD: config.trustThreshold || 70
      };
      
      const state = {
        ws: null,
        wsReady: false,
        reconnectTimer: 0,
        queue: [],
        videoEl: null,
        frameCanvas: null,
        frameCtx: null,
        frameTimer: 0,
        isStreaming: false,
        participants: new Map(),
        overlays: new Map()
      };
      
      // Enhanced overlay creation for multiple participants
      function createParticipantOverlay(videoElement, participantId) {
        if (!DEFAULT_CONFIG.SHOW_OVERLAY) return null;
        
        const overlay = document.createElement('div');
        overlay.className = 'trueface-participant-overlay';
        overlay.id = 'trueface-overlay-' + participantId;
        
        overlay.innerHTML = \`
          <div class="trueface-trust-indicator">
            <div class="trueface-trust-circle">
              <span class="trueface-trust-percentage">--</span>
            </div>
            <div class="trueface-trust-label">Trust Score</div>
            <div class="trueface-status">Analyzing...</div>
          </div>
        \`;
        
        // Position overlay relative to video
        overlay.style.position = 'absolute';
        overlay.style.top = '10px';
        overlay.style.right = '10px';
        overlay.style.zIndex = '9999';
        overlay.style.pointerEvents = 'none';
        
        // Find video container and append
        const container = videoElement.parentElement;
        if (container) {
          container.style.position = 'relative';
          container.appendChild(overlay);
        }
        
        return overlay;
      }
      
      // Enhanced video detection for different platforms
      function detectVideos() {
        const selectors = {
          'meet.google.com': 'video[autoplay]',
          'zoom.us': 'video',
          'teams.microsoft.com': 'video',
          'webex.com': 'video'
        };
        
        const hostname = window.location.hostname;
        const selector = selectors[hostname] || 'video';
        
        const videos = Array.from(document.querySelectorAll(selector));
        const remoteVideos = videos.filter(video => {
          // Filter out local video (usually muted or has specific attributes)
          return !video.muted && video.videoWidth > 0 && video.videoHeight > 0;
        });
        
        return remoteVideos;
      }
      
      // WebSocket connection with extension messaging
      function connectWebSocket() {
        if (state.ws) {
          try { state.ws.close(); } catch(e) {}
          state.ws = null;
        }
        
        try {
          state.ws = new WebSocket(DEFAULT_CONFIG.WS_URL);
          
          state.ws.onopen = function() {
            console.log('TrueFace: Connected to backend');
            state.wsReady = true;
            
            // Notify extension of connection status
            if (window.chrome && chrome.runtime) {
              chrome.runtime.sendMessage({
                type: 'BACKEND_STATUS',
                connected: true
              });
            }
          };
          
          state.ws.onmessage = function(event) {
            try {
              const data = JSON.parse(event.data);
              handleAnalysisResult(data);
            } catch(e) {
              console.error('TrueFace: Error parsing message:', e);
            }
          };
          
          state.ws.onclose = function() {
            console.log('TrueFace: Disconnected from backend');
            state.wsReady = false;
            
            // Notify extension of connection status
            if (window.chrome && chrome.runtime) {
              chrome.runtime.sendMessage({
                type: 'BACKEND_STATUS',
                connected: false
              });
            }
            
            // Reconnect
            setTimeout(connectWebSocket, DEFAULT_CONFIG.RECONNECT_INTERVAL_MS);
          };
          
        } catch(e) {
          console.error('TrueFace: WebSocket connection failed:', e);
          setTimeout(connectWebSocket, DEFAULT_CONFIG.RECONNECT_INTERVAL_MS);
        }
      }
      
      // Handle analysis results
      function handleAnalysisResult(data) {
        if (data.type === 'video_analysis_result' && data.results) {
          const trustScore = Math.round((1 - data.results.overall_score) * 100);
          const participantId = data.participant_id || 'unknown';
          
          updateParticipantOverlay(participantId, trustScore, data.results.confidence);
          
          // Notify extension of detection
          if (window.chrome && chrome.runtime) {
            chrome.runtime.sendMessage({
              type: 'DEEPFAKE_DETECTED',
              participantId: participantId,
              trustScore: trustScore,
              confidence: data.results.confidence
            });
          }
        }
      }
      
      // Update participant overlay
      function updateParticipantOverlay(participantId, trustScore, confidence) {
        const overlay = document.getElementById('trueface-overlay-' + participantId);
        if (!overlay) return;
        
        const percentageEl = overlay.querySelector('.trueface-trust-percentage');
        const statusEl = overlay.querySelector('.trueface-status');
        const circleEl = overlay.querySelector('.trueface-trust-circle');
        
        if (percentageEl) percentageEl.textContent = trustScore + '%';
        if (statusEl) {
          const status = trustScore >= DEFAULT_CONFIG.TRUST_THRESHOLD ? 'Trusted' : 'Suspicious';
          statusEl.textContent = status;
        }
        
        // Color coding
        if (circleEl) {
          let color = '#ef4444'; // Red
          if (trustScore >= 80) color = '#22c55e'; // Green
          else if (trustScore >= 60) color = '#f59e0b'; // Yellow
          
          circleEl.style.borderColor = color;
          circleEl.style.color = color;
        }
      }
      
      // Start video analysis
      function startVideoAnalysis() {
        const videos = detectVideos();
        
        videos.forEach((video, index) => {
          const participantId = 'participant_' + index;
          
          if (!state.participants.has(participantId)) {
            state.participants.set(participantId, video);
            
            // Create overlay for this participant
            const overlay = createParticipantOverlay(video, participantId);
            if (overlay) {
              state.overlays.set(participantId, overlay);
            }
          }
        });
        
        // Capture and analyze frames
        if (state.wsReady && videos.length > 0) {
          videos.forEach((video, index) => {
            const participantId = 'participant_' + index;
            captureAndAnalyzeFrame(video, participantId);
          });
        }
      }
      
      // Capture and analyze frame
      function captureAndAnalyzeFrame(video, participantId) {
        if (!video.videoWidth || !video.videoHeight) return;
        
        // Create canvas if needed
        if (!state.frameCanvas) {
          state.frameCanvas = document.createElement('canvas');
          state.frameCtx = state.frameCanvas.getContext('2d');
        }
        
        const canvas = state.frameCanvas;
        const ctx = state.frameCtx;
        
        // Set canvas size
        canvas.width = Math.min(video.videoWidth, DEFAULT_CONFIG.MAX_WIDTH);
        canvas.height = Math.min(video.videoHeight, DEFAULT_CONFIG.MAX_WIDTH);
        
        try {
          // Draw video frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Convert to base64
          const dataUrl = canvas.toDataURL('image/jpeg', DEFAULT_CONFIG.JPEG_QUALITY);
          const base64 = dataUrl.split(',')[1];
          
          // Send to backend
          if (state.ws && state.ws.readyState === 1) {
            state.ws.send(JSON.stringify({
              type: 'video_frame',
              data: base64,
              participant_id: participantId,
              timestamp: new Date().toISOString()
            }));
          }
          
        } catch(e) {
          console.error('TrueFace: Error capturing frame:', e);
        }
      }
      
      // Initialize TrueFace
      function init() {
        console.log('TrueFace: Initializing on', window.location.hostname);
        
        // Connect to backend
        connectWebSocket();
        
        // Start video analysis loop
        setInterval(startVideoAnalysis, DEFAULT_CONFIG.FRAME_INTERVAL_MS);
        
        // Monitor for new videos (dynamic content)
        const observer = new MutationObserver(() => {
          startVideoAnalysis();
        });
        
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      }
      
      // Start when ready
      if (document.readyState === 'complete') {
        setTimeout(init, 1000);
      } else {
        window.addEventListener('load', () => setTimeout(init, 1000));
      }
      
    })();
    `;
  }
  
  function monitorBackendStatus() {
    // Check backend connection periodically
    setInterval(() => {
      fetch('http://localhost:8000/health')
        .then(response => response.ok)
        .then(isHealthy => {
          chrome.runtime.sendMessage({
            type: 'BACKEND_STATUS',
            connected: isHealthy
          });
        })
        .catch(() => {
          chrome.runtime.sendMessage({
            type: 'BACKEND_STATUS',
            connected: false
          });
        });
    }, 10000); // Check every 10 seconds
  }
  
  // Listen for messages from extension popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
      case 'TOGGLE_DETECTION':
        extensionEnabled = !extensionEnabled;
        if (extensionEnabled) {
          initializeTrueFace();
        } else {
          // Remove overlays and stop detection
          document.querySelectorAll('.trueface-participant-overlay').forEach(el => el.remove());
        }
        sendResponse({ enabled: extensionEnabled });
        break;
        
      case 'UPDATE_SETTINGS':
        settings = { ...settings, ...message.settings };
        // Reinitialize with new settings
        if (extensionEnabled) {
          document.querySelectorAll('.trueface-participant-overlay').forEach(el => el.remove());
          setTimeout(initializeTrueFace, 1000);
        }
        sendResponse({ success: true });
        break;
    }
  });
  
})();
