/**
 * TrueFace - Real Deepfake Detection Analysis
 * Captures video frames and sends to backend for actual analysis
 * Lightweight and non-intrusive
 */

(function() {
    'use strict';
    
    console.log('🎯 TrueFace: Real analysis script loaded');
    
    const CONFIG = {
        backendUrl: 'ws://localhost:8000/ws',
        analysisInterval: 2000, // Analyze every 2 seconds for real-time updates
        maxFrameSize: 480, // Max frame dimension
        enableLogging: true
    };
    
    let ws = null;
    let analysisInProgress = false;
    const videoAnalysisMap = new Map(); // Track analysis per video
    let analysisCounter = 0; // Track number of analyses
    
    // Connect to backend WebSocket
    function connectToBackend() {
        try {
            ws = new WebSocket(CONFIG.backendUrl);
            
            ws.onopen = () => {
                console.log('🎯 TrueFace: Connected to backend');
                sendMessage({
                    type: 'start_stream',
                    message: 'TrueFace extension connected'
                });
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('🎯 TrueFace: Backend response:', data);
                    
                    if (data.type === 'video_analysis_result') {
                        handleAnalysisResult(data);
                    }
                } catch (error) {
                    console.error('Error parsing backend message:', error);
                }
            };
            
            ws.onerror = (error) => {
                console.error('🎯 TrueFace: WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('🎯 TrueFace: Disconnected from backend, reconnecting...');
                setTimeout(connectToBackend, 5000);
            };
            
        } catch (error) {
            console.error('🎯 TrueFace: Connection error:', error);
            setTimeout(connectToBackend, 5000);
        }
    }
    
    // Send message to backend
    function sendMessage(message) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify(message));
            } catch (error) {
                console.error('Error sending message:', error);
            }
        }
    }
    
    // Capture video frame and send for analysis
    function captureAndAnalyzeFrame(video, participantId) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        try {
            
            // Create canvas for frame capture
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size (smaller for performance)
            const scale = Math.min(1, CONFIG.maxFrameSize / Math.max(video.videoWidth, video.videoHeight));
            canvas.width = video.videoWidth * scale;
            canvas.height = video.videoHeight * scale;
            
            // Draw video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const frameData = canvas.toDataURL('image/jpeg', 0.7);
            
            // Send for analysis
            analysisCounter++;
            sendMessage({
                type: 'video_frame',
                data: frameData,
                participant_id: participantId,
                timestamp: new Date().toISOString(),
                frame_number: analysisCounter
            });
            
            console.log(`🎯 TrueFace: Frame ${analysisCounter} sent for analysis - ${participantId}`);
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            analysisInProgress = false;
        }
    }
    
    // Handle analysis results from backend
    function handleAnalysisResult(result) {
        const { participant_id, results } = result;
        
        if (!results) return;
        
        const trustScore = Math.round((1 - results.overall_score) * 100);
        const confidence = results.confidence;
        const classification = results.classification;
        
        console.log(`🎯 TrueFace: ${participant_id} - Trust: ${trustScore}%, Classification: ${classification}`);
        
        // Store result
        videoAnalysisMap.set(participant_id, {
            trustScore,
            confidence,
            classification,
            timestamp: new Date()
        });
        
        // Update UI
        updateVideoOverlay(participant_id, trustScore, classification);
    }
    
    // Update video overlay with real analysis result
    function updateVideoOverlay(participantId, trustScore, classification) {
        // Find video element
        const videos = document.querySelectorAll('video[autoplay]');
        
        videos.forEach((video, index) => {
            if (!video.dataset.participantId) {
                video.dataset.participantId = `participant-${index}`;
            }
            
            if (video.dataset.participantId === participantId || index === 0) {
                let container = video.parentElement;
                let depth = 0;
                
                while (container && depth < 8) {
                    if (container.style.position === 'relative' || 
                        container.hasAttribute('data-participant-id')) {
                        break;
                    }
                    container = container.parentElement;
                    depth++;
                }
                
                if (!container) container = video.parentElement;
                
                // Remove existing overlay to force update
                const existingOverlay = container.querySelector('.trueface-analysis-overlay');
                if (existingOverlay) {
                    existingOverlay.remove();
                }
                
                // Determine color based on trust score
                let color = '#22c55e'; // Green
                let status = 'Verified';
                
                if (trustScore < 50) {
                    color = '#ef4444'; // Red
                    status = 'Suspicious';
                } else if (trustScore < 70) {
                    color = '#f59e0b'; // Yellow
                    status = 'Caution';
                }
                
                // Create overlay
                const overlay = document.createElement('div');
                overlay.className = 'trueface-analysis-overlay';
                overlay.style.cssText = `
                    position: absolute !important;
                    top: 10px !important;
                    right: 10px !important;
                    z-index: 9999 !important;
                    pointer-events: none !important;
                    background: rgba(0, 0, 0, 0.85) !important;
                    backdrop-filter: blur(10px) !important;
                    border-radius: 12px !important;
                    padding: 12px !important;
                    min-width: 130px !important;
                    text-align: center !important;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
                    border: 1px solid rgba(255, 255, 255, 0.2) !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                `;
                
                overlay.innerHTML = `
                    <div style="
                        font-size: 28px !important;
                        font-weight: bold !important;
                        color: ${color} !important;
                        margin-bottom: 4px !important;
                        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
                        animation: pulse 0.5s ease-in-out !important;
                    ">
                        ${trustScore}%
                    </div>
                    <div style="
                        font-size: 10px !important;
                        color: rgba(255, 255, 255, 0.8) !important;
                        text-transform: uppercase !important;
                        letter-spacing: 0.5px !important;
                        margin-bottom: 6px !important;
                    ">
                        Authenticity
                    </div>
                    <div style="
                        font-size: 11px !important;
                        color: ${color} !important;
                        font-weight: 600 !important;
                    ">
                        ${status}
                    </div>
                    <div style="
                        font-size: 9px !important;
                        color: rgba(255, 255, 255, 0.6) !important;
                        margin-top: 4px !important;
                    ">
                        ${classification}
                    </div>
                    <style>
                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.8; }
                        }
                    </style>
                `;
                
                container.style.position = 'relative';
                container.appendChild(overlay);
            }
        });
    }
    
    // Continuously analyze videos
    function startContinuousAnalysis() {
        setInterval(() => {
            const videos = document.querySelectorAll('video[autoplay]');
            
            videos.forEach((video, index) => {
                // Check if video has data and is playing
                if (video.readyState >= video.HAVE_CURRENT_DATA && !video.paused) {
                    const participantId = `participant-${index}`;
                    captureAndAnalyzeFrame(video, participantId);
                }
            });
        }, CONFIG.analysisInterval);
    }
    
    // Initialize
    console.log('🎯 TrueFace: Initializing real deepfake detection...');
    connectToBackend();
    
    // Wait a bit for page to load, then start analysis
    setTimeout(() => {
        startContinuousAnalysis();
        console.log('🎯 TrueFace: Real-time analysis started - updating every 2 seconds');
    }, 1500);
    
})();
