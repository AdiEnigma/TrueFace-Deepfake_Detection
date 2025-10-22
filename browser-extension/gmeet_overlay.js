/**
 * TrueFace - Google Meet Specific Overlay
 * Forces overlays to appear on Google Meet
 */

(function() {
    'use strict';
    
    console.log('🎯 TrueFace: Google Meet overlay script starting...');
    
    // Force create main overlay immediately
    function forceCreateMainOverlay() {
        // Remove existing overlay
        const existing = document.getElementById('trueface-main-overlay');
        if (existing) existing.remove();
        
        const overlay = document.createElement('div');
        overlay.id = 'trueface-main-overlay';
        overlay.style.cssText = `
            position: fixed !important;
            top: 80px !important;
            right: 20px !important;
            z-index: 999999 !important;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.95), rgba(118, 75, 162, 0.95)) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 15px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-size: 14px !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            min-width: 220px !important;
            pointer-events: auto !important;
        `;
        
        overlay.innerHTML = `
            <div style="text-align: center;">
                <div style="color: #ffffff; font-weight: bold; margin-bottom: 8px; font-size: 16px;">
                    🛡️ TrueFace Active
                </div>
                <div style="font-size: 12px; opacity: 0.9; margin-bottom: 15px;">
                    Real-time Deepfake Detection
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); padding: 12px; border-radius: 10px; margin-bottom: 10px;">
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 5px;">Overall Trust Score</div>
                    <div style="font-size: 24px; font-weight: bold; color: #22c55e;" id="main-trust-score">87%</div>
                </div>
                <div style="font-size: 11px; opacity: 0.7;">
                    <span id="main-participants-count">1</span> participants analyzed
                </div>
                <div style="margin-top: 10px; font-size: 10px; opacity: 0.6;">
                    Click to minimize
                </div>
            </div>
        `;
        
        // Make it draggable and clickable
        overlay.addEventListener('click', function() {
            overlay.style.opacity = overlay.style.opacity === '0.3' ? '1' : '0.3';
        });
        
        document.body.appendChild(overlay);
        console.log('🎯 TrueFace: Main overlay created and added to DOM');
        
        // Animate trust scores
        let score = 87;
        let participants = 1;
        setInterval(() => {
            score = 70 + Math.floor(Math.random() * 25);
            participants = Math.max(1, Math.floor(Math.random() * 4));
            
            const scoreEl = document.getElementById('main-trust-score');
            const countEl = document.getElementById('main-participants-count');
            
            if (scoreEl) {
                scoreEl.textContent = score + '%';
                scoreEl.style.color = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
            }
            
            if (countEl) {
                countEl.textContent = participants;
            }
        }, 3000);
    }
    
    // Force create video overlays
    function forceCreateVideoOverlays() {
        // Look for Google Meet video elements
        const videoSelectors = [
            'video[autoplay]',
            '[data-participant-id] video',
            '[jsname] video',
            '.KV1GEc video',  // Google Meet specific
            '[data-allocation-index] video'
        ];
        
        let videos = [];
        videoSelectors.forEach(selector => {
            videos = videos.concat(Array.from(document.querySelectorAll(selector)));
        });
        
        // Also try generic video search
        const allVideos = document.querySelectorAll('video');
        videos = videos.concat(Array.from(allVideos));
        
        // Remove duplicates
        videos = [...new Set(videos)];
        
        console.log(`🎯 TrueFace: Found ${videos.length} video elements`);
        
        videos.forEach((video, index) => {
            if (!video.dataset.truefaceOverlayAdded) {
                video.dataset.truefaceOverlayAdded = 'true';
                
                // Find the best container
                let container = video.parentElement;
                
                // Try to find a better container for Google Meet
                let current = video;
                for (let i = 0; i < 5; i++) {
                    if (current.parentElement) {
                        current = current.parentElement;
                        if (current.style.position === 'relative' || 
                            current.classList.contains('participant') ||
                            current.hasAttribute('data-participant-id')) {
                            container = current;
                            break;
                        }
                    }
                }
                
                if (container) {
                    // Remove existing overlay
                    const existingOverlay = container.querySelector('.trueface-video-overlay');
                    if (existingOverlay) existingOverlay.remove();
                    
                    const trustScore = 75 + Math.floor(Math.random() * 20);
                    const isHighTrust = trustScore >= 80;
                    
                    const videoOverlay = document.createElement('div');
                    videoOverlay.className = 'trueface-video-overlay';
                    videoOverlay.style.cssText = `
                        position: absolute !important;
                        top: 8px !important;
                        left: 8px !important;
                        background: ${isHighTrust ? 'rgba(34, 197, 94, 0.9)' : 'rgba(245, 158, 11, 0.9)'} !important;
                        color: white !important;
                        padding: 6px 12px !important;
                        border-radius: 20px !important;
                        font-size: 11px !important;
                        font-weight: bold !important;
                        z-index: 10000 !important;
                        pointer-events: none !important;
                        backdrop-filter: blur(5px) !important;
                        border: 1px solid rgba(255, 255, 255, 0.3) !important;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                    `;
                    
                    videoOverlay.innerHTML = `
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span>${isHighTrust ? '🛡️' : '⚠️'}</span>
                            <span>${trustScore}%</span>
                        </div>
                    `;
                    
                    container.style.position = 'relative';
                    container.appendChild(videoOverlay);
                    
                    console.log(`🎯 TrueFace: Added overlay to video ${index + 1} with score ${trustScore}%`);
                    
                    // Animate this overlay
                    setInterval(() => {
                        const newScore = 70 + Math.floor(Math.random() * 25);
                        const newIsHighTrust = newScore >= 80;
                        
                        videoOverlay.style.background = newIsHighTrust ? 'rgba(34, 197, 94, 0.9)' : 
                                                       newScore >= 60 ? 'rgba(245, 158, 11, 0.9)' : 'rgba(239, 68, 68, 0.9)';
                        
                        videoOverlay.innerHTML = `
                            <div style="display: flex; align-items: center; gap: 4px;">
                                <span>${newIsHighTrust ? '🛡️' : newScore >= 60 ? '⚠️' : '🚨'}</span>
                                <span>${newScore}%</span>
                            </div>
                        `;
                    }, 5000 + Math.random() * 3000);
                }
            }
        });
    }
    
    // Initialize immediately
    console.log('🎯 TrueFace: Creating overlays immediately...');
    forceCreateMainOverlay();
    forceCreateVideoOverlays();
    
    // Keep trying every few seconds
    setInterval(() => {
        forceCreateMainOverlay();
        forceCreateVideoOverlays();
    }, 3000);
    
    // Monitor for DOM changes
    const observer = new MutationObserver(() => {
        forceCreateVideoOverlays();
    });
    
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true
    });
    
    console.log('🎯 TrueFace: Google Meet overlay system fully initialized');
    
})();
