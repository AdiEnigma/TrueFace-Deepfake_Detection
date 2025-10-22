/**
 * TrueFace Browser Extension - Minimal Background Script
 * Only essential functionality to avoid Chrome API errors
 */

console.log('TrueFace minimal background script loaded');

// Safe extension installation handler
if (chrome.runtime && chrome.runtime.onInstalled) {
  chrome.runtime.onInstalled.addListener((details) => {
    console.log('TrueFace extension installed:', details.reason);
    
    // Set default settings safely
    if (chrome.storage && chrome.storage.sync) {
      chrome.storage.sync.set({
        enabled: true,
        backendUrl: 'ws://localhost:8000/ws',
        sensitivity: 'medium',
        showOverlay: true,
        trustThreshold: 70,
        notifications: true
      });
    }
  });
}

// Safe message handler
if (chrome.runtime && chrome.runtime.onMessage) {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background received message:', message);
    
    try {
      switch (message.type) {
        case 'GET_SETTINGS':
          if (chrome.storage && chrome.storage.sync) {
            chrome.storage.sync.get([
              'enabled', 'backendUrl', 'sensitivity', 
              'showOverlay', 'trustThreshold', 'notifications'
            ], (settings) => {
              sendResponse(settings || {});
            });
            return true;
          }
          break;
          
        case 'UPDATE_SETTINGS':
          if (chrome.storage && chrome.storage.sync) {
            chrome.storage.sync.set(message.settings || {}, () => {
              sendResponse({ success: true });
            });
            return true;
          }
          break;
          
        case 'BACKEND_STATUS':
          // Simple logging instead of badge updates
          console.log('Backend status:', message.connected ? 'Connected' : 'Disconnected');
          break;
          
        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
      sendResponse({ error: error.message });
    }
  });
}
