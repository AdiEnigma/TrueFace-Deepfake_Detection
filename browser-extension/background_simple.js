/**
 * TrueFace Browser Extension - Simplified Background Script
 */

console.log('TrueFace background script loaded');

// Extension installation
chrome.runtime.onInstalled.addListener((details) => {
  console.log('TrueFace extension installed:', details.reason);
  
  // Set default settings
  chrome.storage.sync.set({
    enabled: true,
    backendUrl: 'ws://localhost:8000/ws',
    sensitivity: 'medium',
    showOverlay: true,
    trustThreshold: 70,
    notifications: true
  });
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Background received message:', message);
  
  switch (message.type) {
    case 'GET_SETTINGS':
      chrome.storage.sync.get([
        'enabled', 'backendUrl', 'sensitivity', 
        'showOverlay', 'trustThreshold', 'notifications'
      ], (settings) => {
        sendResponse(settings);
      });
      return true;
      
    case 'UPDATE_SETTINGS':
      chrome.storage.sync.set(message.settings, () => {
        sendResponse({ success: true });
      });
      return true;
      
    case 'BACKEND_STATUS':
      // Update extension badge
      try {
        const badgeText = message.connected ? '✓' : '✗';
        const badgeColor = message.connected ? '#22c55e' : '#ef4444';
        
        chrome.action.setBadgeText({
          text: badgeText,
          tabId: sender.tab?.id
        });
        
        chrome.action.setBadgeBackgroundColor({
          color: badgeColor,
          tabId: sender.tab?.id
        });
      } catch (error) {
        console.log('Badge update failed:', error);
      }
      break;
  }
});
