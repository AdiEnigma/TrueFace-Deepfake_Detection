/**
 * TrueFace Browser Extension - Background Script
 * Manages extension lifecycle and settings
 */

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
      return true; // Keep message channel open
      
    case 'UPDATE_SETTINGS':
      chrome.storage.sync.set(message.settings, () => {
        sendResponse({ success: true });
      });
      return true;
      
    case 'DEEPFAKE_DETECTED':
      if (message.trustScore < 50) {
        // Show notification for low trust score
        chrome.storage.sync.get(['notifications'], (settings) => {
          if (settings.notifications) {
            try {
              chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: 'TrueFace Alert',
                message: `Low trust score detected: ${message.trustScore}%`
              });
            } catch (error) {
              console.log('Notifications not available:', error);
            }
          }
        });
      }
      break;
      
    case 'BACKEND_STATUS':
      // Update extension badge based on backend connection
      const badgeText = message.connected ? '✓' : '✗';
      const badgeColor = message.connected ? '#22c55e' : '#ef4444';
      
      chrome.action.setBadgeText({
        text: badgeText,
        tabId: sender.tab.id
      });
      
      chrome.action.setBadgeBackgroundColor({
        color: badgeColor,
        tabId: sender.tab.id
      });
      break;
  }
});

// Handle tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    const supportedSites = [
      'meet.google.com',
      'zoom.us',
      'teams.microsoft.com',
      'webex.com'
    ];
    
    const isSupported = supportedSites.some(site => tab.url.includes(site));
    
    if (isSupported) {
      console.log('TrueFace: Supported video call site detected');
      // Extension will automatically inject via content script
    }
  }
});

// Context menu (optional) - wrapped in try-catch
try {
  chrome.contextMenus.create({
    id: 'trueface-toggle',
    title: 'Toggle TrueFace Detection',
    contexts: ['page']
  });

  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'trueface-toggle') {
      chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_DETECTION' });
    }
  });
} catch (error) {
  console.log('Context menu not available:', error);
}
