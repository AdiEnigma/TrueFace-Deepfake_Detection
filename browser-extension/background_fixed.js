/**
 * TrueFace Browser Extension - Error-Free Background Script
 * No Chrome API calls that can cause "Cannot read properties of undefined" errors
 */

console.log('TrueFace background script starting...');

// Simple storage wrapper to avoid undefined errors
const safeStorage = {
  get: function(keys, callback) {
    try {
      if (chrome && chrome.storage && chrome.storage.sync) {
        chrome.storage.sync.get(keys, callback);
      } else {
        callback({});
      }
    } catch (e) {
      console.log('Storage get error:', e);
      callback({});
    }
  },
  
  set: function(items, callback) {
    try {
      if (chrome && chrome.storage && chrome.storage.sync) {
        chrome.storage.sync.set(items, callback || function() {});
      } else if (callback) {
        callback();
      }
    } catch (e) {
      console.log('Storage set error:', e);
      if (callback) callback();
    }
  }
};

// Safe initialization
function initializeExtension() {
  console.log('Initializing TrueFace extension...');
  
  // Set default settings
  const defaultSettings = {
    enabled: true,
    backendUrl: 'ws://localhost:8000/ws',
    sensitivity: 'medium',
    showOverlay: true,
    trustThreshold: 70,
    notifications: true
  };
  
  safeStorage.set(defaultSettings, function() {
    console.log('Default settings saved');
  });
}

// Safe message handler
function handleMessage(message, sender, sendResponse) {
  console.log('Received message:', message.type);
  
  try {
    switch (message.type) {
      case 'GET_SETTINGS':
        safeStorage.get([
          'enabled', 'backendUrl', 'sensitivity', 
          'showOverlay', 'trustThreshold', 'notifications'
        ], function(settings) {
          sendResponse(settings || {
            enabled: true,
            backendUrl: 'ws://localhost:8000/ws',
            sensitivity: 'medium',
            showOverlay: true,
            trustThreshold: 70,
            notifications: true
          });
        });
        return true; // Keep message channel open
        
      case 'UPDATE_SETTINGS':
        safeStorage.set(message.settings || {}, function() {
          sendResponse({ success: true });
        });
        return true;
        
      case 'BACKEND_STATUS':
        console.log('Backend status update:', message.connected ? 'Connected' : 'Disconnected');
        sendResponse({ received: true });
        break;
        
      case 'DEEPFAKE_DETECTED':
        console.log('Deepfake detection:', message);
        sendResponse({ received: true });
        break;
        
      default:
        console.log('Unknown message type:', message.type);
        sendResponse({ error: 'Unknown message type' });
    }
  } catch (error) {
    console.error('Message handling error:', error);
    sendResponse({ error: error.message });
  }
}

// Initialize when extension starts
try {
  // Check if runtime is available
  if (typeof chrome !== 'undefined' && chrome.runtime) {
    
    // Handle installation
    if (chrome.runtime.onInstalled) {
      chrome.runtime.onInstalled.addListener(function(details) {
        console.log('Extension installed:', details.reason);
        initializeExtension();
      });
    }
    
    // Handle messages
    if (chrome.runtime.onMessage) {
      chrome.runtime.onMessage.addListener(handleMessage);
    }
    
    console.log('TrueFace background script loaded successfully');
    
  } else {
    console.error('Chrome runtime not available');
  }
} catch (error) {
  console.error('Background script initialization error:', error);
}

// Initialize immediately for existing installations
initializeExtension();
