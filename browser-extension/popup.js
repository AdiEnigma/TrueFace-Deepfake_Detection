/**
 * TrueFace Browser Extension - Popup Script
 * Handles extension popup UI and settings
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const enableToggle = document.getElementById('enableToggle');
    const backendUrl = document.getElementById('backendUrl');
    const sensitivitySelect = document.getElementById('sensitivitySelect');
    const trustThreshold = document.getElementById('trustThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const showOverlay = document.getElementById('showOverlay');
    const notifications = document.getElementById('notifications');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const testConnection = document.getElementById('testConnection');
    const resetSettings = document.getElementById('resetSettings');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Load current settings
    loadSettings();
    
    // Event listeners
    enableToggle.addEventListener('change', saveSettings);
    backendUrl.addEventListener('change', saveSettings);
    sensitivitySelect.addEventListener('change', saveSettings);
    trustThreshold.addEventListener('input', updateThresholdValue);
    trustThreshold.addEventListener('change', saveSettings);
    showOverlay.addEventListener('change', saveSettings);
    notifications.addEventListener('change', saveSettings);
    testConnection.addEventListener('click', testBackendConnection);
    resetSettings.addEventListener('click', resetToDefaults);
    
    function loadSettings() {
        showLoading(true);
        
        chrome.storage.sync.get([
            'enabled', 'backendUrl', 'sensitivity', 
            'showOverlay', 'trustThreshold', 'notifications'
        ], function(settings) {
            // Apply settings to UI
            enableToggle.checked = settings.enabled !== false;
            backendUrl.value = settings.backendUrl || 'ws://localhost:8000/ws';
            sensitivitySelect.value = settings.sensitivity || 'medium';
            trustThreshold.value = settings.trustThreshold || 70;
            thresholdValue.textContent = settings.trustThreshold || 70;
            showOverlay.checked = settings.showOverlay !== false;
            notifications.checked = settings.notifications !== false;
            
            showLoading(false);
            checkBackendStatus();
        });
    }
    
    function saveSettings() {
        const settings = {
            enabled: enableToggle.checked,
            backendUrl: backendUrl.value.trim(),
            sensitivity: sensitivitySelect.value,
            trustThreshold: parseInt(trustThreshold.value),
            showOverlay: showOverlay.checked,
            notifications: notifications.checked
        };
        
        chrome.storage.sync.set(settings, function() {
            console.log('TrueFace: Settings saved');
            
            // Notify content scripts of settings change
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                if (tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, {
                        type: 'UPDATE_SETTINGS',
                        settings: settings
                    });
                }
            });
        });
    }
    
    function updateThresholdValue() {
        thresholdValue.textContent = trustThreshold.value;
    }
    
    function testBackendConnection() {
        showLoading(true, 'Testing connection...');
        
        const url = backendUrl.value.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '/health');
        
        fetch(url, { 
            method: 'GET',
            timeout: 5000 
        })
        .then(response => {
            if (response.ok) {
                updateStatus('connected', 'Backend Connected');
                showNotification('✅ Connection successful!', 'success');
            } else {
                throw new Error('Backend responded with error');
            }
        })
        .catch(error => {
            updateStatus('disconnected', 'Connection Failed');
            showNotification('❌ Connection failed. Check backend URL.', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
    }
    
    function checkBackendStatus() {
        const url = backendUrl.value.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '/health');
        
        fetch(url, { 
            method: 'GET',
            timeout: 3000 
        })
        .then(response => {
            if (response.ok) {
                updateStatus('connected', 'Backend Online');
            } else {
                updateStatus('disconnected', 'Backend Error');
            }
        })
        .catch(() => {
            updateStatus('disconnected', 'Backend Offline');
        });
    }
    
    function updateStatus(status, text) {
        statusDot.className = 'status-dot ' + status;
        statusText.textContent = text;
        
        // Update classes for styling
        if (status === 'connected') {
            statusDot.classList.add('connected');
        } else if (status === 'connecting') {
            statusDot.classList.add('connecting');
        }
    }
    
    function resetToDefaults() {
        if (confirm('Reset all settings to defaults?')) {
            const defaults = {
                enabled: true,
                backendUrl: 'ws://localhost:8000/ws',
                sensitivity: 'medium',
                trustThreshold: 70,
                showOverlay: true,
                notifications: true
            };
            
            chrome.storage.sync.set(defaults, function() {
                loadSettings();
                showNotification('✅ Settings reset to defaults', 'success');
            });
        }
    }
    
    function showLoading(show, text = 'Loading...') {
        if (show) {
            loadingOverlay.style.display = 'flex';
            loadingOverlay.querySelector('.loading-text').textContent = text;
        } else {
            loadingOverlay.style.display = 'none';
        }
    }
    
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            animation: slideDown 0.3s ease-out;
        `;
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideDown {
                from { opacity: 0; transform: translateX(-50%) translateY(-10px); }
                to { opacity: 1; transform: translateX(-50%) translateY(0); }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideDown 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    // Check backend status periodically
    setInterval(checkBackendStatus, 10000);
    
    // Handle messages from background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === 'BACKEND_STATUS') {
            if (message.connected) {
                updateStatus('connected', 'Backend Online');
            } else {
                updateStatus('disconnected', 'Backend Offline');
            }
        }
    });
    
    // Get current tab info
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const currentTab = tabs[0];
        if (currentTab && currentTab.url) {
            const supportedSites = [
                'meet.google.com',
                'zoom.us', 
                'teams.microsoft.com',
                'webex.com'
            ];
            
            const isSupported = supportedSites.some(site => currentTab.url.includes(site));
            
            if (isSupported) {
                // Show additional stats section for supported sites
                const statsSection = document.getElementById('statsSection');
                if (statsSection) {
                    statsSection.style.display = 'block';
                }
            }
        }
    });
});
