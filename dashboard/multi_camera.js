// ===== API Base URL =====
const API_BASE = window.location.origin;

// ===== Configuration =====
const CONFIG = {
    cameras: [
        { id: 1, name: 'Camera 1 - Main Gate', rtspUrl: '', enabled: true, status: 'inactive', fps: 0, vehicleCount: { in: 0, out: 0 } },
        { id: 2, name: 'Camera 2 - Exit Gate', rtspUrl: '', enabled: true, status: 'inactive', fps: 0, vehicleCount: { in: 0, out: 0 } },
        { id: 3, name: 'Camera 3 - Parking Entry', rtspUrl: '', enabled: true, status: 'inactive', fps: 0, vehicleCount: { in: 0, out: 0 } },
        { id: 4, name: 'Camera 4 - Parking Exit', rtspUrl: '', enabled: true, status: 'inactive', fps: 0, vehicleCount: { in: 0, out: 0 } }
    ],
    globalSettings: {
        confidenceThreshold: 25,
        processingRate: 2,
        autoRestart: true,
        enableOCR: true
    }
};

// ===== State Management =====
let currentView = 'dashboard';
let activeCameras = 0;
let totalVehicles = 0;

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    // Check Auth first
    checkAuth().then(authenticated => {
        if (authenticated) {
            document.getElementById('login-modal').style.display = 'none';
            loadConfiguration();
            initializeDashboard();
            initializeCameraGrid();
            initializeConfiguration();
            setupEventListeners();
            startDataPolling();
        } else {
            // Show login modal (already visible by default)
        }
    });
});

// ===== Authentication =====
async function checkAuth() {
    try {
        const response = await fetch(`${API_BASE}/api/check-auth`);
        return response.ok;
    } catch (e) {
        return false;
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const errorDiv = document.getElementById('login-error');

    try {
        const response = await fetch(`${API_BASE}/api/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('login-modal').style.display = 'none';
            showNotification('Login successful', 'success');
            // Initialize app
            loadConfiguration();
            initializeDashboard();
            initializeCameraGrid();
            initializeConfiguration();
            setupEventListeners();
            startDataPolling();
        } else {
            errorDiv.textContent = data.message || 'Login failed';
        }
    } catch (error) {
        errorDiv.textContent = 'Connection error';
    }
}

async function logout() {
    await fetch(`${API_BASE}/api/logout`, { method: 'POST' });
    location.reload();
}

// ===== View Switching =====
function switchView(view) {
    currentView = view;
    document.querySelectorAll('.view-container').forEach(v => v.style.display = 'none');
    document.querySelectorAll('.sidebar nav a').forEach(a => a.classList.remove('active'));

    switch (view) {
        case 'dashboard':
            document.getElementById('dashboard-view').style.display = 'block';
            document.getElementById('nav-dashboard').classList.add('active');
            document.getElementById('page-title').textContent = 'Multi-Camera Dashboard';
            updateDashboard();
            break;
        case 'cameras':
            document.getElementById('cameras-view').style.display = 'block';
            document.getElementById('nav-cameras').classList.add('active');
            document.getElementById('page-title').textContent = 'Camera Grid View';
            break;
        case 'config':
            document.getElementById('config-view').style.display = 'block';
            document.getElementById('nav-config').classList.add('active');
            document.getElementById('page-title').textContent = 'Camera Configuration';
            renderConfiguration();
            break;
        case 'history':
            document.getElementById('history-view').style.display = 'block';
            document.getElementById('nav-history').classList.add('active');
            document.getElementById('page-title').textContent = 'Detection History';
            loadHistory();
            break;
        case 'analytics':
            document.getElementById('analytics-view').style.display = 'block';
            document.getElementById('nav-analytics').classList.add('active');
            document.getElementById('page-title').textContent = 'Analytics & Reports';

            // Set default dates if empty
            if (!document.getElementById('ana-filter-start-date').value) {
                const today = new Date().toISOString().split('T')[0];
                document.getElementById('ana-filter-start-date').value = today;
                document.getElementById('ana-filter-end-date').value = today;
            }

            loadAnalytics();
            break;
    }
}

function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
}

// ===== Dashboard Functions =====
function initializeDashboard() {
    renderCameraStatusCards();
    updateDashboard();
}

function renderCameraStatusCards() {
    const grid = document.getElementById('camera-status-grid');
    grid.innerHTML = '';

    CONFIG.cameras.forEach(camera => {
        const card = document.createElement('div');
        card.className = 'camera-status-card';
        card.innerHTML = `
            <div class="camera-header">
                <div class="camera-name">
                    <i class="fas fa-video"></i> ${camera.name}
                </div>
                <span class="camera-status ${camera.status}">${camera.status.toUpperCase()}</span>
            </div>
            <div class="camera-info">
                <div class="info-item">
                    <span class="info-label">IN</span>
                    <span class="info-value" id="cam${camera.id}-in">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">OUT</span>
                    <span class="info-value" id="cam${camera.id}-out">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">FPS</span>
                    <span class="info-value" id="cam${camera.id}-fps">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Status</span>
                    <span class="info-value" style="font-size: 12px;">${camera.enabled ? 'Enabled' : 'Disabled'}</span>
                </div>
            </div>
            <div class="camera-controls">
                <button class="camera-btn start" onclick="startCamera(${camera.id})">
                    <i class="fas fa-play"></i> Start
                </button>
                <button class="camera-btn stop" onclick="stopCamera(${camera.id})">
                    <i class="fas fa-stop"></i> Stop
                </button>
                <button class="camera-btn config" onclick="configureCamera(${camera.id})">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function updateDashboard() {
    await fetchStats();
    await loadRecentActivity();
}

async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        let totalIn = 0;
        let totalOut = 0;

        // Get totals
        data.summary.forEach(item => {
            if (item.direction === 'IN') totalIn = item.count;
            if (item.direction === 'OUT') totalOut = item.count;
        });

        // Update per-camera counts
        data.by_camera.forEach(item => {
            const camId = item.camera_id;
            if (camId) {
                const inEl = document.getElementById(`cam${camId}-in`);
                const outEl = document.getElementById(`cam${camId}-out`);

                if (inEl && item.direction === 'IN') inEl.textContent = item.count;
                if (outEl && item.direction === 'OUT') outEl.textContent = item.count;
            }
        });

        // Update main stats
        document.getElementById('stat-total-in').textContent = totalIn;
        document.getElementById('stat-total-out').textContent = totalOut;
        document.getElementById('stat-active-cameras').textContent = data.active_cameras || 0;

        // Calculate average FPS
        let avgFps = 0;
        let fpsCount = 0;
        Object.values(data.camera_stats || {}).forEach(stats => {
            if (stats.fps > 0) {
                avgFps += stats.fps;
                fpsCount++;
                document.getElementById(`cam${stats.camera_id || fpsCount}-fps`).textContent = stats.fps.toFixed(1);
            }
        });
        avgFps = fpsCount > 0 ? (avgFps / fpsCount).toFixed(1) : 0;
        document.getElementById('stat-avg-fps').textContent = avgFps;

        // Update header
        document.getElementById('active-cameras').textContent = `${data.active_cameras || 0}/4`;
        document.getElementById('total-vehicles').textContent = totalIn + totalOut;

        // AUTO-SYNC: Check for status changes to show/hide feeds automatically
        Object.values(data.camera_stats || {}).forEach(stats => {
            const camId = stats.camera_id;
            const status = stats.status;

            const feedContainer = document.querySelector(`#camera-feed-${camId} .camera-feed-content`);
            if (feedContainer) {
                const hasImg = feedContainer.querySelector('img');

                if (status === 'active' && !hasImg) {
                    console.log(`Auto-starting feed for Cam ${camId}`);
                    feedContainer.innerHTML = `<img src="${API_BASE}/api/camera/${camId}/feed" style="width: 100%; height: 100%; object-fit: cover;">`;
                } else if (status === 'inactive' && hasImg) {
                    console.log(`Auto-stopping feed for Cam ${camId}`);
                    feedContainer.innerHTML = `
                        <div class="camera-placeholder">
                            <i class="fas fa-video-slash"></i>
                            <p>Camera Offline</p>
                        </div>
                    `;
                }
            }
        });

    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

async function loadRecentActivity() {
    const tbody = document.getElementById('recent-activity-body');

    try {
        const response = await fetch(`${API_BASE}/api/logs?limit=10`);
        const logs = await response.json();

        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: var(--text-muted);">No recent activity</td></tr>';
            return;
        }

        tbody.innerHTML = logs.map(item => {
            const cameraName = `Camera ${item.camera_id || 'Unknown'}`;
            const time = new Date(item.timestamp).toLocaleString();
            return `
                <tr>
                    <td><i class="fas fa-video"></i> ${cameraName}</td>
                    <td style="text-transform: capitalize;">${item.vehicle_type || 'Unknown'}</td>
                    <td><span class="plate-badge">${item.plate_number || 'N/A'}</span></td>
                    <td><span class="state-badge">${item.vehicle_state || '-'}</span></td>
                    <td><span class="badge ${item.direction.toLowerCase()}">${item.direction}</span></td>
                    <td>${time}</td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading recent activity:', error);
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: var(--text-muted);">Error loading data</td></tr>';
    }
}

// ===== Camera Grid Functions =====
function initializeCameraGrid() {
    const grid = document.getElementById('camera-grid');
    grid.innerHTML = '';

    CONFIG.cameras.forEach(camera => {
        const feed = document.createElement('div');
        feed.className = 'camera-feed';
        feed.id = `camera-feed-${camera.id}`;
        feed.innerHTML = `
            <div class="camera-feed-header">
                <div class="camera-feed-title">${camera.name}</div>
                <div class="camera-feed-fps">
                    <i class="fas fa-tachometer-alt"></i> <span id="fps-${camera.id}">0</span> FPS
                </div>
            </div>
            <div class="camera-feed-content">
                <div class="camera-placeholder">
                    <i class="fas fa-video-slash"></i>
                    <p>Camera Offline</p>
                </div>
            </div>
        `;
        grid.appendChild(feed);
    });
}

async function startCamera(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/api/camera/${cameraId}/start`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification(`Camera ${cameraId} started successfully`, 'success');

            // Update the feed container to show live stream
            const feedContent = document.querySelector(`#camera-feed-${cameraId} .camera-feed-content`);
            if (feedContent) {
                feedContent.innerHTML = `<img src="${API_BASE}/api/camera/${cameraId}/feed" style="width: 100%; height: 100%; object-fit: cover;">`;
            }

            await updateDashboard();
        } else {
            showNotification(data.message || 'Failed to start camera', 'error');
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        showNotification('Error starting camera', 'error');
    }
}

async function stopCamera(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/api/camera/${cameraId}/stop`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification(`Camera ${cameraId} stopped`, 'info');

            // Update the feed container to show offline placeholder
            const feedContent = document.querySelector(`#camera-feed-${cameraId} .camera-feed-content`);
            if (feedContent) {
                feedContent.innerHTML = `
                    <div class="camera-placeholder">
                        <i class="fas fa-video-slash"></i>
                        <p>Camera Offline</p>
                    </div>
                `;
            }

            await updateDashboard();
        } else {
            showNotification(data.message || 'Failed to stop camera', 'error');
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
        showNotification('Error stopping camera', 'error');
    }
}

function startAllCameras() {
    CONFIG.cameras.forEach(camera => {
        if (camera.enabled) startCamera(camera.id);
    });
}

function stopAllCameras() {
    CONFIG.cameras.forEach(camera => stopCamera(camera.id));
}

function configureCamera(cameraId) {
    switchView('config');
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// ===== Configuration Functions =====
function initializeConfiguration() {
    renderConfiguration();
    setupConfigListeners();
}

function renderConfiguration() {
    const grid = document.getElementById('camera-config-grid');
    grid.innerHTML = '';

    CONFIG.cameras.forEach(camera => {
        const card = document.createElement('div');
        card.className = 'camera-config-card';
        card.id = `config-camera-${camera.id}`;
        card.innerHTML = `
            <h3><i class="fas fa-video"></i> Camera ${camera.id}</h3>
            <div class="config-form">
                <div class="form-group">
                    <label>Camera Name</label>
                    <input type="text" id="name-${camera.id}" value="${camera.name}" placeholder="Enter camera name">
                </div>
                <div class="form-group">
                    <label>RTSP URL</label>
                    <input type="text" id="rtsp-${camera.id}" value="${camera.rtspUrl}" placeholder="rtsp://username:password@ip:port/stream">
                </div>
                <div class="form-group">
                    <label>Status</label>
                    <select id="enabled-${camera.id}">
                        <option value="true" ${camera.enabled ? 'selected' : ''}>Enabled</option>
                        <option value="false" ${!camera.enabled ? 'selected' : ''}>Disabled</option>
                    </select>
                </div>
                <div class="form-group">
                    <button onclick="testCamera(${camera.id})" class="control-btn" style="width: 100%;">
                        <i class="fas fa-flask"></i> Test Connection
                    </button>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });

    document.getElementById('confidence-threshold').value = CONFIG.globalSettings.confidenceThreshold;
    document.getElementById('confidence-value').textContent = CONFIG.globalSettings.confidenceThreshold + '%';
    document.getElementById('processing-rate').value = CONFIG.globalSettings.processingRate;
    document.getElementById('auto-restart').checked = CONFIG.globalSettings.autoRestart;
    document.getElementById('enable-ocr').checked = CONFIG.globalSettings.enableOCR;
}

function setupConfigListeners() {
    const slider = document.getElementById('confidence-threshold');
    if (slider) {
        slider.addEventListener('input', (e) => {
            document.getElementById('confidence-value').textContent = e.target.value + '%';
        });
    }
}

async function testCamera(cameraId) {
    const rtspUrl = document.getElementById(`rtsp-${cameraId}`).value;

    try {
        showNotification(`Testing camera ${cameraId}...`, 'info');
        const response = await fetch(`${API_BASE}/api/test-camera`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rtsp_url: rtspUrl })
        });
        const data = await response.json();

        if (data.success) {
            showNotification(`Camera ${cameraId} connection successful!`, 'success');
        } else {
            showNotification(data.message || 'Connection failed', 'error');
        }
    } catch (error) {
        console.error('Error testing camera:', error);
        showNotification('Error testing camera', 'error');
    }
}

function saveConfiguration() {
    CONFIG.cameras.forEach(camera => {
        camera.name = document.getElementById(`name-${camera.id}`).value;
        camera.rtspUrl = document.getElementById(`rtsp-${camera.id}`).value;
        camera.enabled = document.getElementById(`enabled-${camera.id}`).value === 'true';
    });

    CONFIG.globalSettings.confidenceThreshold = parseInt(document.getElementById('confidence-threshold').value);
    CONFIG.globalSettings.processingRate = parseInt(document.getElementById('processing-rate').value);
    CONFIG.globalSettings.autoRestart = document.getElementById('auto-restart').checked;
    CONFIG.globalSettings.enableOCR = document.getElementById('enable-ocr').checked;

    localStorage.setItem('cameraConfig', JSON.stringify(CONFIG));
    showNotification('Configuration saved successfully!', 'success');
    updateDashboard();
}

async function loadConfiguration() {
    try {
        const response = await fetch(`${API_BASE}/api/cameras`);
        const cameras = await response.json();

        if (Array.isArray(cameras)) {
            cameras.forEach(cam => {
                const target = CONFIG.cameras.find(c => c.id === cam.id);
                if (target) {
                    target.name = cam.name;
                    target.rtspUrl = cam.rtsp_url;
                    target.enabled = cam.enabled === 1;
                    target.status = cam.status || 'inactive';
                }
            });
            renderConfiguration();
            initializeCameraGrid();
        }
    } catch (e) {
        console.error("Error loading configuration from DB:", e);
    }
}

// ===== History Functions =====
function loadHistory() {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('filter-start-date').value = today;
    document.getElementById('filter-end-date').value = today;
    applyFilters();
}

async function applyFilters() {
    const camera = document.getElementById('filter-camera').value;
    const type = document.getElementById('filter-type').value;
    const plate = document.getElementById('filter-plate').value;
    const startDate = document.getElementById('filter-start-date').value;
    const endDate = document.getElementById('filter-end-date').value;

    const tbody = document.getElementById('history-table-body');

    try {
        const params = new URLSearchParams();
        if (camera !== 'all') params.append('camera_id', camera);
        if (type !== 'all') params.append('vehicle_type', type);
        if (plate) params.append('plate_number', plate);
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        params.append('limit', '100');

        const response = await fetch(`${API_BASE}/api/logs?${params}`);
        const logs = await response.json();

        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px; color: var(--text-muted);">No records found</td></tr>';
            return;
        }

        tbody.innerHTML = logs.map(item => `
            <tr>
                <td>#${item.id}</td>
                <td><i class="fas fa-video"></i> Camera ${item.camera_id || 'Unknown'}</td>
                <td style="text-transform: capitalize;">${item.vehicle_type || 'Unknown'}</td>
                <td><span class="plate-badge">${item.plate_number || 'N/A'}</span></td>
                <td><span class="state-badge">${item.vehicle_state || '-'}</span></td>
                <td><span class="badge ${item.direction.toLowerCase()}">${item.direction}</span></td>
                <td>${item.timestamp}</td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error applying filters:', error);
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px; color: var(--text-muted);">Error loading data</td></tr>';
    }
}

function exportToCSV() {
    showNotification('Exporting to CSV...', 'info');
}

function exportToPDF() {
    showNotification('Exporting to PDF...', 'info');
}

async function loadAnalytics() {
    showNotification('Loading analytics...', 'info');

    try {
        const camera = document.getElementById('ana-filter-camera').value;
        const type = document.getElementById('ana-filter-type').value;
        const dir = document.getElementById('ana-filter-dir').value;
        const start = document.getElementById('ana-filter-start-date').value;
        const end = document.getElementById('ana-filter-end-date').value;

        const params = new URLSearchParams();
        if (camera !== 'all') params.append('camera_id', camera);
        if (type !== 'all') params.append('vehicle_type', type);
        if (dir !== 'all') params.append('direction', dir);
        if (start) params.append('start_date', start);
        if (end) params.append('end_date', end);

        const response = await fetch(`${API_BASE}/api/analytics/hourly?${params}`);
        const data = await response.json();

        // Update Stats Summary Card
        const grid = document.querySelector('.analytics-grid');
        let statsCard = document.getElementById('detailed-stats-card');
        if (!statsCard) {
            statsCard = document.createElement('div');
            statsCard.id = 'detailed-stats-card';
            statsCard.className = 'chart-card';
            grid.insertBefore(statsCard, grid.firstChild);
        }

        const stats = data.stats || {};
        const peakHour = stats.peak_hour ? `${stats.peak_hour}:00` : 'N/A';
        const lowHour = stats.low_hour ? `${stats.low_hour}:00` : 'N/A';

        statsCard.innerHTML = `
            <h3>Traffic Highlights</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div class="mini-stat" style="background: var(--bg-main); padding: 15px; border-radius: 8px;">
                    <div style="color: var(--accent); font-size: 14px; margin-bottom: 5px;">Peak Traffic Hour</div>
                    <div style="font-size: 24px; font-weight: bold;">${peakHour}</div>
                    <div style="font-size: 12px; color: var(--text-muted);">${stats.peak_count || 0} vehicles</div>
                </div>
                <div class="mini-stat" style="background: var(--bg-main); padding: 15px; border-radius: 8px;">
                    <div style="color: var(--success); font-size: 14px; margin-bottom: 5px;">Lowest Traffic Hour</div>
                    <div style="font-size: 24px; font-weight: bold;">${lowHour}</div>
                    <div style="font-size: 12px; color: var(--text-muted);">${stats.low_count || 0} vehicles</div>
                </div>
            </div>
        `;

        // Note: Chart updates would go here if charts are implemented.
        showNotification('Analytics updated', 'success');

    } catch (e) {
        console.error("Analytics error", e);
        showNotification('Failed to update analytics', 'error');
    }
}

// ===== Data Polling =====
function startDataPolling() {
    setInterval(async () => {
        if (currentView === 'dashboard') {
            await updateDashboard();
        } else if (currentView === 'cameras') {
            await fetchStats(); // Refresh statuses and triggers feeds in Grid mode
        }
    }, 2000); // 2 seconds for faster sync
}

function refreshData() {
    showNotification('Refreshing data...', 'info');
    switch (currentView) {
        case 'dashboard':
            updateDashboard();
            break;
        case 'history':
            applyFilters();
            break;
    }
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);

    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--info)'};
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        max-width: 400px;
    `;

    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}" 
               style="color: ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--info)'}; font-size: 20px;"></i>
            <span style="color: var(--text-primary); font-weight: 500;">${message}</span>
        </div>
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);

function setupEventListeners() {
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            document.body.classList.remove('sidebar-open');
        }
    });
}

window.cameraAPI = {
    startCamera,
    stopCamera,
    startAllCameras,
    stopAllCameras,
    saveConfiguration,
    getConfig: () => CONFIG
};
