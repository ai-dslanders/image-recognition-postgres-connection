// ============================================================================
// DIGIT RECOGNITION AI - FRONTEND JAVASCRIPT
// Image file upload version (no canvas drawing)
// ============================================================================

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    API_BASE_URL: 'http://localhost:8001/api',
    GOOGLE_CLIENT_ID: '104742617040-?.apps.googleusercontent.com',
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

let state = {
    currentUser: null,
    isLoggedIn: false,
    currentPrediction: null,
    predictions: [],
    selectedImage: null,
};

// ============================================================================
// IMAGE UPLOADER CLASS
// ============================================================================

class ImageUploader {
    constructor(inputId, previewId) {
        this.fileInput = document.getElementById(inputId);
        this.preview = document.getElementById(previewId);
        this.selectedImage = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showNotification('Please select an image file', 'error');
            return;
        }

        // Read file and preview
        const reader = new FileReader();
        reader.onload = (event) => {
            this.selectedImage = event.target.result;
            this.displayPreview(this.selectedImage);
            showNotification('Image loaded successfully', 'success');
        };
        reader.readAsDataURL(file);
    }

    displayPreview(imageData) {
        const img = new Image();
        img.src = imageData;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '300px';
        img.style.borderRadius = '8px';
        img.style.margin = '10px 0';
        
        // Clear preview and add new image
        this.preview.innerHTML = '';
        this.preview.appendChild(img);
    }

    getImageData() {
        return this.selectedImage;
    }

    clear() {
        this.fileInput.value = '';
        this.selectedImage = null;
        this.preview.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">No image selected</p>';
    }
}

// ============================================================================
// GOOGLE OAUTH
// ============================================================================

function initGoogleSignIn() {
    google.accounts.id.initialize({
        client_id: CONFIG.GOOGLE_CLIENT_ID,
        callback: handleGoogleSignIn,
    });
    
    google.accounts.id.renderButton(
        document.getElementById('googleSignInBtn'),
        {
            theme: 'dark',
            size: 'large',
            text: 'signin_with'
        }
    );
}

async function handleGoogleSignIn(response) {
    const token = response.credential;
    
    try {
        const res = await fetch(`${CONFIG.API_BASE_URL}/auth/google-signin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token }),
        });

        if (!res.ok) throw new Error('Sign in failed');
        
        const data = await res.json();
        state.currentUser = data.user;
        state.isLoggedIn = true;
        localStorage.setItem('jwt_token', data.token);
        
        showNotification(`Welcome, ${data.user.name}!`, 'success');
        updateUI();
    } catch (error) {
        console.error('Sign in error:', error);
        showNotification('Sign in error. Please try again.', 'error');
    }
}

function logout() {
    state.isLoggedIn = false;
    state.currentUser = null;
    localStorage.removeItem('jwt_token');
    showNotification('Logged out', 'success');
    updateUI();
}

// ============================================================================
// PREDICTION API
// ============================================================================

async function makePrediction() {
    if (!state.isLoggedIn) {
        showNotification('Please sign in first', 'error');
        return;
    }

    const imageData = uploader.getImageData();
    if (!imageData) {
        showNotification('Please select an image first', 'error');
        return;
    }

    try {
        const token = localStorage.getItem('jwt_token');
        const res = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ image_data: imageData }),
        });

        if (!res.ok) throw new Error('Prediction failed');
        
        const prediction = await res.json();
        state.currentPrediction = prediction;
        
        displayPredictionResult(prediction);
        loadPredictionHistory();
        uploader.clear();
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Prediction failed. Please try again.', 'error');
    }
}

function displayPredictionResult(prediction) {
    const resultDiv = document.getElementById('predictionResult');
    if (!resultDiv) return;

    const confidencePercent = (prediction.confidence * 100).toFixed(2);
    resultDiv.innerHTML = `
        <div style="text-align: center; padding: 30px; background: rgba(99, 102, 241, 0.1); border-radius: 12px;">
            <h2 style="font-size: 48px; margin: 0; color: #6366f1;">
                ${prediction.predicted_label}
            </h2>
            <p style="font-size: 18px; margin: 10px 0; color: #999;">
                Confidence: <strong style="color: #fff;">${confidencePercent}%</strong>
            </p>
            <p style="font-size: 14px; color: #666;">
                Processing time: ${prediction.processing_time_ms}ms
            </p>
        </div>
    `;
}

// ============================================================================
// HISTORY
// ============================================================================

async function loadPredictionHistory() {
    if (!state.isLoggedIn) return;

    try {
        const token = localStorage.getItem('jwt_token');
        const res = await fetch(`${CONFIG.API_BASE_URL}/predictions/history`, {
            headers: { 'Authorization': `Bearer ${token}` },
        });

        if (!res.ok) throw new Error('Failed to load history');
        
        const data = await res.json();
        state.predictions = data.predictions || [];
        displayHistory();
    } catch (error) {
        console.error('History error:', error);
    }
}

function displayHistory() {
    const historyDiv = document.getElementById('historyContainer');
    if (!historyDiv) return;

    if (state.predictions.length === 0) {
        historyDiv.innerHTML = '<p style="text-align: center; color: #999;">No predictions yet</p>';
        return;
    }

    historyDiv.innerHTML = `
        <div style="max-height: 400px; overflow-y: auto;">
            ${state.predictions.map((pred, idx) => `
                <div style="padding: 12px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #6366f1; font-size: 18px;">${pred.predicted_label}</strong>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: #999;">
                            Confidence: ${(pred.confidence * 100).toFixed(1)}% | ${new Date(pred.created_at).toLocaleString()}
                        </p>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================================================
// ADMIN STATS
// ============================================================================

async function loadSystemStats() {
    try {
        const res = await fetch(`${CONFIG.API_BASE_URL}/stats/system`);
        if (!res.ok) throw new Error('Failed to load stats');
        
        const stats = await res.json();
        displaySystemStats(stats);
    } catch (error) {
        console.error('Stats error:', error);
    }
}

function displaySystemStats(stats) {
    const statsDiv = document.getElementById('systemStats');
    if (!statsDiv) return;

    statsDiv.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; padding: 20px 0;">
            <div style="background: rgba(99, 102, 241, 0.1); padding: 20px; border-radius: 8px; text-align: center;">
                <p style="color: #999; margin: 0; font-size: 12px;">Total Users</p>
                <h3 style="margin: 10px 0 0 0; color: #6366f1; font-size: 28px;">${stats.total_users}</h3>
            </div>
            <div style="background: rgba(99, 102, 241, 0.1); padding: 20px; border-radius: 8px; text-align: center;">
                <p style="color: #999; margin: 0; font-size: 12px;">Total Predictions</p>
                <h3 style="margin: 10px 0 0 0; color: #6366f1; font-size: 28px;">${stats.total_predictions}</h3>
            </div>
            <div style="background: rgba(99, 102, 241, 0.1); padding: 20px; border-radius: 8px; text-align: center;">
                <p style="color: #999; margin: 0; font-size: 12px;">Avg Confidence</p>
                <h3 style="margin: 10px 0 0 0; color: #6366f1; font-size: 28px;">${(stats.avg_confidence * 100).toFixed(1)}%</h3>
            </div>
        </div>
    `;
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateUI() {
    const loginBtn = document.getElementById('googleSignInBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const predictPage = document.getElementById('predictPage');
    const homePage = document.getElementById('homePage');
    const userInfo = document.getElementById('userInfo');

    if (state.isLoggedIn) {
        if (loginBtn) loginBtn.style.display = 'none';
        if (logoutBtn) logoutBtn.style.display = 'block';
        if (predictPage) predictPage.style.display = 'block';
        if (homePage) homePage.style.display = 'none';
        if (userInfo && state.currentUser) {
            userInfo.innerHTML = `
                ${state.currentUser.profile_picture_url ? `<img src="${state.currentUser.profile_picture_url}" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">` : ''}
                <span>${state.currentUser.name}</span>
            `;
        }
        loadPredictionHistory();
        loadSystemStats();
    } else {
        if (loginBtn) loginBtn.style.display = 'block';
        if (logoutBtn) logoutBtn.style.display = 'none';
        if (predictPage) predictPage.style.display = 'none';
        if (homePage) homePage.style.display = 'block';
        if (userInfo) userInfo.innerHTML = '';
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.remove(), 3000);
}

function showPage(pageName) {
    document.querySelectorAll('[id$="Page"]').forEach(el => el.style.display = 'none');
    const page = document.getElementById(pageName + 'Page');
    if (page) page.style.display = 'block';
}

// ============================================================================
// INITIALIZATION
// ============================================================================

let uploader;

document.addEventListener('DOMContentLoaded', () => {
    uploader = new ImageUploader('imageInput', 'imagePreview');
    
    // Load JWT token if exists
    const token = localStorage.getItem('jwt_token');
    if (token) {
        state.isLoggedIn = true;
        // Could verify token here
    }
    
    // Setup Google Sign In
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.onload = initGoogleSignIn;
    document.head.appendChild(script);
    
    updateUI();
});