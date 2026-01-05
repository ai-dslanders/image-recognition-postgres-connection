User draws digit image
       ↓
Sends to backend
       ↓
Backend:
  • Decodes image
  • Resizes to 28x28
  • Loads pre-trained model
  • Runs inference (~50ms)
  • Gets predicted_label + confidence
  • Saves to database
       ↓
Returns prediction + confidence
       ↓
User sees result + history



-- ============================================================================
-- Image Recognition Database Schema
-- Simplified design focusing on user predictions with pre-trained model
-- ============================================================================

-- ============================================================================
-- CREATE SCHEMA FOR IMAGE RECOGNITION
-- ============================================================================

CREATE SCHEMA image_db;

SET search_path TO image_db, public;

-- ============================================================================
-- TABLE 1: USERS - Store Google OAuth Users
-- ============================================================================

CREATE TABLE image_db.users (
  user_id SERIAL PRIMARY KEY,
  google_id VARCHAR(255) NOT NULL UNIQUE,
  email VARCHAR(100) NOT NULL UNIQUE,
  name VARCHAR(100) NOT NULL,
  profile_picture_url TEXT,
  total_predictions INT NOT NULL DEFAULT 0 CHECK (total_predictions >= 0),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_login TIMESTAMP
);

-- ============================================================================
-- TABLE 2: PREDICTIONS - Store User Image Predictions
-- ============================================================================

CREATE TABLE image_db.predictions (
  prediction_id SERIAL PRIMARY KEY,
  user_id INT NOT NULL REFERENCES image_db.users(user_id) ON DELETE CASCADE,
  image_filename VARCHAR(255) NOT NULL,
  image_path TEXT NOT NULL,
  predicted_label VARCHAR(100) NOT NULL,
  confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  processing_time_ms INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE 3: ACTIVITY_LOGS - Track User Activity
-- ============================================================================

CREATE TABLE image_db.activity_logs (
  log_id SERIAL PRIMARY KEY,
  user_id INT NOT NULL REFERENCES image_db.users(user_id) ON DELETE CASCADE,
  activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN ('login', 'upload_image', 'view_profile', 'view_history')),
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================================================

-- Users table indexes
CREATE INDEX idx_users_google_id ON image_db.users(google_id);
CREATE INDEX idx_users_email ON image_db.users(email);

-- Predictions table indexes
CREATE INDEX idx_predictions_user_id ON image_db.predictions(user_id);
CREATE INDEX idx_predictions_created_at ON image_db.predictions(created_at);
CREATE INDEX idx_predictions_user_date ON image_db.predictions(user_id, created_at);

-- Activity logs indexes
CREATE INDEX idx_activity_logs_user_id ON image_db.activity_logs(user_id);
CREATE INDEX idx_activity_logs_created_at ON image_db.activity_logs(created_at);

-- ============================================================================
-- CREATE TRIGGERS & FUNCTIONS
-- ============================================================================

-- Function to increment user prediction count after new prediction
CREATE OR REPLACE FUNCTION increment_user_predictions()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE image_db.users
  SET total_predictions = total_predictions + 1
  WHERE user_id = NEW.user_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to increment prediction count
CREATE TRIGGER trigger_increment_predictions
AFTER INSERT ON image_db.predictions
FOR EACH ROW
EXECUTE FUNCTION increment_user_predictions();

-- ============================================================================
-- ANALYTICS VIEWS
-- ============================================================================

-- View: User Statistics
CREATE VIEW image_db.user_statistics AS
SELECT 
  u.user_id,
  u.name,
  u.email,
  u.total_predictions,
  u.created_at,
  u.last_login,
  COUNT(DISTINCT p.prediction_id) as total_predictions_verified,
  COUNT(DISTINCT p.prediction_id) as predictions_count,
  AVG(p.confidence) as avg_confidence
FROM image_db.users u
LEFT JOIN image_db.predictions p ON u.user_id = p.user_id
GROUP BY u.user_id, u.name, u.email, u.total_predictions, u.created_at, u.last_login;

-- View: Daily Analytics
CREATE VIEW image_db.daily_analytics AS
SELECT 
  DATE(p.created_at) as prediction_date,
  COUNT(DISTINCT p.user_id) as unique_users,
  COUNT(DISTINCT p.prediction_id) as total_predictions,
  AVG(p.confidence) as avg_confidence,
  ROUND(AVG(p.processing_time_ms), 2) as avg_processing_time_ms
FROM image_db.predictions p
GROUP BY DATE(p.created_at);

-- View: Popular Labels
CREATE VIEW image_db.popular_labels AS
SELECT 
  predicted_label,
  COUNT(*) as total_count,
  ROUND(AVG(confidence), 4) as avg_confidence
FROM image_db.predictions
GROUP BY predicted_label
ORDER BY total_count DESC;

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert sample user
INSERT INTO image_db.users (google_id, email, name, profile_picture_url) VALUES
('123456789', 'test.user@gmail.com', 'Test User', 'https://example.com/profile.jpg');

-- ============================================================================
-- USEFUL QUERIES FOR DASHBOARD
-- ============================================================================

-- Query 1: Get user by Google ID (for login)
-- SELECT * FROM image_db.users WHERE google_id = :google_id;

-- Query 2: Get user predictions history
-- SELECT * FROM image_db.predictions WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 20;

-- Query 3: Get user activity logs
-- SELECT * FROM image_db.activity_logs WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 30;

-- Query 4: Get daily statistics
-- SELECT * FROM image_db.daily_analytics ORDER BY prediction_date DESC LIMIT 7;

-- Query 5: Get popular labels
-- SELECT * FROM image_db.popular_labels LIMIT 10;

-- Query 6: Overall system statistics
-- SELECT 
--   COUNT(DISTINCT user_id) as total_users,
--   COUNT(DISTINCT prediction_id) as total_predictions,
--   AVG(confidence) as avg_confidence
-- FROM image_db.predictions;

-- ============================================================================
-- END OF IMAGE RECOGNITION DATABASE SCHEMA
-- ============================================================================


Step 1: Go to Google Cloud Console

Open: https://console.cloud.google.com/
Sign in with your Google My Project 38745account
At the top, click "Select a Project"
Click "NEW PROJECT"
Name it: Digit Recognition
Click CREATE
Wait for it to create (takes ~1 minute)



Step 2: Enable Google+ API

In Google Cloud Console, search for: Google+ API
Click on it
Click ENABLE
Wait for it to enable



Step 3: Create OAuth Credentials

Go to APIs & Services → Credentials (left menu)
Click + CREATE CREDENTIALS (top)
Choose OAuth 2.0 Client ID
It asks "What type of application?" → Choose Web application
Name it: Digit Recognition Web Client



Step 4: Add Authorized URLs
Under Authorized JavaScript origins, add:
http://localhost:8080
Under Authorized redirect URIs, add:
http://localhost:8080/
Click CREATE



Step 5: Copy Your Client ID
A popup shows your credentials. Copy the Client ID (looks like):
123456789-abc123xyz.apps.googleusercontent.com



Step 6: Update Your Files
In script.js (line ~10):
Find this:
javascriptconst CONFIG = {
    API_BASE_URL: 'http://localhost:8000/api',
    GOOGLE_CLIENT_ID: 'YOUR_CLIENT_ID_HERE.apps.googleusercontent.com',
Replace YOUR_CLIENT_ID_HERE with your actual Client ID



Terminal 1: Backend
bash# Navigate to your project directory
cd your-project-folder

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
python main.py
```

**Expected output:**
```
Model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete

Terminal 2: Frontend
Open a NEW terminal (don't close the first one!)
bash# Navigate to your project directory
cd your-project-folder

# Start HTTP server
python -m http.server 8080
```

**Expected output:**
```
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/)
```

---

### **Step 3: Access the App**

Open your browser and go to:
```
http://localhost:8080










