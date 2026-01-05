# ============================================================================
# IMAGE RECOGNITION API - FASTAPI BACKEND
# Uses pre-trained model for inference only (no training)
# Handles authentication, predictions, and database operations
# ============================================================================

import os
import io
import base64
import time
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import psycopg2
from google.auth.transport import requests
from google.oauth2 import id_token

# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_CLIENT_ID = "104742617040-?.apps.googleusercontent.com"
JWT_SECRET = "your-secret-key-change-this-in-production"
JWT_ALGORITHM = "HS256"

DATABASE_URL = "postgresql://postgres:password@localhost:5432/ecommerce_db"
IMAGE_UPLOAD_DIR = Path("./uploaded_images")
PRETRAINED_MODEL_PATH = "./models/pretrained_model.h5"  # Download your pre-trained model

# Create directories
IMAGE_UPLOAD_DIR.mkdir(exist_ok=True)
Path("./models").mkdir(exist_ok=True)

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Image Recognition API",
    description="Pre-trained model for handwritten digit recognition with Google OAuth",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db():
    """Get database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class GoogleSignInRequest(BaseModel):
    token: str

class PredictionRequest(BaseModel):
    image_data: str  # Base64 encoded image

class UserResponse(BaseModel):
    user_id: int
    name: str
    email: str
    profile_picture_url: str
    total_predictions: int

class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_label: str
    confidence: float
    processing_time_ms: int

# ============================================================================
# PRE-TRAINED MODEL LOADING
# ============================================================================

class PreTrainedModel:
    _instance = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(PRETRAINED_MODEL_PATH):
            self.model = keras.models.load_model(PRETRAINED_MODEL_PATH)
            print(f"Pre-trained model loaded from {PRETRAINED_MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"Pre-trained model not found at {PRETRAINED_MODEL_PATH}\n"
                "Please download your pre-trained model and place it in ./models/pretrained_model.h5"
            )

    def predict(self, image_array: np.ndarray) -> tuple:
        """
        Predict label from image using pre-trained model
        Returns: (predicted_label, confidence)
        """
        # Normalize image
        image_array = image_array.astype("float32") / 255.0
        
        # Reshape for model (assume 28x28 input)
        image_array = image_array.reshape(1, 28, 28)
        
        # Predict
        predictions = self.model.predict(image_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Map index to label (0-9 for digits)
        predicted_label = str(predicted_idx)
        
        return predicted_label, confidence

ml_model = None

def load_model_on_startup():
    """Load model when app starts"""
    global ml_model
    try:
        ml_model = PreTrainedModel()
        return True
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def process_image(image_data: str) -> np.ndarray:
    """
    Convert base64 image to numpy array resized to 28x28
    """
    # Remove data:image/png;base64, prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert('L')
    image_array = np.array(image)
    
    # Invert colors if needed (drawing should be black on white)
    if np.mean(image_array) > 128:
        image_array = 255 - image_array
    
    # Resize to 28x28
    image = Image.fromarray(image_array)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    return np.array(image)

def generate_jwt_token(user_id: int) -> str:
    """Generate JWT token"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token: str) -> int:
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(authorization: Optional[str] = Header(None)) -> int:
    """Extract user_id from JWT token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    token = authorization.split(" ")[1]
    user_id = verify_jwt_token(token)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return user_id

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def get_or_create_user(google_id: str, email: str, name: str, picture_url: str, conn):
    """Get existing user or create new one"""
    cursor = conn.cursor()
    
    # Try to get existing user
    cursor.execute(
        "SELECT user_id FROM image_db.users WHERE google_id = %s",
        (google_id,)
    )
    result = cursor.fetchone()
    
    if result:
        user_id = result[0]
        # Update last login
        cursor.execute(
            "UPDATE image_db.users SET last_login = NOW() WHERE user_id = %s",
            (user_id,)
        )
        conn.commit()
    else:
        # Create new user
        cursor.execute(
            """INSERT INTO image_db.users 
               (google_id, email, name, profile_picture_url) 
               VALUES (%s, %s, %s, %s) 
               RETURNING user_id""",
            (google_id, email, name, picture_url)
        )
        user_id = cursor.fetchone()[0]
        conn.commit()
    
    cursor.close()
    return user_id

def get_user_data(user_id: int, conn) -> dict:
    """Get user data"""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT user_id, name, email, profile_picture_url, total_predictions 
           FROM image_db.users WHERE user_id = %s""",
        (user_id,)
    )
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return {
            "user_id": result[0],
            "name": result[1],
            "email": result[2],
            "profile_picture_url": result[3],
            "total_predictions": result[4],
        }
    return None

def save_prediction(user_id: int, predicted_label: str, confidence: float, 
                   processing_time: int, image_filename: str, image_path: str, conn) -> int:
    """Save prediction to database"""
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO image_db.predictions 
           (user_id, image_filename, image_path, predicted_label, confidence, processing_time_ms) 
           VALUES (%s, %s, %s, %s, %s, %s) 
           RETURNING prediction_id""",
        (user_id, image_filename, image_path, predicted_label, confidence, processing_time)
    )
    prediction_id = cursor.fetchone()[0]
    
    # Log activity
    cursor.execute(
        """INSERT INTO image_db.activity_logs 
           (user_id, activity_type, description) 
           VALUES (%s, %s, %s)""",
        (user_id, "upload_image", f"Predicted: {predicted_label} (confidence: {confidence:.4f})")
    )
    
    conn.commit()
    cursor.close()
    return prediction_id

def get_user_predictions(user_id: int, conn) -> list:
    """Get user prediction history"""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT prediction_id, predicted_label, confidence, processing_time_ms, created_at 
           FROM image_db.predictions 
           WHERE user_id = %s 
           ORDER BY created_at DESC 
           LIMIT 50""",
        (user_id,)
    )
    results = cursor.fetchall()
    cursor.close()
    
    predictions = []
    for row in results:
        predictions.append({
            "prediction_id": row[0],
            "predicted_label": row[1],
            "confidence": float(row[2]),
            "processing_time_ms": row[3],
            "created_at": row[4].isoformat(),
        })
    
    return predictions

# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.post("/api/auth/google-signin")
async def google_signin(request: GoogleSignInRequest, conn=Depends(get_db)):
    """
    Handle Google OAuth signin
    """
    try:
        # Verify token with Google
        idinfo = id_token.verify_oauth2_token(
            request.token, requests.Request(), GOOGLE_CLIENT_ID
        )
        
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")
        
        # Get or create user
        user_id = get_or_create_user(google_id, email, name, picture, conn)
        
        # Get user data
        user = get_user_data(user_id, conn)
        
        # Generate JWT token
        token = generate_jwt_token(user_id)
        
        # Log activity
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO image_db.activity_logs 
               (user_id, activity_type, description) 
               VALUES (%s, %s, %s)""",
            (user_id, "login", "User logged in")
        )
        conn.commit()
        cursor.close()
        
        return {
            "user": user,
            "token": token,
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid token: {str(e)}")

# ============================================================================
# ROUTES - PREDICTIONS
# ============================================================================

@app.post("/api/predict")
async def predict(request: PredictionRequest, user_id: int = Depends(get_current_user), conn=Depends(get_db)):
    """
    Predict label from image using pre-trained model
    """
    if ml_model is None or ml_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please upload your pre-trained model.")
    
    try:
        start_time = time.time()
        
        # Process image
        image_array = process_image(request.image_data)
        
        # Predict using pre-trained model
        predicted_label, confidence = ml_model.predict(image_array)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"user_{user_id}_{timestamp}.png"
        image_path = IMAGE_UPLOAD_DIR / image_filename
        
        # Convert array back to image and save
        img = Image.fromarray(image_array.astype('uint8'))
        img.save(image_path)
        
        # Save to database
        prediction_id = save_prediction(
            user_id, predicted_label, confidence, processing_time,
            image_filename, str(image_path), conn
        )
        
        return {
            "prediction_id": prediction_id,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "processing_time_ms": processing_time,
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# ============================================================================
# ROUTES - HISTORY
# ============================================================================

@app.get("/api/predictions/history")
async def get_history(user_id: int = Depends(get_current_user), conn=Depends(get_db)):
    """Get user prediction history"""
    predictions = get_user_predictions(user_id, conn)
    return {"predictions": predictions}

# ============================================================================
# ROUTES - STATISTICS
# ============================================================================

@app.get("/api/stats/system")
async def get_system_stats(conn=Depends(get_db)):
    """Get system-wide statistics"""
    cursor = conn.cursor()
    
    # Total users
    cursor.execute("SELECT COUNT(*) FROM image_db.users")
    total_users = cursor.fetchone()[0]
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM image_db.predictions")
    total_predictions = cursor.fetchone()[0]
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM image_db.predictions")
    avg_confidence = cursor.fetchone()[0] or 0
    
    cursor.close()
    
    return {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "avg_confidence": float(avg_confidence),
    }

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if (ml_model and ml_model.model) else "not_loaded"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": model_status,
    }

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Image Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
# Load model when module is imported
ml_model = None
try:
    ml_model = PreTrainedModel()
    print("✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠️ Model not found: {e}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )