# MediScan AI – Disease Prediction & Health Assistant

> AI-powered health intelligence platform built with Next.js, Express.js, MongoDB, and Python ML.

## 🏗 Project Structure

```
/frontend    → Next.js (App Router) + Tailwind CSS + Framer Motion
/backend     → Node.js + Express.js + MongoDB (Mongoose)
/ml-service  → Python Flask + scikit-learn (Decision Tree)
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** v18+
- **Python** 3.8+
- **MongoDB** running locally (or MongoDB Atlas URI)

### 1. Backend Setup
```bash
cd backend
npm install
# Edit .env if needed (MongoDB URI, JWT secret)
npm run dev
# → Running on http://localhost:5001
```

### 2. ML Service Setup
```bash
cd ml-service
pip install -r requirements.txt
python train_model.py    # Train the model first!
python app.py
# → Running on http://localhost:5002
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# → Running on http://localhost:3000
```

## 🔑 Environment Variables

### Backend (`backend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5001` | Backend server port |
| `MONGO_URI` | `mongodb://127.0.0.1:27017/mediscan` | MongoDB connection string |
| `JWT_SECRET` | `mediscan_ai_super_secret_key_2026` | JWT signing secret |
| `ML_SERVICE_URL` | `http://localhost:5002` | Python ML service URL |

### Frontend (`frontend/.env.local`)
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:5001` | Backend API URL |

## 📱 Pages

### User Pages
| Page | Route | Description |
|------|-------|-------------|
| Landing | `/` | SaaS landing page with hero, features, pricing |
| Predict | `/predict` | Disease prediction with symptom input |
| Dashboard | `/dashboard` | User dashboard with stats & history |
| Chatbot | `/chatbot` | AI health assistant chat interface |
| Reports | `/reports` | Download prediction reports |
| Profile | `/profile` | User profile & settings |
| Login | `/auth/login` | JWT authentication |
| Register | `/auth/register` | New user registration |

### Admin Pages
| Page | Route | Description |
|------|-------|-------------|
| Overview | `/admin` | System stats & analytics |
| Diseases | `/admin/diseases` | CRUD disease management |
| Dataset | `/admin/dataset` | Upload CSV & train model |
| Logs | `/admin/logs` | Prediction audit trail |
| Users | `/admin/users` | User management & roles |
| Settings | `/admin/settings` | System configuration |

## 🔗 API Endpoints

### Auth
- `POST /api/auth/register` – Register new user
- `POST /api/auth/login` – Login
- `GET /api/auth/me` – Get current user profile

### Prediction
- `POST /api/predict` – Predict disease from symptoms
- `GET /api/symptoms` – Get all known symptoms
- `GET /api/history` – User's prediction history

### Admin
- `GET /api/admin/stats` – Dashboard statistics
- `GET/POST/DELETE /api/admin/diseases` – Disease CRUD
- `GET /api/admin/users` – User management
- `GET /api/admin/history` – All prediction logs

### ML Service
- `POST /predict` – ML prediction (port 5002)
- `GET /health` – Service health check
- `GET /model-info` – Model metadata

## ⚠️ Disclaimer

This AI tool is for educational purposes only. MediScan AI predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
