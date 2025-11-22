# Deployment Guide

**Loan Default Prediction System**  
**Date:** November 18, 2025  
**Version:** 2.0.0

---

## Table of Contents

1. [GitHub Setup & Collaboration](#github-setup--collaboration)
2. [Deployment Options](#deployment-options)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Heroku Deployment](#heroku-deployment)
5. [AWS/Azure Deployment](#awsazure-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Post-Deployment Testing](#post-deployment-testing)

---

## GitHub Setup & Collaboration

### Step 1: Create .gitignore (✅ Done)

The `.gitignore` file has been created to exclude:

- Virtual environments (myenv/)
- Python cache files (**pycache**/)
- IDE configurations
- Temporary files

**Note**: Large data files and model files are currently included. If you want to exclude them (recommended for GitHub), uncomment these lines in `.gitignore`:

```
# data/*.csv
# models/*.pkl
```

### Step 2: Initialize Git Repository

```bash
# Open PowerShell in project directory
cd "C:\Users\user\Desktop\Loan Default Hybrid System"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Loan Default Hybrid System v2.0.0"
```

### Step 3: Create GitHub Repositories

**For your account:**

```bash
# Create a new repository on GitHub.com
# Name: loan-default-hybrid-system
# Description: Machine learning system for loan default prediction using hybrid ensemble approach

# Link your local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/loan-default-hybrid-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**For co-collaborator's account:**

**Option A: Fork the repository**

1. Co-collaborator goes to your repository
2. Clicks "Fork" button
3. Creates a fork in their account

**Option B: Push to their repository directly**

```bash
# Add second remote
git remote add collaborator https://github.com/COLLABORATOR_USERNAME/loan-default-hybrid-system.git

# Push to both remotes
git push origin main
git push collaborator main
```

### Step 4: Add Co-Collaborator

**On your GitHub repository:**

1. Go to repository → Settings → Collaborators
2. Click "Add people"
3. Enter collaborator's GitHub username
4. Select "Write" or "Admin" permissions
5. Send invitation

**Collaborator setup:**

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/loan-default-hybrid-system.git
cd loan-default-hybrid-system

# Create virtual environment
python -m venv myenv

# Activate environment (Windows)
.\myenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Deployment Options

### Comparison Table

| Platform               | Cost          | Ease       | Best For    | Model Size Limit |
| ---------------------- | ------------- | ---------- | ----------- | ---------------- |
| **Streamlit Cloud**    | Free          | ⭐⭐⭐⭐⭐ | Quick demos | 1GB              |
| **Heroku**             | Free tier     | ⭐⭐⭐⭐   | Small apps  | 500MB slug       |
| **AWS EC2**            | Pay-as-you-go | ⭐⭐⭐     | Production  | Unlimited        |
| **Azure Web Apps**     | Pay-as-you-go | ⭐⭐⭐     | Enterprise  | Unlimited        |
| **Docker + Cloud Run** | Pay-as-you-go | ⭐⭐⭐⭐   | Scalability | Configurable     |

**Recommendation for your project**: **Streamlit Cloud** (easiest) or **Docker + Cloud Run** (production)

---

## Streamlit Cloud Deployment

### ✅ Easiest Option - FREE

**Prerequisites:**

- GitHub repository (created above)
- Streamlit Cloud account (free)

### Step 1: Prepare Files

**Create `requirements.txt` (if not exists):**

```bash
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
joblib>=1.3.0
```

**Verify `app.py` has correct paths:**

```python
# Use relative paths, not absolute
models_dir = Path("models")
data_dir = Path("data")
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Select**:
   - Repository: `YOUR_USERNAME/loan-default-hybrid-system`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click** "Deploy"

**Deployment time**: 5-10 minutes

**Your app URL**: `https://YOUR_USERNAME-loan-default-hybrid-system.streamlit.app`

### Step 3: Share with Co-Collaborator

Co-collaborator can deploy their own instance:

1. Fork your repository (or use their copy)
2. Follow same Streamlit Cloud steps
3. Deploy from their GitHub account

---

## Heroku Deployment

### Step 1: Create Required Files

**Create `Procfile`:**

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**Create `setup.sh`:**

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**Update `requirements.txt`:**

```
# Add these lines
gunicorn==21.2.0
```

### Step 2: Deploy to Heroku

```bash
# Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create loan-default-predictor

# Add Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

**Note**: Free tier has 512MB RAM limit. Your models (~9MB) should fit.

---

## AWS/Azure Deployment

### AWS EC2 Option

**Step 1: Launch EC2 Instance**

- Instance type: t2.medium (4GB RAM recommended)
- AMI: Ubuntu 22.04 LTS
- Security group: Open port 8501 (Streamlit default)

**Step 2: Setup Server**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Python
sudo apt update
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/loan-default-hybrid-system.git
cd loan-default-hybrid-system

# Setup environment
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

**Step 3: Keep Running (Use screen or systemd)**

```bash
# Using screen
screen -S streamlit
streamlit run app.py
# Press Ctrl+A, then D to detach
```

**Access**: http://your-ec2-ip:8501

### Azure Web Apps Option

```bash
# Install Azure CLI
# Download from: https://aka.ms/installazurecliwindows

# Login
az login

# Create resource group
az group create --name loan-default-rg --location eastus

# Create App Service plan
az appservice plan create --name loan-default-plan --resource-group loan-default-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group loan-default-rg --plan loan-default-plan --name loan-default-predictor --runtime "PYTHON|3.11"

# Deploy from GitHub
az webapp deployment source config --name loan-default-predictor --resource-group loan-default-rg --repo-url https://github.com/YOUR_USERNAME/loan-default-hybrid-system --branch main --manual-integration
```

---

## Docker Deployment

### Step 1: Create Dockerfile

**Create `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Create `.dockerignore`:**

```
myenv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
.vscode/
.idea/
```

### Step 2: Build and Run Locally

```bash
# Build image
docker build -t loan-default-predictor .

# Run container
docker run -p 8501:8501 loan-default-predictor

# Access at: http://localhost:8501
```

### Step 3: Deploy to Google Cloud Run

```bash
# Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/loan-default-predictor

# Deploy to Cloud Run
gcloud run deploy loan-default-predictor \
    --image gcr.io/YOUR_PROJECT_ID/loan-default-predictor \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

**Your app URL**: Provided after deployment (e.g., `https://loan-default-predictor-xxx.run.app`)

---

## Post-Deployment Testing

### Checklist

**Functionality Tests:**

- [ ] Home page loads correctly
- [ ] EDA page displays visualizations
- [ ] Prediction page accepts manual input
- [ ] CSV batch upload works
- [ ] All three models load successfully
- [ ] Feature importance displays SHAP plots
- [ ] Model metrics show ROC curves

**Performance Tests:**

- [ ] Page load time < 3 seconds
- [ ] Single prediction < 2 seconds
- [ ] Batch prediction (100 rows) < 10 seconds
- [ ] No memory errors with large batches

**Data Tests:**

- [ ] Upload valid CSV: Success
- [ ] Upload invalid CSV: Error message shown
- [ ] Missing features: Handled gracefully
- [ ] Extra features: Ignored correctly

### Monitoring

**Streamlit Cloud:**

- Built-in analytics at: https://share.streamlit.io/
- View logs, usage stats, errors

**Heroku:**

```bash
heroku logs --tail
```

**AWS/Azure:**

- Set up CloudWatch/Azure Monitor
- Create alerts for errors/high memory

---

## Troubleshooting

### Common Issues

**1. Model files too large for GitHub**

```bash
# Use Git LFS (Large File Storage)
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/*.pkl
git commit -m "Add model files with LFS"
```

**2. Memory errors on deployment**

- Reduce batch size in app.py
- Use smaller model files
- Upgrade to higher tier (more RAM)

**3. Missing dependencies**

```bash
# Regenerate requirements.txt
pip freeze > requirements.txt
```

**4. Streamlit Cloud build fails**

- Check Python version compatibility (use 3.11)
- Verify all file paths are relative
- Check requirements.txt has all packages

**5. Models don't load**

```python
# Add error handling in app.py
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
```

---

## Security Considerations

### Before Public Deployment

**1. Remove sensitive data**

- Check for API keys, passwords in code
- Use environment variables for secrets

**2. Add authentication (optional)**

```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'loan_default_app',
    'secret_key',
    cookie_expiry_days=30
)

name, authentication_status = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

**3. Rate limiting**

- Implement request limits per user
- Prevent abuse of prediction API

**4. Data privacy**

- Don't log user input data
- Comply with data protection regulations

---

## Cost Estimates

### Monthly Costs

**Free Tier:**

- Streamlit Cloud: $0 (1GB app, public only)
- Heroku Free: $0 (512MB RAM, sleeps after 30min)

**Paid Options:**

- AWS EC2 t2.medium: ~$30/month
- Azure B1 Basic: ~$13/month
- Google Cloud Run: Pay per request (~$5-20/month for moderate use)
- Heroku Hobby: $7/month (no sleep)

**Recommendation**: Start with **Streamlit Cloud (free)** for testing, upgrade to **Cloud Run** for production.

---

## Next Steps

### Quick Start (Streamlit Cloud)

1. ✅ Push to GitHub (see instructions above)
2. ✅ Go to https://share.streamlit.io/
3. ✅ Connect repository
4. ✅ Click Deploy
5. ✅ Share URL with users

**Estimated time**: 15 minutes

### For Production

1. ✅ Create Dockerfile
2. ✅ Test locally with Docker
3. ✅ Deploy to Cloud Run or AWS
4. ✅ Set up monitoring
5. ✅ Configure custom domain (optional)

**Estimated time**: 2-3 hours

---

## Support & Resources

**Streamlit Documentation**: https://docs.streamlit.io/  
**Heroku Python Guide**: https://devcenter.heroku.com/articles/python-support  
**AWS EC2 Tutorial**: https://docs.aws.amazon.com/ec2/  
**Google Cloud Run**: https://cloud.google.com/run/docs  
**Docker Documentation**: https://docs.docker.com/

---

**Ready to deploy!** Follow the Streamlit Cloud section for the quickest path to production.
