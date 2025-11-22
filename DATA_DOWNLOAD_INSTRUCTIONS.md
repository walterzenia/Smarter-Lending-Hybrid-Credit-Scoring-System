# Large Data Files - Download Instructions

**Note**: The following large data files (2.6GB+) are excluded from the GitHub repository to stay within size limits.

## Excluded Files

### Home Credit Dataset Files (2.5GB)

These files are from the [Home Credit Default Risk competition on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk):

1. `installments_payments.csv` (690 MB)
2. `credit_card_balance.csv` (405 MB)
3. `previous_application.csv` (386 MB)
4. `POS_CASH_balance.csv` (375 MB)
5. `bureau_balance.csv` (358 MB)
6. `bureau.csv` (162 MB)
7. `application_train.csv` (158 MB)
8. `application_test.csv` (25 MB)

### How to Get These Files

**Option 1: Download from Kaggle**

```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle credentials (create API token at kaggle.com/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle competitions download -c home-credit-default-risk

# Unzip files to data/ directory
unzip home-credit-default-risk.zip -d data/
```

**Option 2: Manual Download**

1. Go to: https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Click "Download All" (requires Kaggle account)
3. Extract files to `data/` directory

**Option 3: Contact Repository Owner**

- These files may be available via cloud storage link
- Check with the repository maintainer for access

## Included Files (In Repository)

The following essential files ARE included in the repository:

### Working Datasets

- ✅ `smoke_hybrid_features.csv` (58 MB) - Preprocessed training data
- ✅ `smoke_engineered.csv` (49 MB) - Engineered features
- ✅ `UCI_Credit_Card.csv` (3 MB) - Behavioral dataset
- ✅ `uci_hybrid_features.csv` (1 MB) - UCI engineered features

### Test Cases

- ✅ `test_traditional_high_risk.csv` - Traditional model test cases
- ✅ `test_behavioral_high_risk.csv` - Behavioral model test cases
- ✅ `test_hybrid_high_risk.csv` - Ensemble model test cases

### Models

- ✅ All model `.pkl` files (23 MB total) - Required for predictions

## Running the Application

**Good news!** The application can run with just the included files:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/loan-default-hybrid-system.git
cd loan-default-hybrid-system

# Setup environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1  # Windows
# source myenv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

The app will work with:

- ✅ Predictions (all models loaded)
- ✅ Model metrics (stored in pickle files)
- ✅ Feature importance
- ✅ Batch predictions with CSV upload

**Limited functionality without large files:**

- ⚠️ EDA page may have limited visualizations (uses smoke_hybrid_features.csv which IS included)
- ⚠️ Cannot retrain models (requires full dataset)

## For Model Training/Retraining

If you need to train models from scratch, you MUST download the large files:

```bash
# After downloading Home Credit files to data/:
python src/model_training.py
python src/train_ensemble_hybrid.py
```

## Alternative: Git LFS (Large File Storage)

If you want to include large files in GitHub, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/installments_payments.csv"
git lfs track "data/credit_card_balance.csv"
git lfs track "data/previous_application.csv"
git lfs track "data/POS_CASH_balance.csv"
git lfs track "data/bureau_balance.csv"
git lfs track "data/bureau.csv"
git lfs track "data/application_train.csv"
git lfs track "data/application_test.csv"

# Commit and push
git add .gitattributes
git add data/
git commit -m "Add large files with LFS"
git push
```

**Note**: GitHub free tier includes 1GB LFS storage, so this would exceed limits. Consider:

- GitHub LFS data packs ($5 for 50GB)
- Alternative storage (Google Drive, Dropbox, AWS S3)

## Repository Size Summary

**Without large files**: ~150 MB (GitHub-friendly ✅)  
**With large files**: ~2.7 GB (Requires Git LFS or external storage)

**Recommendation**: Keep large files excluded, provide Kaggle download link in README.
