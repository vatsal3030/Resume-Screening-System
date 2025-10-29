# Resume Screening System

AI-powered resume screening application with PDF upload support and multiple file processing.

## Features
- 📄 Single & Multiple PDF Resume Upload
- 🤖 ML-based Resume Categorization
- 🎨 Modern Drag & Drop Interface
- 🎓 Model Training with Custom Dataset
- 📊 Real-time Results Display

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (first time):
   - Prepare CSV with columns: `Resume`, `Category`
   - Use the web interface to upload and train

3. Run the application:
```bash
python app.py
```

4. Open browser: `http://localhost:5000`

## Usage

### Predict Resumes
1. Upload single or multiple PDF files
2. Click "Analyze Resumes"
3. View categorized results

### Train Model
1. Prepare CSV dataset with `Resume` and `Category` columns
2. Upload in "Train Model" section
3. Wait for training completion

## Dataset Format
CSV file with columns:
- `Resume`: Full resume text
- `Category`: Job category (e.g., "Data Science", "Web Developer", etc.)

## Model
- Algorithm: K-Nearest Neighbors with OneVsRest
- Vectorization: TF-IDF (1500 features)
- Text Preprocessing: URL removal, special character cleaning, lowercase conversion
