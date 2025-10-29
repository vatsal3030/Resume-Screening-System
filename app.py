from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import PyPDF2
import os
import pandas as pd
from io import BytesIO
from resume_screening import ResumeScreening

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

screener = ResumeScreening()

try:
    screener.load_model()
except:
    print("Model not found. Please train the model first.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        results = []
        files = request.files.getlist('resumes')
        
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and file.filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                resume_text = extract_text_from_pdf(filepath)
                category, confidence, score = screener.predict(resume_text)
                
                results.append({
                    'filename': file.filename,
                    'category': category,
                    'confidence': round(confidence, 2),
                    'score': score
                })
                
                os.remove(filepath)
        
        # Calculate statistics
        categories = [r['category'] for r in results]
        stats = {
            'total': len(results),
            'categories': {cat: categories.count(cat) for cat in set(categories)},
            'avg_confidence': round(sum(r['confidence'] for r in results) / len(results), 2) if results else 0,
            'avg_score': round(sum(r['score'] for r in results) / len(results), 2) if results else 0
        }
        
        return jsonify({'results': results, 'stats': stats})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.json['results']
        df = pd.DataFrame(data)
        df = df[['filename', 'category', 'confidence', 'score']]
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'resume_results.csv')
        df.to_csv(filepath, index=False)
        
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name='resume_results.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
