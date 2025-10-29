import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class ResumeScreening:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nb = MultinomialNB()
        self.model = VotingClassifier(estimators=[('knn', self.knn), ('rf', self.rf), ('nb', self.nb)], voting='soft')
        
    def clean_resume(self, text):
        text = re.sub(r'http\S+\s*', ' ', text)
        text = re.sub(r'RT|cc', ' ', text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'@\S+', '  ', text)
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        df['cleaned_resume'] = df['Resume'].apply(self.clean_resume)
        
        X = self.vectorizer.fit_transform(df['cleaned_resume'])
        y = df['Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def predict(self, resume_text):
        cleaned = self.clean_resume(resume_text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)
        probabilities = self.model.predict_proba(vectorized)
        confidence = float(np.max(probabilities) * 100)
        
        # Calculate ranking score (1-10) based on text quality metrics
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        score = min(10, max(1, int((word_count / 100) + (char_count / 1000) + (confidence / 20))))
        
        return prediction[0], confidence, score
    
    def save_model(self, model_path='resume_model.pkl', vectorizer_path='vectorizer.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self, model_path='resume_model.pkl', vectorizer_path='vectorizer.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

if __name__ == "__main__":
    screener = ResumeScreening()
    # Train with your dataset
    screener.train('UpdatedResumeDataSet.csv')
    screener.save_model()
