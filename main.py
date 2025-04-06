import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class EmployeePerformanceAnalyzer:
    def __init__(self):
        self.model = None
        self.sia = SentimentIntensityAnalyzer()
        self.important_keywords = [
            'communication', 'deadline', 'growth', 'teamwork', 'leadership',
            'initiative', 'problem-solving', 'creativity', 'reliability',
            'innovation', 'collaboration', 'punctuality', 'attitude',
            'productivity', 'knowledge', 'quality', 'customer'
        ]
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.generator = pipeline('text2text-generation', model = 'google/flan-t5-base')

        # Create preprocessing pipeline
        self.preprocessor = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),  # Using TfidfVectorizer
            ('scaler', StandardScaler(with_mean=False))  # Scaling TF-IDF features
        ])

        self.scaler = self.preprocessor.named_steps['scaler']

        # Create feature extraction pipeline
        self.feature_extractor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['avg_rating', 'projects_completed', 'training_hours']),
                ('text', self.preprocessor, 'feedback')
            ])
        self.positive_phrases = [
            "excellent communication skills", "always meets deadlines",
            "shows great teamwork", "demonstrates leadership",
            "takes initiative", "solves problems creatively",
            "reliable team member", "innovative thinking",
            "collaborative approach", "punctual and disciplined"
        ]
        self.negative_phrases = [
            "poor communication", "misses deadlines",
            "doesn't work well in teams", "lacks leadership",
            "waits for instructions", "avoids difficult problems",
            "unreliable at times", "conventional thinking",
            "works in isolation", "often late to meetings"
        ]
    def preprocess_feedback(self, feedback):
        """Preprocess feedback text"""
        # Tokenize feedback
        tokens = word_tokenize(feedback.lower())

        # Remove stop words and punctuation
        filtered_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

        # Calculate keyword score
        keyword_mentions = {keyword: 0 for keyword in self.important_keywords}
        for token in filtered_tokens:
            if token in keyword_mentions:
                keyword_mentions[token] += 1

        keyword_score = sum(keyword_mentions.values())

        # Get sentiment score
        sentiment_score = self.sia.polarity_scores(feedback)['compound']

        # Extract positive and negative aspects (using a simple heuristic)
        positive_aspects = [phrase for phrase in self.positive_phrases if phrase in feedback]
        negative_aspects = [phrase for phrase in self.negative_phrases if phrase in feedback]

        return {
            'sentiment_score': sentiment_score,
            'keyword_score': keyword_score,
            'positive_aspects': positive_aspects,
            'negative_aspects': negative_aspects,
            'keyword_mentions': keyword_mentions
        }

    def prepare_data(self, data):
        """Prepare data for model training"""
        # Process text feedback to get sentiment scores
        data['sentiment_score'] = data['feedback'].apply(lambda x: self.sia.polarity_scores(x)['compound'])

        # Extract features using the pipeline
        X = self.feature_extractor.fit_transform(data)
        y = data['performance_score']

        return X, y

    def train_model(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train model
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {'model': self.model, 'mse': mse, 'r2': r2, 'test_actual': y_test, 'test_predicted': y_pred}

    # ... (Rest of the class methods remain largely the same, except for using the pipeline in prepare_data) ...
    def generate_performance_report(self, employee_data):
        # Process feedback
        feedback_analysis = self.preprocess_feedback(employee_data['feedback'])

        # Create features for prediction
        features = pd.DataFrame({
            'avg_rating': [employee_data['avg_rating']],
            'projects_completed': [employee_data['projects_completed']],
            'training_hours': [employee_data['training_hours']],
            'feedback': [employee_data['feedback']],
            'sentiment_score': [feedback_analysis['sentiment_score']],
            'keyword_score': [feedback_analysis['keyword_score']]
        })

        # Scale features
        features_scaled_num = self.feature_extractor.named_transformers_['num'].transform(features[['avg_rating', 'projects_completed', 'training_hours']])
        features_scaled_text = self.feature_extractor.named_transformers_['text'].transform(features[['feedback']])

        # Combine the scaled features
        features_scaled = np.concatenate([features_scaled_num, features_scaled_text.toarray()], axis=1)

        # Make prediction and assign it to predicted_score
        predicted_score = self.model.predict(features_scaled)[0]

        # Generate performance categories
        if predicted_score >= 4.5:
            performance_category = "Outstanding"
        elif predicted_score >= 3.5:
            performance_category = "Exceeds Expectations"
        elif predicted_score >= 2.5:
            performance_category = "Meets Expectations"
        elif predicted_score >= 1.5:
            performance_category = "Needs Improvement"
        else:
            performance_category = "Unsatisfactory"

        # --- LLM Integration ---
        import re  # Import re for regular expressions

        def filter_incomplete_sentences(text_list):
              filtered_text = []
              for text in text_list:
            # Allow sentences without a full stop at the end
                  text = text.strip()
                  if text:  # Check if text is not empty
                      if text[0].isupper():
                          filtered_text.append(text)
                      else:
                    # If doesn't start with uppercase, still consider if it's a valid phrase
                          if len(text.split()) > 2 and any(keyword in text for keyword in self.important_keywords):
                              filtered_text.append(text)
              return filtered_text
        def generate_unique_text(prompt, max_length=150, num_return_sequences=3):
              generated_texts = self.generator(
                  prompt,
                  max_length=max_length,
                  num_return_sequences=num_return_sequences,
                  num_beams=num_return_sequences, # Add num_beams to enable beam search
                  repetition_penalty=1.2 # Changed 'repition_penalty' to 'repetition_penalty'
              )
              unique_text = max(
                  generated_texts,
                  key=lambda x: len(set(x['generated_text'].split())) / len(x['generated_text'].split())
              )['generated_text']
              return unique_text

        # Generate strengths using LLM with repetition penalty and uniqueness check
        prompt = f"Based on this feedback: '{employee_data['feedback']}', what are the employee's top 3 strengths, specifically focusing on areas like communication, problem-solving, and leadership? Please provide each strength as a complete sentence."
        strengths = filter_incomplete_sentences(generate_unique_text(prompt).split(','))

        # Generate areas for improvement using LLM with repetition penalty and uniqueness check
        prompt = f"Based on this feedback: '{employee_data['feedback']}', what are 2 specific areas where the employee could improve? Please provide each area as a complete sentence with actionable suggestions."
        areas_for_improvement = filter_incomplete_sentences(generate_unique_text(prompt).split(','))

        # Generate recommendations using LLM with repetition penalty and uniqueness check
        prompt = f"Based on this feedback: '{employee_data['feedback']}', provide 3 concrete recommendations for the employee's professional development, tailored to the areas for improvement identified. Please write each recommendation as a complete sentence."
        recommendations = filter_incomplete_sentences(generate_unique_text(prompt).split(','))

        # ... (Rest of the report generation code) ...
        # --- End LLM Integration ---

        # Prepare keyword analysis
        keywords = feedback_analysis['keyword_mentions']

        # Feature importance for recommendations
        if self.model:
            feature_importance = dict(zip(
                ['Average Rating', 'Projects Completed', 'Training Hours',
                 'Feedback Sentiment', 'Keyword Relevance'],
                self.model.feature_importances_
            ))
        else:
            feature_importance = {}

        # Compile report
        report = {
            'employee_id': employee_data.get('employee_id', 'N/A'),
            'name': employee_data.get('name', 'N/A'),
            'metrics': {
                'average_rating': employee_data['avg_rating'],
                'projects_completed': employee_data['projects_completed'],
                'training_hours': employee_data['training_hours'],
                'feedback_sentiment': round(feedback_analysis['sentiment_score'], 2),
                'keyword_relevance': round(feedback_analysis['keyword_score'], 2)
            },
            'performance': {
                'predicted_score': round(predicted_score, 2), # Using the predicted_score variable
                'category': performance_category
            },
            'feedback_analysis': {
                'strengths': strengths,  # Using LLM generated strengths
                'areas_for_improvement': areas_for_improvement,  # Using LLM generated areas for improvement
                'keyword_mentions': keywords
            },
            'recommendations': recommendations,  # Using LLM generated recommendations
            'feature_importance': feature_importance
        }

        return report
    
    def generate_team_insights(self, team_data):
        """Generate insights for a team of employees"""
        team_df = pd.DataFrame(team_data)

        # Process all feedback
        team_df['feedback_processed'] = team_df['feedback'].apply(self.preprocess_feedback)
        team_df['sentiment_score'] = team_df['feedback_processed'].apply(lambda x: x['sentiment_score'])
        team_df['keyword_score'] = team_df['feedback_processed'].apply(lambda x: x['keyword_score'])

        # Prepare feature matrix for prediction
        X = self.feature_extractor.transform(team_df)


        # Make predictions for all team members
        team_df['predicted_score'] = self.model.predict(X)

        # Categorize performance
        def categorize(score):
            if score >= 4.5:
                return "Outstanding"
            elif score >= 3.5:
                return "Exceeds Expectations"
            elif score >= 2.5:
                return "Meets Expectations"
            elif score >= 1.5:
                return "Needs Improvement"
            else:
                return "Unsatisfactory"

        team_df['performance_category'] = team_df['predicted_score'].apply(categorize)

        # Generate insights
        insights = {
            'team_size': len(team_df),
            'performance_distribution': team_df['performance_category'].value_counts().to_dict(),
            'avg_performance_score': team_df['predicted_score'].mean(),
            'top_performers': team_df.nlargest(3, 'predicted_score')[['name', 'predicted_score', 'performance_category']].to_dict('records'),
            'needs_attention': team_df.nsmallest(3, 'predicted_score')[['name', 'predicted_score', 'performance_category']].to_dict('records'),
            'feedback_sentiment_analysis': {
                'avg_sentiment': team_df['sentiment_score'].mean(),
                'positive_feedback_ratio': (team_df['sentiment_score'] > 0).mean(),
                'negative_feedback_ratio': (team_df['sentiment_score'] < 0).mean(),
            },
            'training_insights': {
                'avg_training_hours': team_df['training_hours'].mean(),
                'training_correlation_with_performance': team_df[['training_hours', 'predicted_score']].corr().iloc[0, 1]
            },
            'project_insights': {
                'avg_projects_completed': team_df['projects_completed'].mean(),
                'project_correlation_with_performance': team_df[['projects_completed', 'predicted_score']].corr().iloc[0, 1]
            }
        }

        return insights

    
import nltk
nltk.download('punkt_tab')
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pymongo
from datetime import datetime
import traceback
from transformers import pipeline
from flask import Flask, request, jsonify
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_very_secret_key_for_hackathon')
@app.route('/api/performace',methods=['POST'])

def get_performance_report():
    employee_data = request.get_json()
    report = analyzer.generate_performance_report(employee_data)
    return jsonify(report)
if __name__ == "__main__":
    app.run(debug=True, port=5000)

# --- 3. MongoDB Configuration ---
try:
    # IMPORTANT: Use environment variables or a config file for sensitive info in production!
    MONGO_URI = os.environ.get('MONGO_URI',"mongodb://localhost:27017/passwordmanager")
    DB_NAME = "hackathon_feedback"
    COLLECTION_NAME = "feedback_data"

    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    feedback_collection = db[COLLECTION_NAME]
    # Test connection
    client.admin.command('ping')
    print("MongoDB connection successful.")
except pymongo.errors.ConnectionFailure as e:
    print(f"Error connecting to MongoDB: {e}")
    # Depending on severity, you might want to exit or disable DB features
    feedback_collection = None # Disable DB operations if connection fails
except Exception as e:
    print(f"An unexpected error occurred during MongoDB setup: {e}")
    feedback_collection = None

analyzer = EmployeePerformanceAnalyzer

@app.route('/', methods=['GET'])
def index():
    """Displays the feedback submission form."""
    return render_template('feedback_form.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handles form submission, calls ML model, stores data, shows results."""
    employee_name = request.form.get('employee_name', '').strip()
    feedback_text = request.form.get('feedback_text', '').strip()
    submitter_name = request.form.get('submitter_name', 'Anonymous').strip() # Optional

    if not employee_name or not feedback_text:
        flash("Employee name and feedback text cannot be empty.", "error")
        return redirect(url_for('index'))

    # --- Call the ML Model ---
    try:
        prediction_result = analyzer.predict(feedback_text)
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        print(traceback.format_exc()) # Log the full traceback
        flash("An error occurred during feedback analysis.", "error")
        # Store the raw feedback anyway, maybe with an error flag?
        prediction_result = {"error": "Prediction failed", "sentiment": "Error", "rating": 0.0}
        # Decide if you want to proceed with saving or redirect back

    # --- Prepare data for MongoDB ---
    feedback_record = {
        "employee_name": employee_name,
        "feedback_text": feedback_text,
        "submitter_name": submitter_name,
        "submitted_at": datetime.utcnow(),
        "analysis": prediction_result # Store the entire result from the ML model
    }

    # --- Store in MongoDB ---
    insert_id = None
    if feedback_collection is not None:
        try:
            result = feedback_collection.insert_one(feedback_record)
            insert_id = result.inserted_id
            print(f"Feedback stored in MongoDB with ID: {insert_id}")
            flash("Feedback submitted and analyzed successfully!", "success")
        except Exception as e:
            print(f"Error storing feedback in MongoDB: {e}")
            print(traceback.format_exc())
            flash("Feedback analyzed, but failed to store it in the database.", "warning")
    else:
         flash("Database connection is unavailable. Feedback analyzed but not stored.", "warning")


    # --- Display Results ---
    # Pass necessary data to the results template
    return render_template('result.html',
                           employee_name=employee_name,
                           feedback_text=feedback_text,
                           prediction=prediction_result,
                           db_id=str(insert_id) if insert_id else "Not Stored")
