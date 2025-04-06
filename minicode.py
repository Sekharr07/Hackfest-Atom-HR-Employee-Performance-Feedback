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

from flask import Flask, request, jsonify
app = Flask(__name__)
analyzer = EmployeePerformanceAnalyzer()
@app.route('/api/performace',methods=['POST'])

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
        """Generate a detailed performance report for an employee"""
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
        # Use the preprocessor directly for 'feedback' and StandardScaler for numerical features
        features_scaled_num = self.feature_extractor.named_transformers_['num'].transform(features[['avg_rating', 'projects_completed', 'training_hours']])
        features_scaled_text = self.feature_extractor.named_transformers_['text'].transform(features[['feedback']])

        # Combine the scaled features
        features_scaled = np.concatenate([features_scaled_num, features_scaled_text.toarray()], axis=1)

        # Make prediction
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

        # Prepare feedback analysis
        strengths = feedback_analysis['positive_aspects']
        areas_for_improvement = feedback_analysis['negative_aspects']

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

        # Generate recommendations
        recommendations = []

        if feedback_analysis['sentiment_score'] < 0:
            recommendations.append("Address negative feedback from team members")

        if employee_data['training_hours'] < 20:
            recommendations.append("Increase participation in training programs")

        if employee_data['projects_completed'] < 3:
            recommendations.append("Assign to more projects to gain experience")

        if 'communication' not in keywords and 'teamwork' not in keywords:
            recommendations.append("Focus on improving communication and teamwork skills")

        if len(recommendations) == 0:
            recommendations.append("Continue current performance trajectory")

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
                'predicted_score': round(predicted_score, 2),
                'category': performance_category
            },
            'feedback_analysis': {
                'strengths': strengths,
                'areas_for_improvement': areas_for_improvement,
                'keyword_mentions': keywords
            },
            'recommendations': recommendations,
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

def get_performance_report():
    employee_data = request.get_json()
    report = analyzer.generate_performance_report(employee_data)
    return jsonify(report)
if __name__ == "__main__":
    app.run(debug=True, port=5000)