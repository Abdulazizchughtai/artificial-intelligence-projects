import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

class StudentPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        self.label_encoders = {}
        self.training_df = None
       
    def create_training_dataset(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'study_hours': np.random.randint(1, 25, n_samples),
            'previous_grade': np.random.randint(30, 100, n_samples),
            'attendance': np.random.randint(50, 100, n_samples),
            'extracurricular': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'family_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'internet_access': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
            'assignments_completed': np.random.randint(40, 100, n_samples),
        }
        df = pd.DataFrame(data)
        score = (df['study_hours'] * 2.2 +
                df['previous_grade'] * 0.45 +
                df['attendance'] * 0.35 +
                df['assignments_completed'] * 0.25 +
                (df['extracurricular'] == 'Yes') * 5 +
                (df['family_support'] == 'Yes') * 4 +
                (df['internet_access'] == 'Yes') * 3 +
                np.random.normal(0, 8, n_samples))
        df['final_grade'] = np.clip(score, 0, 100).astype(int)
        df['performance'] = (df['final_grade'] >= 60).astype(int)
        return df
   
    def preprocess_data(self, df):
        df_processed = df.copy()
        categorical_cols = ['extracurricular', 'parent_education', 'family_support', 'internet_access']
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        return df_processed
   
    def train_models(self):
        df = self.create_training_dataset(1000)
        self.training_df = df
        df_processed = self.preprocess_data(df)
        X = df_processed.drop(['final_grade', 'performance'], axis=1)
        y = df_processed['performance']
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_dict = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        results = []
        for name, model in models_dict.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get ROC curve data for each model
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': roc_auc,
                'roc_data': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            }
            results.append({'name': name, 'accuracy': accuracy, 'auc': roc_auc})
        
        best_model_info = max(results, key=lambda x: x['accuracy'])
        self.best_model = self.models[best_model_info['name']]['model']
        return results

    def predict(self, student_data):
        df = pd.DataFrame([student_data])
        df_processed = self.preprocess_data(df)
        X = df_processed[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        prediction = int(self.best_model.predict(X_scaled)[0])
        probability = self.best_model.predict_proba(X_scaled)[0].tolist()
        
        estimated_grade = self.estimate_grade(student_data)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'estimated_grade': estimated_grade,
            'letter_grade': self.get_letter_grade(estimated_grade),
            'risk_level': self.get_risk_level(estimated_grade),
            'recommendations': self.get_recommendations(student_data, estimated_grade),
            'impacts': self.calculate_impacts(student_data)
        }

    def estimate_grade(self, data):
        score = (data['study_hours'] * 2.2 +
                data['previous_grade'] * 0.45 +
                data['attendance'] * 0.35 +
                data['assignments_completed'] * 0.25 +
                (data['extracurricular'] == 'Yes') * 5 +
                (data['family_support'] == 'Yes') * 4 +
                (data['internet_access'] == 'Yes') * 3)
        return int(np.clip(score, 0, 100))

    def get_letter_grade(self, score):
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    def get_risk_level(self, score):
        if score >= 80: return "Low"
        if score >= 65: return "Medium"
        return "High"

    def get_recommendations(self, data, estimated_grade):
        recommendations = []
        if data['study_hours'] < 10: recommendations.append("Increase study hours to at least 10-15 hours per week")
        if data['attendance'] < 80: recommendations.append("Improve attendance - aim for 90% or higher")
        if data['assignments_completed'] < 85: recommendations.append("Complete more assignments - target 90%+ completion rate")
        if data['extracurricular'] == 'No': recommendations.append("Join extracurricular activities for holistic development")
        if data['internet_access'] == 'No': recommendations.append("Seek internet access for better learning resources")
        if data['family_support'] == 'No': recommendations.append("Engage family in academic journey for better support")
        if data['previous_grade'] < 70: recommendations.append("Focus on understanding fundamentals from previous semester")
        return recommendations

    def calculate_impacts(self, data):
        return {
            'Study Hours': data['study_hours'] * 2.2,
            'Previous Grade': data['previous_grade'] * 0.45,
            'Attendance': data['attendance'] * 0.35,
            'Assignments': data['assignments_completed'] * 0.25,
            'Extracurricular': 5 if data['extracurricular'] == 'Yes' else 0,
            'Family Support': 4 if data['family_support'] == 'Yes' else 0,
            'Internet Access': 3 if data['internet_access'] == 'Yes' else 0
        }
    
    def get_stats(self):
        if self.training_df is None:
            return None
        return {
            'total_students': len(self.training_df),
            'pass_rate': float(self.training_df['performance'].mean()),
            'avg_grade': float(self.training_df['final_grade'].mean()),
            'min_grade': int(self.training_df['final_grade'].min()),
            'max_grade': int(self.training_df['final_grade'].max()),
            'grade_dist': self.training_df['final_grade'].value_counts().sort_index().to_dict()
        }
