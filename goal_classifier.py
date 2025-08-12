#!/usr/bin/env python3
"""
Goal Classifier with Confidence Score
Classifies user goals into 5 predefined categories with confidence scores.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import datetime
import csv
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GoalClassifier:
    """Goal classifier that categorizes goals into predefined categories with confidence scores."""
    
    def __init__(self):
        self.categories = [
            "Project Management",
            "HR & Onboarding", 
            "Finance & Reporting",
            "Marketing & Sales",
            "Operations & Maintenance"
        ]
        
        self.training_data = self._create_training_data()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False
        
    def _create_training_data(self) -> List[Tuple[str, str]]:
        """Create training data with goals and their corresponding categories."""
        training_data = [
            # Project Management
            ("Plan a product launch", "Project Management"),
            ("Create project timeline", "Project Management"),
            ("Set project milestones", "Project Management"),
            ("Coordinate team meetings", "Project Management"),
            ("Track project progress", "Project Management"),
            ("Manage project budget", "Project Management"),
            ("Create project documentation", "Project Management"),
            ("Assign project tasks", "Project Management"),
            ("Review project deliverables", "Project Management"),
            ("Plan project kickoff", "Project Management"),
            ("Develop project risk assessment", "Project Management"),
            ("Create stakeholder communication plan", "Project Management"),
            ("Establish project governance framework", "Project Management"),
            ("Implement project management software", "Project Management"),
            ("Conduct project status reviews", "Project Management"),
            ("Plan resource allocation strategy", "Project Management"),
            ("Create project quality assurance plan", "Project Management"),
            ("Develop change management process", "Project Management"),
            ("Establish project success metrics", "Project Management"),
            ("Plan project closure activities", "Project Management"),
            
            # HR & Onboarding
            ("Update company onboarding guide", "HR & Onboarding"),
            ("Hire new employees", "HR & Onboarding"),
            ("Conduct performance reviews", "HR & Onboarding"),
            ("Update employee handbook", "HR & Onboarding"),
            ("Plan team building activities", "HR & Onboarding"),
            ("Process payroll", "HR & Onboarding"),
            ("Conduct exit interviews", "HR & Onboarding"),
            ("Update job descriptions", "HR & Onboarding"),
            ("Plan training sessions", "HR & Onboarding"),
            ("Review benefits packages", "HR & Onboarding"),
            ("Develop employee retention strategy", "HR & Onboarding"),
            ("Create diversity and inclusion program", "HR & Onboarding"),
            ("Implement employee wellness initiatives", "HR & Onboarding"),
            ("Establish career development framework", "HR & Onboarding"),
            ("Create employee recognition program", "HR & Onboarding"),
            ("Develop succession planning process", "HR & Onboarding"),
            ("Implement employee feedback system", "HR & Onboarding"),
            ("Create workplace safety protocols", "HR & Onboarding"),
            ("Establish remote work policies", "HR & Onboarding"),
            ("Develop compensation benchmarking", "HR & Onboarding"),
            
            # Finance & Reporting
            ("Check revenue for last month", "Finance & Reporting"),
            ("Prepare quarterly financial report", "Finance & Reporting"),
            ("Analyze budget variance", "Finance & Reporting"),
            ("Review expense reports", "Finance & Reporting"),
            ("Prepare tax documents", "Finance & Reporting"),
            ("Analyze cash flow", "Finance & Reporting"),
            ("Review profit margins", "Finance & Reporting"),
            ("Prepare balance sheet", "Finance & Reporting"),
            ("Analyze financial ratios", "Finance & Reporting"),
            ("Review investment portfolio", "Finance & Reporting"),
            ("Create annual budget forecast", "Finance & Reporting"),
            ("Develop cost reduction strategies", "Finance & Reporting"),
            ("Implement financial controls", "Finance & Reporting"),
            ("Analyze market risk exposure", "Finance & Reporting"),
            ("Prepare board presentation materials", "Finance & Reporting"),
            ("Review vendor payment terms", "Finance & Reporting"),
            ("Develop financial modeling tools", "Finance & Reporting"),
            ("Create investor relations materials", "Finance & Reporting"),
            ("Establish audit compliance procedures", "Finance & Reporting"),
            ("Analyze competitor financial performance", "Finance & Reporting"),
            
            # Marketing & Sales
            ("Launch new marketing campaign", "Marketing & Sales"),
            ("Analyze customer feedback", "Marketing & Sales"),
            ("Update website content", "Marketing & Sales"),
            ("Plan sales strategy", "Marketing & Sales"),
            ("Create marketing materials", "Marketing & Sales"),
            ("Analyze market trends", "Marketing & Sales"),
            ("Plan product promotion", "Marketing & Sales"),
            ("Review sales performance", "Marketing & Sales"),
            ("Update social media strategy", "Marketing & Sales"),
            ("Plan customer events", "Marketing & Sales"),
            ("Develop brand positioning strategy", "Marketing & Sales"),
            ("Create customer segmentation analysis", "Marketing & Sales"),
            ("Implement lead generation campaigns", "Marketing & Sales"),
            ("Develop pricing strategy", "Marketing & Sales"),
            ("Create competitive analysis report", "Marketing & Sales"),
            ("Plan influencer marketing partnerships", "Marketing & Sales"),
            ("Develop customer loyalty program", "Marketing & Sales"),
            ("Create sales training materials", "Marketing & Sales"),
            ("Implement CRM system", "Marketing & Sales"),
            ("Develop international market entry plan", "Marketing & Sales"),
            
            # Operations & Maintenance
            ("Update system security", "Operations & Maintenance"),
            ("Maintain server infrastructure", "Operations & Maintenance"),
            ("Update software systems", "Operations & Maintenance"),
            ("Monitor system performance", "Operations & Maintenance"),
            ("Backup data systems", "Operations & Maintenance"),
            ("Update IT policies", "Operations & Maintenance"),
            ("Maintain office equipment", "Operations & Maintenance"),
            ("Update safety protocols", "Operations & Maintenance"),
            ("Monitor quality control", "Operations & Maintenance"),
            ("Update operational procedures", "Operations & Maintenance"),
            ("Implement disaster recovery plan", "Operations & Maintenance"),
            ("Develop supply chain optimization", "Operations & Maintenance"),
            ("Create facility maintenance schedule", "Operations & Maintenance"),
            ("Establish inventory management system", "Operations & Maintenance"),
            ("Implement process automation", "Operations & Maintenance"),
            ("Develop vendor management strategy", "Operations & Maintenance"),
            ("Create operational efficiency metrics", "Operations & Maintenance"),
            ("Establish quality assurance standards", "Operations & Maintenance"),
            ("Implement lean manufacturing principles", "Operations & Maintenance"),
            ("Develop environmental compliance procedures", "Operations & Maintenance")
        ]
        return training_data
    
    def train(self):
        """Train the classifier using the training data."""
        print("Training the goal classifier...")
        
        # Prepare training data
        goals, labels = zip(*self.training_data)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            goals, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize the text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train the classifier
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed! Model accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.categories))
        
        self.is_trained = True
        return accuracy
    
    def classify_goal(self, goal: str) -> Tuple[str, float]:
        """
        Classify a single goal and return category with confidence score.
        
        Args:
            goal (str): The goal text to classify
            
        Returns:
            Tuple[str, float]: (category, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions!")
        
        # Vectorize the input goal
        goal_vectorized = self.vectorizer.transform([goal])
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(goal_vectorized)[0]
        
        # Get predicted category
        predicted_category = self.classifier.predict(goal_vectorized)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = max(probabilities)
        
        return predicted_category, confidence
    
    def classify_multiple_goals(self, goals: List[str]) -> List[Tuple[str, str, float]]:
        """
        Classify multiple goals at once.
        
        Args:
            goals (List[str]): List of goal texts to classify
            
        Returns:
            List[Tuple[str, str, float]]: List of (goal, category, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions!")
        
        results = []
        for goal in goals:
            category, confidence = self.classify_goal(goal)
            results.append((goal, category, confidence))
        
        return results
    
    def save_predictions_to_csv(self, predictions: List[Tuple[str, str, float]], 
                               filename: str = None):
        """
        Save predictions to a CSV file with timestamp.
        
        Args:
            predictions (List[Tuple[str, str, float]]): List of (goal, category, confidence)
            filename (str): Optional filename, defaults to timestamp-based name
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"goal_predictions_{timestamp}.csv"
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'goal', 'category', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write headers if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write predictions
            for goal, category, confidence in predictions:
                writer.writerow({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'goal': goal,
                    'category': category,
                    'confidence': f"{confidence:.3f}"
                })
        
        print(f"Predictions saved to {filename}")
        return filename

def load_test_data_from_file(filename: str) -> List[str]:
    """
    Load test goals from a text or CSV file.
    
    Args:
        filename (str): Path to the file containing test goals
        
    Returns:
        List[str]: List of goal texts
    """
    goals = []
    
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(filename)
            # Assume first column contains goals
            goals = df.iloc[:, 0].dropna().tolist()
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
    
    elif filename.endswith('.txt'):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                goals = [line.strip() for line in file if line.strip()]
        except Exception as e:
            print(f"Error reading text file: {e}")
            return []
    
    else:
        print("Unsupported file format. Please use .txt or .csv files.")
        return []
    
    return goals

def display_results_table(predictions: List[Tuple[str, str, float]]):
    """
    Display classification results in a formatted table.
    
    Args:
        predictions (List[Tuple[str, str, float]]): List of (goal, category, confidence)
    """
    print("\n" + "="*80)
    print("GOAL CLASSIFICATION RESULTS")
    print("="*80)
    print(f"{'Goal':<40} {'Category':<20} {'Confidence':<10}")
    print("-"*80)
    
    for goal, category, confidence in predictions:
        # Truncate long goals for display
        display_goal = goal[:37] + "..." if len(goal) > 40 else goal
        print(f"{display_goal:<40} {category:<20} {confidence:.3f}")
    
    print("-"*80)
    print(f"Total goals classified: {len(predictions)}")

def main():
    """Main function to run the goal classifier."""
    print("🎯 Goal Classifier with Confidence Score")
    print("="*50)
    
    # Initialize and train the classifier
    classifier = GoalClassifier()
    accuracy = classifier.train()
    
    print(f"\n✅ Model trained successfully with {accuracy:.2f} accuracy!")
    
    # Example single goal classification
    print("\n🔍 Example Single Goal Classification:")
    example_goal = "Plan a product launch"
    category, confidence = classifier.classify_goal(example_goal)
    print(f"Goal: {example_goal}")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.3f}")
    
    # Load and classify test data
    print("\n📊 Loading and classifying test data...")
    
    # Try to load from test file, otherwise use built-in examples
    test_goals = load_test_data_from_file("test_goals.txt")
    
    if not test_goals:
        print("No test file found, using built-in examples...")
        # Use some examples from training data for demonstration
        test_goals = [
            "Plan a product launch",
            "Update company onboarding guide", 
            "Check revenue for last month",
            "Launch new marketing campaign",
            "Update system security",
            "Create project timeline",
            "Hire new employees",
            "Prepare quarterly financial report",
            "Analyze customer feedback",
            "Maintain server infrastructure"
        ]
    
    # Classify all test goals
    predictions = classifier.classify_multiple_goals(test_goals)
    
    # Display results
    display_results_table(predictions)
    
    # Save predictions to CSV
    csv_filename = classifier.save_predictions_to_csv(predictions)
    
    print(f"\n💾 All predictions have been saved to: {csv_filename}")
    print("\n🎉 Goal classification completed successfully!")

if __name__ == "__main__":
    main()

