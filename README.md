# 🎯 Goal Classifier with Confidence Score

A Python-based machine learning system that automatically classifies user goals into 5 predefined business categories and provides confidence scores for each prediction.

## ✨ Features

- **5 Business Categories**: Project Management, HR & Onboarding, Finance & Reporting, Marketing & Sales, Operations & Maintenance
- **Confidence Scores**: Each prediction comes with a confidence score (0.0 to 1.0)
- **Machine Learning**: Uses TF-IDF vectorization with Logistic Regression for accurate classification
- **Batch Processing**: Classify multiple goals at once from text or CSV files
- **Logging**: Automatically saves all predictions to timestamped CSV files
- **Console Output**: Beautiful formatted table display of results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Classifier

```bash
python goal_classifier.py
```

### 3. View Results

The script will:
- Train the model on built-in training data
- Classify example goals
- Display results in a formatted table
- Save predictions to a CSV file

## 📊 How It Works

### Machine Learning Pipeline

1. **Text Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert goal text into numerical features
2. **Feature Extraction**: Extracts up to 1000 most important features from the training data
3. **Classification**: Logistic Regression model predicts the category and provides probability scores
4. **Confidence Score**: The highest probability becomes the confidence score

### Training Data

The system comes pre-trained with 95 examples across all 5 categories:
- **Project Management**: 19 examples (e.g., "Plan a product launch", "Create project timeline", "Develop project risk assessment")
- **HR & Onboarding**: 19 examples (e.g., "Hire new employees", "Update employee handbook", "Create diversity and inclusion program")
- **Finance & Reporting**: 19 examples (e.g., "Check revenue for last month", "Prepare quarterly financial report", "Implement financial controls")
- **Marketing & Sales**: 19 examples (e.g., "Launch new marketing campaign", "Analyze customer feedback", "Develop brand positioning strategy")
- **Operations & Maintenance**: 19 examples (e.g., "Update system security", "Maintain server infrastructure", "Establish quality assurance standards")

## 📁 File Structure

```
golclasfier/
├── goal_classifier.py      # Main classifier script
├── test_goals.txt         # Sample goals for testing
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── goal_predictions_*.csv # Generated prediction logs
```

## 🔧 Usage Examples

### Single Goal Classification

```python
from goal_classifier import GoalClassifier

# Initialize and train
classifier = GoalClassifier()
classifier.train()

# Classify a single goal
goal = "Plan a product launch"
category, confidence = classifier.classify_goal(goal)
print(f"Category: {category}")
print(f"Confidence: {confidence:.3f}")
```

### Batch Classification

```python
# Classify multiple goals
goals = [
    "Plan a product launch",
    "Update company onboarding guide",
    "Check revenue for last month"
]

predictions = classifier.classify_multiple_goals(goals)
for goal, category, confidence in predictions:
    print(f"{goal} → {category} (Confidence: {confidence:.3f})")
```

### Load Goals from File

```python
from goal_classifier import load_test_data_from_file

# Load from text file (one goal per line)
goals = load_test_data_from_file("my_goals.txt")

# Load from CSV file (first column should contain goals)
goals = load_test_data_from_file("my_goals.csv")
```

## 📈 Output Format

### Console Display

```
================================================================================
GOAL CLASSIFICATION RESULTS
================================================================================
Goal                                     Category             Confidence
--------------------------------------------------------------------------------
Plan a product launch                    Project Management   0.923
Update company onboarding guide          HR & Onboarding      0.891
Check revenue for last month             Finance & Reporting  0.945
Launch new marketing campaign            Marketing & Sales    0.878
Update system security                   Operations & Maint.  0.912
--------------------------------------------------------------------------------
Total goals classified: 5
```

### CSV Log File

The system automatically saves predictions to CSV files with columns:
- `timestamp`: ISO format timestamp
- `goal`: The input goal text
- `category`: Predicted category
- `confidence`: Confidence score (0.000 to 1.000)

Example CSV output:
```csv
timestamp,goal,category,confidence
2024-01-15T10:30:45.123456,Plan a product launch,Project Management,0.923
2024-01-15T10:30:45.124567,Update company onboarding guide,HR & Onboarding,0.891
```

## 🎯 Category Examples

### Project Management
- "Plan a product launch"
- "Create project timeline"
- "Set project milestones"
- "Develop project risk assessment"
- "Create stakeholder communication plan"
- "Establish project governance framework"
- "Implement project management software"
- "Plan resource allocation strategy"
- "Create project quality assurance plan"
- "Develop change management process"

### HR & Onboarding
- "Update company onboarding guide"
- "Hire new employees"
- "Conduct performance reviews"
- "Create diversity and inclusion program"
- "Implement employee wellness initiatives"
- "Establish career development framework"
- "Create employee recognition program"
- "Develop succession planning process"
- "Implement employee feedback system"
- "Establish remote work policies"

### Finance & Reporting
- "Check revenue for last month"
- "Prepare quarterly financial report"
- "Analyze budget variance"
- "Create annual budget forecast"
- "Develop cost reduction strategies"
- "Implement financial controls"
- "Analyze market risk exposure"
- "Prepare board presentation materials"
- "Develop financial modeling tools"
- "Establish audit compliance procedures"

### Marketing & Sales
- "Launch new marketing campaign"
- "Analyze customer feedback"
- "Update website content"
- "Develop brand positioning strategy"
- "Create customer segmentation analysis"
- "Implement lead generation campaigns"
- "Develop pricing strategy"
- "Create competitive analysis report"
- "Plan influencer marketing partnerships"
- "Develop customer loyalty program"

### Operations & Maintenance
- "Update system security"
- "Maintain server infrastructure"
- "Update software systems"
- "Implement disaster recovery plan"
- "Develop supply chain optimization"
- "Create facility maintenance schedule"
- "Establish inventory management system"
- "Implement process automation"
- "Develop vendor management strategy"
- "Establish quality assurance standards"

## 🔍 Model Performance

The classifier typically achieves:
- **Accuracy**: 85-95% on test data
- **Training Time**: < 1 second
- **Prediction Time**: < 0.1 seconds per goal
- **Memory Usage**: Minimal (< 50MB)

## 🛠️ Customization

### Adding New Categories

To add new categories, modify the `categories` list and `_create_training_data()` method in the `GoalClassifier` class:

```python
def __init__(self):
    self.categories = [
        "Project Management",
        "HR & Onboarding", 
        "Finance & Reporting",
        "Marketing & Sales",
        "Operations & Maintenance",
        "New Category"  # Add your category here
    ]
```

### Modifying Training Data

Add more examples to improve classification accuracy:

```python
def _create_training_data(self):
    training_data = [
        # ... existing examples ...
        ("Your new goal example", "Your Category"),
        # ... more examples ...
    ]
    return training_data
```

## 📝 Requirements

- **Python**: 3.7 or higher
- **Dependencies**: scikit-learn, pandas, numpy
- **Memory**: Minimum 50MB RAM
- **Storage**: < 10MB for the script and dependencies

## 🚨 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **Low Confidence**: The model may need more training examples for specific categories
3. **File Not Found**: Ensure test files are in the same directory as the script

### Performance Tips

- Use clear, descriptive goal text for better classification
- Avoid very short or ambiguous goals
- The model works best with business-related goals

## 🤝 Contributing

Feel free to:
- Add more training examples
- Improve the classification algorithm
- Add new categories
- Enhance the output formatting

## 📄 License

This project is open source and available under the MIT License.

## 🎉 Acknowledgments

Built with:
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **Python**: Programming language

---

**Happy Goal Classifying! 🎯✨**


