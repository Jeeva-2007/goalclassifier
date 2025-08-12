#!/usr/bin/env python3
"""
Demo script for the Goal Classifier
Shows various ways to use the classifier and demonstrates its capabilities.
"""

from goal_classifier import GoalClassifier, load_test_data_from_file
import time

def demo_single_classification(classifier):
    """Demonstrate single goal classification."""
    print("\n🔍 Single Goal Classification Demo")
    print("-" * 40)
    
    # Example goals to classify
    example_goals = [
        "Plan a product launch",
        "Update company onboarding guide",
        "Check revenue for last month",
        "Launch new marketing campaign",
        "Update system security",
        "Develop project risk assessment",
        "Create diversity and inclusion program",
        "Implement financial controls",
        "Develop brand positioning strategy",
        "Establish quality assurance standards"
    ]
    
    for goal in example_goals:
        category, confidence = classifier.classify_goal(goal)
        print(f"Goal: {goal}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.3f}")
        print("-" * 40)

def demo_batch_classification(classifier):
    """Demonstrate batch classification."""
    print("\n📊 Batch Classification Demo")
    print("-" * 40)
    
    # Load goals from test file
    goals = load_test_data_from_file("test_goals.txt")
    
    if goals:
        print(f"Loaded {len(goals)} goals from test_goals.txt")
        
        # Classify all goals at once
        start_time = time.time()
        predictions = classifier.classify_multiple_goals(goals)
        end_time = time.time()
        
        print(f"Classification completed in {end_time - start_time:.3f} seconds")
        print(f"Average time per goal: {(end_time - start_time) / len(goals):.3f} seconds")
        
        # Display results
        print("\nResults:")
        for i, (goal, category, confidence) in enumerate(predictions, 1):
            print(f"{i:2d}. {goal:<35} → {category:<20} ({confidence:.3f})")
        
        return predictions
    else:
        print("No test goals found. Using built-in examples...")
        goals = [
            "Plan a product launch",
            "Update company onboarding guide",
            "Check revenue for last month"
        ]
        predictions = classifier.classify_multiple_goals(goals)
        return predictions

def demo_custom_goals(classifier):
    """Demonstrate classification of custom goals."""
    print("\n✏️  Custom Goals Classification Demo")
    print("-" * 40)
    print("Enter your own goals to classify (type 'done' when finished):")
    
    custom_goals = []
    while True:
        goal = input("\nEnter a goal: ").strip()
        if goal.lower() == 'done':
            break
        if goal:
            custom_goals.append(goal)
    
    if custom_goals:
        print(f"\nClassifying {len(custom_goals)} custom goals...")
        predictions = classifier.classify_multiple_goals(custom_goals)
        
        print("\nResults:")
        for goal, category, confidence in predictions:
            print(f"'{goal}' → {category} (Confidence: {confidence:.3f})")
        
        # Save to CSV
        csv_filename = classifier.save_predictions_to_csv(predictions, "custom_goals_predictions.csv")
        print(f"\nCustom predictions saved to: {csv_filename}")
        
        return predictions
    else:
        print("No custom goals entered.")
        return []

def demo_performance_metrics(classifier):
    """Demonstrate model performance metrics."""
    print("\n📈 Performance Metrics Demo")
    print("-" * 40)
    
    # Test with known examples from training data
    test_goals = [
        "Plan a product launch",
        "Hire new employees", 
        "Check revenue for last month",
        "Launch new marketing campaign",
        "Update system security"
    ]
    
    expected_categories = [
        "Project Management",
        "HR & Onboarding",
        "Finance & Reporting", 
        "Marketing & Sales",
        "Operations & Maintenance"
    ]
    
    correct_predictions = 0
    total_confidence = 0
    
    print("Testing with known examples:")
    for goal, expected in zip(test_goals, expected_categories):
        predicted, confidence = classifier.classify_goal(goal)
        is_correct = predicted == expected
        if is_correct:
            correct_predictions += 1
        total_confidence += confidence
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {goal:<30} → {predicted:<20} (Expected: {expected})")
    
    accuracy = correct_predictions / len(test_goals)
    avg_confidence = total_confidence / len(test_goals)
    
    print(f"\nAccuracy: {accuracy:.1%} ({correct_predictions}/{len(test_goals)})")
    print(f"Average Confidence: {avg_confidence:.3f}")

def main():
    """Main demo function."""
    print("🎯 Goal Classifier Demo")
    print("=" * 50)
    
    # Initialize and train the classifier
    print("Initializing and training the classifier...")
    classifier = GoalClassifier()
    accuracy = classifier.train()
    
    print(f"\n✅ Model ready! Training accuracy: {accuracy:.2f}")
    
    # Run various demos
    demo_single_classification(classifier)
    demo_batch_classification(classifier)
    demo_performance_metrics(classifier)
    
    # Ask if user wants to try custom goals
    try_custom = input("\nWould you like to try custom goals? (y/n): ").lower().strip()
    if try_custom in ['y', 'yes']:
        demo_custom_goals(classifier)
    
    print("\n🎉 Demo completed! Check the generated CSV files for prediction logs.")
    print("\nTo run the full classifier, use: python goal_classifier.py")

if __name__ == "__main__":
    main()

