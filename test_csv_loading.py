#!/usr/bin/env python3
"""
Test script to demonstrate loading goals from different file formats.
"""

from goal_classifier import load_test_data_from_file
import pandas as pd

def test_file_loading():
    """Test loading goals from different file formats."""
    print("📁 Testing File Loading Functionality")
    print("=" * 50)
    
    # Test loading from text file
    print("\n📝 Loading from test_goals.txt:")
    txt_goals = load_test_data_from_file("test_goals.txt")
    if txt_goals:
        print(f"✅ Successfully loaded {len(txt_goals)} goals from text file")
        for i, goal in enumerate(txt_goals[:5], 1):  # Show first 5
            print(f"  {i}. {goal}")
        if len(txt_goals) > 5:
            print(f"  ... and {len(txt_goals) - 5} more")
    else:
        print("❌ Failed to load from text file")
    
    # Test loading from CSV file
    print("\n📊 Loading from sample_goals.csv:")
    csv_goals = load_test_data_from_file("sample_goals.csv")
    if csv_goals:
        print(f"✅ Successfully loaded {len(csv_goals)} goals from CSV file")
        for i, goal in enumerate(csv_goals[:5], 1):  # Show first 5
            print(f"  {i}. {goal}")
        if len(csv_goals) > 5:
            print(f"  ... and {len(csv_goals) - 5} more")
    else:
        print("❌ Failed to load from CSV file")
    
    # Debug: Try reading CSV directly with pandas
    print("\n🔍 Debug: Reading CSV directly with pandas:")
    try:
        df = pd.read_csv("sample_goals.csv")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Try to get first column
        first_col = df.iloc[:, 0]
        print(f"First column type: {type(first_col)}")
        print(f"First column values: {first_col.tolist()}")
        
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
    
    # Test loading from non-existent file
    print("\n🚫 Testing non-existent file:")
    nonexistent_goals = load_test_data_from_file("nonexistent_file.txt")
    if not nonexistent_goals:
        print("✅ Correctly handled non-existent file")
    else:
        print("❌ Unexpectedly loaded goals from non-existent file")

if __name__ == "__main__":
    test_file_loading()
