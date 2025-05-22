import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3

# Create output directory if it doesn't exist
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# Connect to the database using sqlite3
conn = sqlite3.connect('/Users/pup/Desktop/Arch/conversations.db')

# Function to execute SQL and return DataFrame
def run_query(query, conn):
    return pd.read_sql_query(query, conn)

print("Testing model usage analysis...")
model_query = "SELECT * FROM model_usage"
try:
    model_data = run_query(model_query, conn)
    
    print(f"Retrieved {len(model_data)} rows from model_usage view")
    print("Sample data:")
    print(model_data.head())
    
    if not model_data.empty:
        plt.figure(figsize=(12, 8))
        platforms = model_data['platform'].unique()
        
        for platform in platforms:
            platform_data = model_data[model_data['platform'] == platform]
            plt.bar(platform_data['model'], platform_data['message_count'], label=platform)
        
        plt.title('Message Count by Model')
        plt.xlabel('Model')
        plt.ylabel('Message Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_usage.png'))
        print(f"Saved model_usage.png")
        
    # Save to CSV
    model_data.to_csv(os.path.join(output_dir, 'model_usage.csv'), index=False)
    print(f"Saved model_usage.csv")
    
except Exception as e:
    print(f"Error processing model data: {e}")
