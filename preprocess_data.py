import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append('src')
from preprocessing.text_cleaner import CommitMessageCleaner

def load_data(filePath):
    print(f"loading data from {filePath} ...")
    df = pd.read_csv(filePath)
    print(f"loaded {len(df)} commits..")
    return df

def analyze_message_length(df, column='message'):
    """Analyze the distribution of message lengths"""
    print("==message length analysing==")
    length = df[column].str.len()
    word_count = df[column].str.split().str.len()

    print(f"Character length - Mean: {length.mean():.1f}, Median: {length.median():.1f}")
    print(f"Character length - Min: {length.min()}, Max: {length.max()}")
    print(f"Word count - Mean: {word_count.mean():.1f}, Median: {word_count.median():.1f}")
    print(f"Word count - Min: {word_count.min()}, Max: {word_count.max()}")
    print(f"95th percentile length: {length.quantile(0.95):.0f} characters")
    print(f"99th percentile length: {length.quantile(0.99):.0f} characters")
    return length, word_count

def time_aware_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print("=== time_aware data split ===")
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date').reset_index(drop = True)

    n = len(df_sorted)
    trained_end = int(n * train_ratio)
    val_end = int (n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:trained_end].copy()
    val_df = df_sorted.iloc[trained_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"Total commits: {n}")
    print(f"Train set: {len(train_df)} ({len(train_df)/n*100:.1f}%) - {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Bug-introducing: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"Val set: {len(val_df)} ({len(val_df)/n*100:.1f}%) - {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"  Bug-introducing: {val_df['label'].sum()} ({val_df['label'].mean()*100:.1f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df)/n*100:.1f}%) - {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"  Bug-introducing: {test_df['label'].sum()} ({test_df['label'].mean()*100:.1f}%)")

    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, output_dir = 'data/processed'):
    """Save the processed datasets"""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n=== Data Saved ===")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Test: {test_path}")

def generate_preprocessing_report(df, train_df, val_df, test_df, output_dir='data/processed'):
    """Generate a report summarizing the preprocessing"""
    report =[]
    report.append("=" * 80)
    report.append("PREPROCESSING REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    report.append("\n1. Original Dataset")
    report.append(f" total commits: {len(df)}")
    report.append(f"bug introducing: {df['label'].sum()}({df['label'].mean()*100:.2f})")
    report.append(f"Clean: {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.2f}%)")
    report.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    report.append("\n2. DATA SPLITS")
    report.append(f"Train: {len(train_df)} commits ({len(train_df)/len(df)*100:.1f}%)")
    report.append(f"Val: {len(val_df)} commits ({len(val_df)/len(df)*100:.1f}%)")
    report.append(f"Test: {len(test_df)} commits ({len(test_df)/len(df)*100:.1f}%)")

    report.append("\n3. MESSAGE STATISTICS (After Cleaning)")
    for name, dataset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        avg_len = dataset['clean_message'].str.len().mean()
        avg_words = dataset['clean_message'].str.split().str.len().mean()
        report.append(f"{name}: Avg {avg_len:.0f} chars, {avg_words:.0f} words")

    report.append("\n4. LABEL DISTRIBUTION")
    for name, dataset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        bug_pct = dataset['label'].mean() * 100
        report.append(f"   {name}: {dataset['label'].sum()} bugs ({bug_pct:.2f}%)")  

    report.append("\n" + "=" * 80)

    #save Report
    report_path = os.path.join(output_dir,'preprocessing_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n' .join(report))

    print('\n'.join(report))
    print(f"\nReport saved to: {report_path}")

def main():
    """Main preprocessing pipeline"""
    INPUT_FILE = 'data/labeled/dotnet_runtime_labeled.csv'
    OUTPUT_DIR = 'data/processed'

    print("=" * 80)
    print("JIT DEFECT PREDICTION - PREPROCESSING PIPELINE")
    print("=" * 80)
    df = load_data(INPUT_FILE)
    analyze_message_length(df, 'message')

    print("\n=== Cleaning Commit Messages ===")
    cleaner = CommitMessageCleaner()
    df = cleaner.clean_dataset(df, message_column='message')

    print("\n=== After Cleaning ===")
    analyze_message_length(df, 'clean_message')

    train_df, val_df, test_df = time_aware_split(df)

    save_processed_data(train_df, val_df, test_df, OUTPUT_DIR)
    generate_preprocessing_report(df, train_df, val_df, test_df, OUTPUT_DIR)

    print("\n Preprocessing complete!")
    print(f" Ready for model training with {len(df)} total commits")

if __name__ == "__main__":
    main()