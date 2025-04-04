import pandas as pd
from fuzzywuzzy import fuzz


def load_and_prepare_data():
    """Read and prepare the data from CSV files."""
    # Read CSV files with UTF-8 encoding
    consolidated_df = pd.read_csv(
        "excel_files/Akan Speech Emotion Dataset with Prosody Labels Consolidated.csv",
        encoding='utf-8',
        delimiter=','
    )
    emotions_df = pd.read_csv(
        "excel_files/Akan Speech Emotion Dataset cleaned.csv",
        encoding='utf-8',
        delimiter=','
    )

    # Print initial row counts
    print(f"Initial emotions_df row count: {len(emotions_df)}")
    print(f"Initial consolidated_df row count: {len(consolidated_df)}")

    # Drop rows in consolidated_df that are entirely NaN
    consolidated_df = consolidated_df.dropna(how='all')
    print(f"Consolidated DF after dropping all-NaN rows: {len(consolidated_df)}")

    # Convert columns for consistency
    consolidated_df['Movie ID'] = consolidated_df['Movie ID'].astype('int64')
    consolidated_df['Sentence No'] = consolidated_df['Sentence No'].astype('Int64')
    emotions_df['Sentence No'] = emotions_df['Sentence No'].astype('Int64')
    consolidated_df['Dialogue ID'] = consolidated_df['Dialogue ID'].astype('Int64')
    emotions_df['Dialogue ID'] = emotions_df['Dialogue ID'].astype('Int64')

    # Normalize key string columns
    key_string_columns = ['Movie Title', 'Utterance ID']
    for col in key_string_columns:
        consolidated_df[col] = consolidated_df[col].str.strip().str.lower()
        emotions_df[col] = emotions_df[col].str.strip().str.lower()

    return consolidated_df, emotions_df


def merge_data(consolidated_df, emotions_df):
    """Merge the emotions dataframe with the consolidated data using key columns: Movie ID, Movie Title, Dialogue ID, and Utterance ID."""
    # Create a subset of consolidated_df with only the columns we need for the merge
    consolidated_subset = consolidated_df[['Movie ID', 'Sentence No', 'Movie Title', 'Dialogue ID', 'Utterance ID', 'Consolidated']].copy()
    
    # Check for duplicates in consolidated_subset on the merge keys
    merge_keys = ['Movie ID', 'Sentence No', 'Movie Title', 'Dialogue ID', 'Utterance ID']
    dup_rows = consolidated_subset[consolidated_subset.duplicated(subset=merge_keys, keep=False)]

    # write duplicate rows to a CSV file for inspection
    dup_rows.to_csv("duplicate_rows.csv", index=False, encoding='utf-8')

    if len(dup_rows) > 0:
        print(f"Warning: Found {len(dup_rows)} duplicate rows in consolidated_df based on merge keys.")
        # Drop duplicates, keeping the first occurrence
        consolidated_subset = consolidated_subset.drop_duplicates(subset=merge_keys, keep='first')
    
    # Merge with left join to preserve all rows from emotions_df
    merged_emotions_df = emotions_df.merge(
        consolidated_subset,
        on=merge_keys,
        how='left'
    )
    
    # Verify the row count matches emotions_df
    if len(merged_emotions_df) != len(emotions_df):
        print(f"Warning: Row count mismatch after merge. Expected {len(emotions_df)}, got {len(merged_emotions_df)}")
    else:
        print(f"Merge successful. Merged dataframe has {len(merged_emotions_df)} rows.")
    
    return merged_emotions_df


def quality_check(merged_emotions_df):
    """Perform a quality check by counting matched and unmatched rows."""
    matched_rows = merged_emotions_df[merged_emotions_df['Consolidated'].notna()]
    unmatched_rows = merged_emotions_df[merged_emotions_df['Consolidated'].isna()]

    total_rows = len(merged_emotions_df)
    num_matched = len(matched_rows)
    num_unmatched = len(unmatched_rows)

    print(f"Total rows: {total_rows}")
    print(f"Matched rows (with Consolidated data): {num_matched} ({num_matched / total_rows:.2%})")
    print(f"Unmatched rows (without Consolidated data): {num_unmatched} ({num_unmatched / total_rows:.2%})")


def fuzzy_match_unmatched(merged_emotions_df, consolidated_df, title_threshold=85, utterance_threshold=85):
    """Attempt fuzzy matching for unmatched rows based on Movie Title and Utterance ID."""
    unmatched_indices = merged_emotions_df[merged_emotions_df['Consolidated'].isna()].index
    match_count = 0
    
    print(f"Attempting fuzzy matching for {len(unmatched_indices)} unmatched rows...")
    
    for idx in unmatched_indices:
        row = merged_emotions_df.loc[idx]
        # Filter candidates from consolidated_df with the same Movie ID
        candidates = consolidated_df[consolidated_df['Movie ID'] == row['Movie ID']]
        
        # If no candidates with same Movie ID, try all rows
        if len(candidates) == 0:
            candidates = consolidated_df
        
        best_score = 0
        best_candidate = None
        
        for _, candidate in candidates.iterrows():
            # Skip candidates without Consolidated value
            if pd.isna(candidate['Consolidated']) or candidate['Consolidated'] == '':
                continue
                
            title_score = fuzz.token_set_ratio(str(row['Movie Title']), str(candidate['Movie Title']))
            utterance_score = fuzz.token_set_ratio(str(row['Utterance ID']), str(candidate['Utterance ID']))
            dialogue_score = 100 if row['Dialogue ID'] == candidate['Dialogue ID'] else 0
            
            # Combined score with more weight on dialogue ID match
            avg_score = (title_score + utterance_score + dialogue_score) / 3
            
            if (title_score >= title_threshold and utterance_score >= utterance_threshold) and avg_score > best_score:
                best_score = avg_score
                best_candidate = candidate
        
        if best_candidate is not None:
            merged_emotions_df.at[idx, 'Consolidated'] = best_candidate['Consolidated']
            match_count += 1
    
    print(f"Fuzzy matching found matches for {match_count} previously unmatched rows.")
    return merged_emotions_df


def save_final_dataset(merged_emotions_df):
    """Save the final merged dataset with emotions and consolidated prosody labels."""
    # Make sure we have all the original columns from emotions_df plus the Consolidated column
   
    final_df = merged_emotions_df.copy()
    # Drop rows where Consolidated is NaN or empty
    final_df = final_df[final_df['Consolidated'].notna() & (final_df['Consolidated'].str.strip() != '') & (final_df['Consolidated'] != ' ')]
    
    # Remove any "Unnamed" columns
    unnamed_cols = [col for col in final_df.columns if 'Unnamed' in col]
    if unnamed_cols:
        print(f"Removing {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        final_df = final_df.drop(columns=unnamed_cols)
    
    # Save the dataset
    output_file = "excel_files/Akan Speech Emotion Dataset with Consolidated Prosody.csv"
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Final dataset saved to {output_file} with {len(final_df)} rows and {len(final_df.columns)} columns.")
    
    # Print all column names to verify
    print("Columns in final dataset:")
    for i, col in enumerate(final_df.columns):
        print(f"  {i+1}. {col}")
    
    # Save unmatched rows to a separate file for inspection
    unmatched_rows = final_df[final_df['Consolidated'].isna()]
    if len(unmatched_rows) > 0:
        # Also remove unnamed columns from unmatched rows
        if unnamed_cols:
            unmatched_rows = unmatched_rows.drop(columns=unnamed_cols)
        unmatched_rows.to_csv("unmatched_rows.csv", index=False, encoding='utf-8')
        print(f"Saved {len(unmatched_rows)} unmatched rows to 'unmatched_rows.csv' for inspection.")


def main():
    # Load and prepare the data
    consolidated_df, emotions_df = load_and_prepare_data()
    
    # Merge data
    merged_emotions_df = merge_data(consolidated_df, emotions_df)
    
    # Check quality of initial merge
    print("\nInitial merge quality:")
    quality_check(merged_emotions_df)
    
    # Apply fuzzy matching to update unmatched rows
    merged_emotions_df = fuzzy_match_unmatched(merged_emotions_df, consolidated_df)
    
    # Check quality after fuzzy matching
    print("\nAfter fuzzy matching:")
    quality_check(merged_emotions_df)
    
    # Save the final dataset
    save_final_dataset(merged_emotions_df)


if __name__ == '__main__':
    main()
