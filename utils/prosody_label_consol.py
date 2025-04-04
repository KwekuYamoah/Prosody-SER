import pandas as pd
import re
from collections import Counter

# read the excel file, utf-8 encoding
prosody_df = pd.read_csv("excel_files/Akan Speech Emotion Dataset with Prosody Labels.csv", encoding='utf-8')

# subset columns "Prosodic Prominence Annotation - Annotator A", "Prosodic Prominence Annotation - Annotator B", "Prosodic Prominence Annotation - Annotator C"
annotate_df = prosody_df[["Prosodic Prominence Annotation - Annotator A", 
                          "Prosodic Prominence Annotation - Annotator B", 
                          "Prosodic Prominence Annotation - Annotator C"]].copy()

# rename the columns
annotate_df.columns = ["Annotator A", "Annotator B", "Annotator C"]

# check if columns contain any NaN values
# print(annotate_df.isnull().sum())
# print rows that have NaN values
# print(annotate_df[annotate_df.isnull().any(axis=1)])

# check out the first few rows of the dataframe
# print(annotate_df.head())




def consolidate_annotations(row):
    """
    Consolidates prosodic prominence annotations from multiple annotators into a single string.
    For each token (word) in the row, the consolidated prosodic prominence is determined as follows:
    - If at least two annotators mark a word as prominent (1), the consolidated label is 1
    - If there's disagreement (e.g., one says 1 and others say 0), the consolidated label is 0
    - Uses all available annotations (1, 2, or 3) for each word
    - Skips words with no annotations
    
    Parameters:
    row (pd.Series): A row from the dataframe containing annotations from Annotators A, B, and C.
    
    Returns:
    str: A string containing the consolidated annotations for each token.
    """
    # More flexible pattern to handle varying space formatting in annotations
    pattern = r'([^ (]+)\(\s*(\d+)\s*\)(\S*)'
    annotators = ['Annotator A', 'Annotator B', 'Annotator C']
    
    # Get tokens from each annotator
    tokens = {}
    for annotator in annotators:
        if pd.notnull(row[annotator]):
            tokens[annotator] = re.findall(pattern, str(row[annotator]))
        else:
            tokens[annotator] = []
    
    # If no annotations were found but there is data, try a more flexible pattern
    total_tokens = sum(len(tokens[annotator]) for annotator in annotators)
    if total_tokens == 0:
        # Check if there's actually data in any of the annotator columns
        has_data = any(pd.notnull(row[annotator]) and len(str(row[annotator])) > 0 for annotator in annotators)
        if has_data:
            # Try a more permissive pattern
            pattern = r'([^ (]+)\s*\(\s*(\d+)\s*\)\s*(\S*)'
            for annotator in annotators:
                if pd.notnull(row[annotator]):
                    tokens[annotator] = re.findall(pattern, str(row[annotator]))
    
    # Find the maximum number of tokens across all annotators
    max_len = max(len(tokens[annotator]) for annotator in annotators) if tokens else 0
    consolidated_tokens = []
    
    for i in range(max_len):
        # Get available tokens for this position
        available_tokens = []
        for annotator in annotators:
            if i < len(tokens[annotator]):
                available_tokens.append(tokens[annotator][i])
        
        # Skip if no annotations available for this position
        if not available_tokens:
            continue
        
        # Get the word and punctuation from the first available token
        word = available_tokens[0][0]
        punct = available_tokens[0][2]
        
        # Get all labels for this position
        labels = []
        for token in available_tokens:
            try:
                # Strip spaces and convert to integer
                labels.append(int(token[1].strip()))
            except ValueError:
                continue
        
        # Skip if no valid labels
        if not labels:
            continue
        
        # Determine consolidated label based on majority
        if len(labels) >= 2 and labels.count(1) >= 2:
            consolidated_label = 1
        else:
            consolidated_label = 0
        
        consolidated_tokens.append(f"{word}({consolidated_label}){punct}")
    
    return ' '.join(consolidated_tokens)

# Apply the consolidation function row-wise to create a new 'Consolidated' column
annotate_df['Consolidated'] = annotate_df.apply(consolidate_annotations, axis=1)

# Drop rows where Consolidated is NaN or empty
annotate_df = annotate_df[annotate_df['Consolidated'].notna() & (annotate_df['Consolidated'].str.strip() != '') & (annotate_df['Consolidated'] != ' ')]

# Print the first few rows of the dataframe with the consolidated column
# print(annotate_df.head())

# Save the consolidated annotations to a new CSV file
annotate_df.to_csv("consolidated_annotations.csv", index=False, encoding='utf-8')

# add consolidated annotations to the original dataframe
prosody_df['Consolidated'] = annotate_df['Consolidated']
# Save the updated dataframe to a new CSV file
prosody_df.to_csv("excel_files/Akan Speech Emotion Dataset with Prosody Labels Consolidated.csv", index=False, encoding='utf-8')