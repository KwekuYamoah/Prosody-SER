import os

def find_mp3_files(directory):
    """
    Recursively search for files with '_1.mp3' in their names in the given directory
    and its subdirectories.
    """
    found_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '_1.mp3' in file:
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    
    return found_files

if __name__ == "__main__":
    # Get the current directory as the starting point
    # current_dir = os.getcwd()
    
    # Find all matching files
    matching_files = find_mp3_files('audio/movie_audio_segments_mp3')
    
    # Print results
    if matching_files:
        print(f"Found {len(matching_files)} files with '_1.mp3' in their names:")
        for file in matching_files:
            print(file)
    else:
        print("No files with '_1.mp3' in their names were found.") 