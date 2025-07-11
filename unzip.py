import zipfile
import os

def unzip_to_same_folder(zip_path):
    """
    Unzips a folder and places its contents in the same directory
    as the zip file.

    Args:
        zip_path (str): The path to the zip file.
    """
    try:
        # Get the directory where the zip file is located.
        zip_dir = os.path.dirname(os.path.abspath(zip_path))

        # Create a ZipFile object.
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the contents of the zip file in the same directory.
            zip_ref.extractall(zip_dir)
            print(f"Successfully extracted '{os.path.basename(zip_path)}' to '{zip_dir}'")

    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: The file '{zip_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Example Usage ---
    # Replace 'path/to/your/folder.zip' with the actual path to your zip file.
    # For example, on Windows: 'C:\\Users\\YourUser\\Documents\\archive.zip'
    # On macOS or Linux: '/home/youruser/documents/archive.zip'

    # To test this, create a file named 'my_archive.zip' in the same
    # directory as this script and run it.
    # You can create a dummy zip file for testing.
    
    zip_file_path = 'audio/movie_audio_segments_wav.zip'
    unzip_to_same_folder(zip_file_path)