"""
File and utility operations module
"""
import os
import logging

class FileUtils:
    """Utilities for file operations"""
    
    def __init__(self, config_manager):
        self.config = config_manager
    
    def clear_all_files(self):
        """Clear all files from upload folder"""
        for root, dirs, files in os.walk(self.config.UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to remove file {file_path}: {e}")
    
    def save_text_to_file(self, text, filename):
        """
        Save text to a file in the upload folder
        
        Args:
            text: Text content to save
            filename: Name of the file to save to
            
        Returns:
            Success message with file path
        """
        file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            logging.debug(f"Text saved to {file_path}")
            return f"Text saved to {file_path}"
        except Exception as e:
            logging.error(f"Failed to save text to {file_path}: {e}")
            raise
