import os

def copy_image_to_static(source_path, dest_folder):
    """
    Copies an image from source_path to the dest_folder within the static directory.
    Creates the dest_folder if it does not exist.

    Args:
        source_path (str): The path to the source image file.
        dest_folder (str): The destination folder within the static directory.
    """
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get the base name of the source file
    file_name = os.path.basename(source_path)

    # Define the full destination path
    dest_path = os.path.join(dest_folder, file_name)

    # Copy the file
    with open(source_path, 'rb') as src_file:
        with open(dest_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
            
    print(f"Copied {source_path} to {dest_path}")
    
if __name__ == "__main__":
    # Example usage
    source_image_path_folder = 'Roy/Test_Images'  # Replace with your source folder containing images
    destination_folder = 'Roy/ML/static/images'  # Replace with your desired destination folder
    for image_file in os.listdir(source_image_path_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            source_image_path = os.path.join(source_image_path_folder, image_file)
            if os.path.isfile(source_image_path):
                copy_image_to_static(source_image_path, destination_folder)