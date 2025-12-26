# !/bin/bash

echo "Starting post-install script..."

# 1. Install pip dependencies
pip install -r requirements.txt

# 2. Download the saved_model.zip from Google Drive
echo "Downloading saved_model.zip from Google Drive..."
wget -O saved_model.zip "https://drive.google.com/drive/folders/1rOLp_GEBoPhMGFAAxy8nwEEDkKmxjn39?usp=sharing"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download saved_model.zip. Please check the Google Drive link."
    exit 1
fi

echo "Extracting saved_model.zip..."
unzip saved_model.zip

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract saved_model.zip. Ensure the file is valid."
    exit 1
fi

# Clean up the zip file to save space (optional)
rm saved_model.zip

echo "Post-install script completed successfully."

# Ensure the Streamlit app can find the model directory
# This assumes your app.py expects 'saved_model' in the root directory.
# If your model is inside another folder after unzipping, adjust accordingly.
