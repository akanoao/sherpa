import urllib.request
import zipfile
import os

url = "https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip"
filename = "vosk-model-small-hi-0.22.zip"

print(f"Downloading lightweight Hindi model ({url})...")
urllib.request.urlretrieve(url, filename)

print("Download complete! Extracting...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(".")

print("Done! You can delete the .zip file now.")