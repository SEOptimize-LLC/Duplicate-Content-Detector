#!/usr/bin/env python3
"""Setup script for the AI-Powered Duplicate Content Detector."""

import os
import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")

def download_spacy_model():
    """Download spaCy English model."""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")

def create_directories():
    """Create necessary directories."""
    directories = ["cache", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print("✅ Python version compatible")

def main():
    """Main setup function."""
    print("🚀 Setting up AI-Powered Duplicate Content Detector...")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return
    
    # Download NLTK data
    print("📚 Downloading NLTK data...")
    download_nltk_data()
    
    # Download spaCy model
    print("🧠 Downloading spaCy model...")
    download_spacy_model()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Or use the CLI: python utils.py --help")
    print("3. Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
