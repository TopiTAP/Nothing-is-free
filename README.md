# Google Cloud Image Generation Web Interface

This application provides a web interface for generating images using Google Cloud's Vertex AI Imagen model.

## Prerequisites

Before using this application, you need to:

1. **Install Google Cloud SDK**
   - Windows: [Download Installer](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)
   - Other platforms: [Installation Guide](https://cloud.google.com/sdk/docs/install)

2. **Authenticate with Google Cloud**
   Run these commands in your terminal or command prompt:
   ```
   gcloud auth login
   gcloud auth application-default login
   gcloud auth application-default set-quota-project [YOUR_PROJECT_ID]
   ```
   Replace `[YOUR_PROJECT_ID]` with your actual Google Cloud project ID.

## Installation

1. Clone this repository or download the source code.
2. Ensure you have Python 3.7+ installed.
3. Run the `run_server.bat` file (Windows) or use the following commands:
   ```
   pip install flask flask-cors pillow google-cloud-aiplatform
   python server.py
   ```

## Usage

1. Run the `run_server.bat` file - this will:
   - Install required packages
   - Start the Flask server
   - Open your browser to http://localhost:5000

2. In the web interface:
   - Enter your prompts (one per line)
   - Configure generation settings
   - Enter your Google Cloud Project ID
   - Click "Start Generation"

3. Generated images will be saved to the `generated_images` folder.

## Features

- Batch processing of multiple prompts
- Customizable image generation parameters
- Support for custom Google Cloud Project ID
- Automatic browser launch
- Detailed progress tracking in the console

## Troubleshooting

If you encounter authentication errors:

1. Ensure you've installed the Google Cloud SDK
2. Run the authentication commands listed in the Prerequisites section
3. Verify your Project ID is correct
4. Check that your Google Cloud account has access to Vertex AI APIs

## Requirements

- Python 3.7+
- Google Cloud SDK
- Google Cloud account with Vertex AI access
- Internet connection