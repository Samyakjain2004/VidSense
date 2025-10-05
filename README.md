# VidSense - Multimodal Video Q&A Assistant

A Streamlit-based chatbot for analyzing videos, extracting audio transcriptions, visual descriptions, and cumulative stories, indexing them in Azure AI Search, and enabling users to query video details with text answers and relevant clips.

## Overview

This project leverages Azure services (OpenAI, AI Search, Computer Vision) to:
- Extract frames and audio from videos.
- Generate transcriptions, embeddings, and visual stories for 10-second segments.
- Create cumulative narratives for story continuity.
- Index data for semantic search.
- Provide a chatbot interface for querying video content with answers and clips.

## Features

- **Video Upload & Processing**: Upload videos or use existing IDs; processes audio, frames, and descriptions in parallel.
- **Audio Transcription**: Splits audio into 10s segments, transcribes with GPT-4o, embeds, and indexes with prev/next segment links.
- **Frame Extraction & Embedding**: Extracts frames at 2 FPS, averages embeddings over 10s windows using Azure Computer Vision.
- **Visual Descriptions**: Uses GPT-4o-vision to describe 10s frame windows (story, key events, OCR texts, objects, actions, etc.).
- **Cumulative Stories**: Links descriptions into a coherent narrative using GPT-4o.
- **Search & Retrieval**: Performs vector/semantic search across audio, visual, and story indexes in Azure AI Search.
- **Chatbot**: Streamlit UI for querying, selecting best chunks, generating answers, and cropping clips (requires ffmpeg).
- **Logging & Error Handling**: Comprehensive logging in files like `app.log`, `chatbot.log`, etc.

## Prerequisites

- Python 3.10+
- Azure subscriptions for:
  - OpenAI (two resources: one for embeddings/transcriptions, one for GPT-4o evaluations/stories)
  - AI Search
  - Computer Vision (for image embeddings)
- FFmpeg (for video clipping; optional but recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Samyakjain2004/VidSense.git
   cd VidSense

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Configure environment variables (see Configuration).
4. Ensure directories ./Uploads, frames, output exist or are created automatically.

## Configuration
Create a .env file in the project root with the following:
```plaintext
# Azure OpenAI Resource 1 (Embeddings & Transcription)
AZURE_OPENAI_API_KEY1=<your-key>
AZURE_OPENAI_ENDPOINT1=<your-endpoint>
AZURE_OPENAI_API_VERSION_EMBEDDING1=2023-12-01-preview
AZURE_OPENAI_API_VERSION_AUDIO1=2024-05-01-preview

# Azure OpenAI Resource 2 (GPT-4o for Evaluations & Stories)
AZURE_OPENAI_API_KEY2=<your-key>
AZURE_OPENAI_ENDPOINT2=<your-endpoint>
AZURE_OPENAI_API_VERSION_GPT4O=2024-12-01-preview
AZURE_DEPLOYMENT_GPT4O=gpt-4o

# Azure AI Search
AZURE_SEARCH_ENDPOINT1=<your-endpoint>
AZURE_SEARCH_ADMIN_KEY1=<your-key>
AZURE_SEARCH_INDEX_NAME1=audio_index
AZURE_SEARCH_INDEX_NAME2=frames_index
AZURE_SEARCH_INDEX_NAME3=gpt_index
AZURE_SEARCH_INDEX_NAME4=cumulative_story_index

# Azure Computer Vision
AZURE_AI_VISION_API_KEY=<your-key>
AZURE_AI_VISION_REGION=<your-region>
AZURE_AI_VISION_ENDPOINT=<your-endpoint>

# Directories & Outputs
IMAGES_DIR=frames
OUTPUT_JSON=output/frameVectors.json
FRAME_DESCRIPTIONS=output/frame_descriptions.jsonl
FRAME_VECTORS=output/frame_descriptions_with_vecs.jsonl
```

Obtain keys and endpoints from the Azure portal.

## Usage
Processing a Video
Run the video processing pipeline:
```bash
  python main.py path/to/video.mp4
```

- Generates a unique video_id.
- Extracts frames at 2 FPS, processes audio, and generates descriptions.
= Indexes data in Azure AI Search.
- Launches the chatbot for querying.

## Running the Streamlit App
Start the Streamlit interface:
```bash
  streamlit run app.py
```
- Upload a video or enter an existing video_id.
- Query video details (e.g., "What happened at 30 seconds?").
- Receive text answers and video clips (if ffmpeg is installed).

## Project Structure

- app.py: Streamlit UI for video upload and chatbot interaction.
- main.py: Orchestrates video processing (frames, audio, descriptions, stories).
- vid.py: Extracts frames at 2 FPS.
- audio.py: Extracts, splits (10s), transcribes, and indexes audio.
- frames.py: Embeds frames in 10s windows using Azure Computer Vision.
- gpt_desc.py: Generates visual descriptions for 10s frame windows.
- cumulative_story.py: Builds coherent narratives across segments.
- chatbot.py: Handles queries, searches indexes, and generates answers/clips.

## Notes

- Ensure Azure credentials and endpoints are valid.
- Install FFmpeg for video clipping (pip install ffmpeg-python and FFmpeg binary).
- Logs are saved in *.log files for debugging.
- Indexes are managed automatically but can be configured via Azure portal.

## Troubleshooting

- Dependency Errors: Verify requirements.txt and install FFmpeg if needed.
- Azure Issues: Check .env for correct keys/endpoints.
- No Video Clips: Ensure ffmpeg-python and FFmpeg binary are installed.
- Index Problems: Confirm index names and Azure Search connectivity.

## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Samyak Jain
🔗 LinkedIn - https://www.linkedin.com/in/samyak-jain-470b7b255

🔗 GitHub - https://github.com/Samyakjain2004
