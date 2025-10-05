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
