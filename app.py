import streamlit as st
import uuid
import asyncio
from pathlib import Path
import os
import sys
import logging
import json
from typing import Dict, Any
from main import process_video  # From provided main.py
from chatbot import retrieve_and_answer  # From updated chatbot.py with VectorizedQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Cricket Video Chatbot", layout="wide")

st.sidebar.header("🎥 Video Options")

# Choose between video upload or existing ID
upload_option = st.sidebar.radio("Choose input method:", ["Upload Video", "Use Existing Video ID"])

video_id = None
video_path = None  # Store video path for retrieval and clip generation

# Function to get video path by video_id
def get_video_path_by_id(video_id: str) -> str:
    """Retrieve video path based on video_id."""
    uploads_dir = Path("./Uploads")
    for file in uploads_dir.glob(f"*{video_id}*"):
        if file.suffix.lower() in [".mp4", ".mov", ".avi"]:
            return str(file)
    return None

if upload_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save video with video_id in filename for easy retrieval
        video_id = uuid.uuid4().hex
        video_path = Path(f"./Uploads/{video_id}_{uploaded_file.name}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        
        # Display video preview in sidebar
        st.sidebar.video(str(video_path), format="video/mp4", start_time=0)
        
        with st.spinner("🚀 Processing video... please wait"):
            try:
                tool_payloads = asyncio.run(process_video(str(video_path), video_id))
                if tool_payloads.get("status") == "failed":
                    st.sidebar.error(f"Pipeline error: {tool_payloads.get('error')}")
                else:
                    st.sidebar.success("✅ Video processed successfully!")
                    st.sidebar.json(tool_payloads)
                    st.session_state["video_id"] = video_id  # Store video_id
                    st.session_state["video_path"] = str(video_path)  # Store video path
            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}", exc_info=True)
                st.sidebar.error(f"Pipeline error: {str(e)}")

elif upload_option == "Use Existing Video ID":
    video_id = st.sidebar.text_input("Enter Video ID")
    if video_id:
        video_path = get_video_path_by_id(video_id)
        if video_path and os.path.exists(video_path):
            st.sidebar.success(f"Using video ID: {video_id}")
            # Display video preview in sidebar
            st.sidebar.video(video_path, format="video/mp4", start_time=0)
            st.session_state["video_id"] = video_id  # Store video_id
            st.session_state["video_path"] = video_path  # Store video path
        else:
            st.sidebar.error(f"No video found for ID: {video_id}")

st.title("💬 Cricket Video Chatbot")

if "video_id" in st.session_state and st.session_state["video_id"]:
    video_id = st.session_state["video_id"]
    video_path = st.session_state.get("video_path")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            if isinstance(content, dict):
                st.write(content["text_answer"])
                if content.get("video_clip") and os.path.exists(content["video_clip"]):
                    st.video(content["video_clip"], format="video/mp4", start_time=0)
                    st.caption(f"Clip: {content['start_time']}s to {content['end_time']}s")
                else:
                    st.warning("Video clip not available.")
            else:
                st.write(content)

    # Handle user query
    if query := st.chat_input("Ask about the cricket video (e.g., 'Who took the third wicket?')"):
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(retrieve_and_answer(query, video_id, video_path))
                    st.write(response["text_answer"])
                    if response.get("video_clip") and os.path.exists(response["video_clip"]):
                        st.video(response["video_clip"], format="video/mp4", start_time=0)
                        st.caption(f"Clip: {response['start_time']}s to {response['end_time']}s")
                    else:
                        st.warning("Video clip not available.")
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    logger.error(f"Query processing error: {str(e)}", exc_info=True)
                    st.error(f"Error processing query: {str(e)}")
                    response = {
                        "text_answer": "Sorry, an error occurred while processing your query. Please try again.",
                        "source": "none",
                        "start_time": None,
                        "end_time": None,
                        "video_clip": None
                    }
                    st.write(response["text_answer"])
                    st.session_state.chat_history.append(("assistant", response))
else:
    st.info("⬅️ Upload a video or enter an existing Video ID to start chatting.")