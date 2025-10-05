import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict
import logging

# Import modified functions (assuming the other files are modified as below)
from audio import process_and_index_audio  # Audio remains similar, 10-sec chunks
from vid import extract_frames  # Modified for 2 fps
from frames import frames  # Modified for 10-sec window aggregation
from gpt_desc import gpt_desc  # Modified for 10-sec story descriptions
from cumulative_story import process_cumulative_stories
from chatbot import run_chatbot  # Assuming this exists from ref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

async def process_video(video_path: str, video_id: str) -> Dict[str, Any]:
    """
    Process a video: Extract frames at 2 fps, then process audio, CV embeddings, GPT descriptions
    in 10-sec windows for story-building.
    """
    try:
        # Step 1: Extract frames at 2 fps (no max_frames limit for full video)
        logger.info(f"Extracting frames from video: {video_path} at 2 fps")
        frame_results = await asyncio.to_thread(extract_frames, str(video_path), str(Path("frames")), fps=2.0, max_frames=None)
        if not isinstance(frame_results, list):
            logger.error(f"Frame extraction failed: {frame_results}")
            return {"status": "failed", "error": f"Frame extraction returned invalid output: {frame_results}"}

        # Step 2: Run audio, frames (CV), and gpt_desc in parallel, all with 10-sec windows
        logger.info("Running audio, CV embedding, and GPT story description in parallel with 10-sec windows")
        audio_task = asyncio.to_thread(process_and_index_audio, video_path, video_id)
        frames_task = asyncio.to_thread(frames, video_id)  # FIXED: Pass video_id
        gpt_desc_task = asyncio.to_thread(gpt_desc)  # Now generates story per 10-sec window
        
        audio_result, frames_result, gpt_desc_result = await asyncio.gather(
            audio_task,
            frames_task,
            gpt_desc_task,
            return_exceptions=True
        )

        # Step 3: Collect results and run cumulative story if gpt_desc succeeds
        tool_payloads = {}
        
        # Process audio result (already 10-sec chunks with story links)
        if isinstance(audio_result, Exception):
            logger.error(f"Audio processing failed: {str(audio_result)}")
            tool_payloads["audio"] = {"status": "failed", "error": str(audio_result)}
        else:
            tool_payloads["audio"] = audio_result if isinstance(audio_result, dict) else {"status": "processed", "video_path": video_path}
        
        # Process frames (CV) result
        if isinstance(frames_result, Exception):
            logger.error(f"CV processing failed: {str(frames_result)}")
            tool_payloads["frames"] = {"status": "failed", "error": str(frames_result)}
        else:
            tool_payloads["frames"] = frames_result if isinstance(frames_result, dict) else {"status": "failed", "error": f"Invalid frames output: {frames_result}"}
        
        # Process gpt_desc result
        if isinstance(gpt_desc_result, Exception):
            logger.error(f"GPT description processing failed: {str(gpt_desc_result)}")
            tool_payloads["gpt_desc"] = {"status": "failed", "error": str(gpt_desc_result)}
        else:
            tool_payloads["gpt_desc"] = gpt_desc_result if isinstance(gpt_desc_result, dict) else {"status": "failed", "error": f"Invalid gpt_desc output: {gpt_desc_result}"}
        
        # Run cumulative story only if gpt_desc succeeds
        if tool_payloads.get("gpt_desc", {}).get("status") == "success":
            logger.info("Running cumulative story generation")
            cumulative_task = asyncio.to_thread(process_cumulative_stories)
            cumulative_result = await cumulative_task
            tool_payloads["cumulative_story"] = cumulative_result if isinstance(cumulative_result, dict) else {"status": "failed", "error": f"Invalid cumulative story output: {cumulative_result}"}
        else:
            logger.info("Skipping cumulative story due to gpt_desc failure")
            tool_payloads["cumulative_story"] = {"status": "failed", "error": "gpt_desc failed, cannot generate cumulative story"}

        logger.info(f"Processing results: {json.dumps(tool_payloads, indent=2)}")
        return tool_payloads

    except Exception as e:
        logger.error(f"Video processing error: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    video_id = uuid.uuid4().hex
    logger.info(f"Processing video: {video_path} with video_id: {video_id}")

    try:
        tool_payloads = asyncio.run(process_video(str(video_path), video_id))
        print("Processing complete. Results:")
        print(json.dumps(tool_payloads, indent=2))
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Starting chatbot...")
    asyncio.run(run_chatbot(video_id))

if __name__ == "__main__":
    main()