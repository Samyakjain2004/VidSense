# import os
# import json
# import asyncio
# import time
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient
# from azure.search.documents.models import VectorizedQuery
# from openai import AzureOpenAI
# from dotenv import load_dotenv
# import logging
# try:
#     import ffmpeg
#     FFMPEG_AVAILABLE = True
# except ImportError as e:
#     logging.error(f"Failed to import ffmpeg-python: {e}. Video clipping will be disabled.")
#     ffmpeg = None
#     FFMPEG_AVAILABLE = False

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler('chatbot.log'), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# load_dotenv()

# # Azure Search clients
# SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "")
# SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()
# AUDIO_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME1", "audio_index")
# GPT_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME3", "gpt_index")
# CUMULATIVE_INDEX = "cumulative_story_index"

# audio_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=AUDIO_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))
# gpt_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=GPT_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))
# cumulative_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=CUMULATIVE_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

# # Embedding client for query
# embedding_client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1", "2024-12-01-preview"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1")
# )
# EMBEDDING_MODEL = "text-embedding-3-small"

# # Eval LLM client (use GPT-4o for evaluation)
# eval_client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY2"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION_GPT4O", "2024-12-01-preview"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT2")
# )
# EVAL_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_GPT4O", "gpt-4o")

# def generate_embedding(text):
#     try:
#         time.sleep(0.1)  # Rate-limiting
#         response = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
#         logger.info(f"Embedding generated for text: {text[:50]}... (length: {len(response.data[0].embedding)})")
#         return response.data[0].embedding
#     except Exception as e:
#         logger.error(f"Query embedding error: {e}")
#         return None

# async def search_index(client, index_name, query, video_id, vector_field, select_fields, top_k=1):
#     """Vector search on an index with filter on video_id."""
#     query_vector = generate_embedding(query)
#     if not query_vector:
#         logger.error(f"No embedding generated for query: {query}")
#         return []

#     vector_query = VectorizedQuery(
#         vector=query_vector,
#         k_nearest_neighbors=top_k,
#         fields=vector_field,
#         kind="vector"
#     )
#     results = client.search(
#         search_text=query,
#         vector_queries=[vector_query],
#         #filter=f"video_id eq '{video_id}'",
#         select=select_fields,
#         top=top_k,
#         semantic_configuration_name="default_semantic_config"
#     )
#     docs = list(results)
#     logger.info(f"Search in {index_name}: {len(docs)} results for query: {query}")
#     for doc in docs:
#         logger.info(f"Result: {json.dumps(doc, indent=2)}")
#     return docs

# async def fetch_matching_chunks(client, index_name, video_id, start_time, end_time, vector_field, select_fields):
#     """Fetch chunks from an index matching start_time and end_time."""
#     results = client.search(
#         search_text="*",
#         filter=f"start_time eq {start_time} and end_time eq {end_time}",
#         select=select_fields,
#         top=1
#     )
#     docs = list(results)
#     logger.info(f"Fetched {len(docs)} matching chunks from {index_name} for start_time={start_time}, end_time={end_time}")
#     return docs

# async def select_best_chunk(query, candidates):
#     """Use GPT-4o to select the best candidate chunk based on query relevance."""
#     eval_prompt = (
#         f"Query: {query}\n\n"
#         "Below are candidate chunks with content from cumulative story, visual descriptions, and audio transcriptions. "
#         "Select the chunk that best answers the query, prioritizing wicket-specific details (e.g., bowler, batsman, wicket number). "
#         "Return a json response with the index of the best chunk (0-based) and a brief reason.\n\n"
#         + "\n".join([f"Chunk {i}:\n{c['combined_content']}" for i, c in enumerate(candidates)])
#         + "\n\nResponse format: {'index': <int>, 'reason': '<string>'}"
#     )
#     try:
#         time.sleep(0.1)  # Rate-limiting
#         response = eval_client.chat.completions.create(
#             model=EVAL_DEPLOYMENT,
#             messages=[{"role": "system", "content": "You are an evaluator returning a json response."}, {"role": "user", "content": eval_prompt}],
#             response_format={"type": "json_object"},
#             max_tokens=150,
#             temperature=0.0
#         )
#         result = json.loads(response.choices[0].message.content.strip())
#         logger.info(f"LLM selected chunk {result['index']}: {result['reason']}")
#         return result
#     except Exception as e:
#         logger.error(f"LLM selection error: {e}")
#         return {"index": 0, "reason": "Fallback to first chunk due to error"} if candidates else None

# async def generate_final_answer(query, selected_chunk):
#     """Generate a concise final answer based on the selected chunk."""
#     prompt = (
#         f"Query: {query}\n"
#         f"Selected Chunk Content: {selected_chunk['combined_content']}\n\n"
#         "Generate a concise answer to the query in json format, focusing on wicket-specific details if relevant. "
#         "Use plain text, suitable for answering questions like 'Who took the third wicket?'"
#         + "\n\nResponse format: {'answer': '<string>'}"
#     )
#     try:
#         time.sleep(0.1)  # Rate-limiting
#         response = eval_client.chat.completions.create(
#             model=EVAL_DEPLOYMENT,
#             messages=[{"role": "system", "content": "You are a concise answer generator returning a json response."}, {"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             max_tokens=100,
#             temperature=0.5
#         )
#         result = json.loads(response.choices[0].message.content.strip())
#         return result['answer']
#     except Exception as e:
#         logger.error(f"Final answer generation error: {e}")
#         return selected_chunk['combined_content'][:200] + "..."  # Fallback

# def crop_video_clip(video_path, start_time, end_time, output_clip_path):
#     """Crop a video clip using ffmpeg-python."""
#     if not FFMPEG_AVAILABLE:
#         logger.error("Cannot crop video: ffmpeg-python is not available.")
#         return None
#     try:
#         # Validate inputs
#         if not os.path.exists(video_path):
#             logger.error(f"Video file not found: {video_path}")
#             return None
#         if start_time < 0 or end_time <= start_time:
#             logger.error(f"Invalid clip times: start_time={start_time}, end_time={end_time}")
#             return None

#         # Probe video to check duration
#         probe = ffmpeg.probe(video_path)
#         video_duration = float(probe['format']['duration'])
#         if end_time > video_duration:
#             logger.warning(f"End time {end_time}s exceeds video duration {video_duration}s. Adjusting to {video_duration}s.")
#             end_time = video_duration

#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)

#         # Crop clip using ffmpeg
#         stream = ffmpeg.input(video_path, ss=start_time, t=end_time - start_time)
#         stream = ffmpeg.output(stream, output_clip_path, c='copy', loglevel='error')
#         ffmpeg.run(stream)
#         logger.info(f"Created video clip: {output_clip_path}")
#         return output_clip_path
#     except ffmpeg.Error as e:
#         logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
#         return None
#     except Exception as e:
#         logger.error(f"Video crop error: {e}")
#         return None

# async def retrieve_and_answer(query, video_id, video_path):
#     """Retrieve top 5 cumulative chunks, fetch corresponding gpt/audio chunks, select best via LLM, generate answer and clip."""
#     logger.info(f"Processing query: {query} for video_id: {video_id}")

#     # Check document counts for debugging
#     async def count_docs(client, index_name, video_id):
#         results = client.search(search_text="*", include_total_count=True)
#         count = results.get_count()
#         logger.info(f"Found {count} documents in {index_name} for video_id: {video_id}")
#         return count

#     await count_docs(cumulative_client, CUMULATIVE_INDEX, video_id)
#     await count_docs(gpt_client, GPT_INDEX, video_id)
#     await count_docs(audio_client, AUDIO_INDEX, video_id)

#     # Step 1: Search top 5 cumulative chunks
#     cumulative_results = await search_index(
#         cumulative_client, CUMULATIVE_INDEX, query, video_id, "story_vector",
#         ["cumulative_story", "start_time", "end_time", "prev_id", "next_id"], top_k=5
#     )
#     if not cumulative_results:
#         logger.warning(f"No results found in cumulative_story_index for query: {query}")
#         return {
#             "text_answer": "No relevant information found.",
#             "source": "none",
#             "start_time": None,
#             "end_time": None,
#             "video_clip": None
#         }

#     # Step 2: Fetch corresponding gpt and audio chunks
#     candidates = []
#     for cum_doc in cumulative_results:
#         start_time = cum_doc["start_time"]
#         end_time = cum_doc["end_time"]
#         logger.info(f"Processing cumulative chunk: start_time={start_time}, end_time={end_time}, story={cum_doc['cumulative_story'][:50]}...")

#         # Fetch matching GPT chunk
#         gpt_results = await fetch_matching_chunks(
#             gpt_client, GPT_INDEX, video_id, start_time, end_time, "image_vector",
#             ["story", "key_events", "texts", "objects", "actions", "scene", "attributes", "tags"]
#         )
#         gpt_content = (
#             f"Visual Story: {gpt_results[0]['story']}\n"
#             f"Key Events: {', '.join(gpt_results[0]['key_events'])}\n"
#             f"Texts: {', '.join(gpt_results[0]['texts'])}\n"
#             f"Objects: {', '.join(gpt_results[0]['objects'])}\n"
#             f"Actions: {', '.join(gpt_results[0]['actions'])}\n"
#             f"Scene: {gpt_results[0]['scene']}\n"
#             f"Attributes: {', '.join(gpt_results[0]['attributes'])}\n"
#             f"Tags: {', '.join(gpt_results[0]['tags'])}"
#         ) if gpt_results else "No visual description available."

#         # Fetch matching audio chunk
#         audio_results = await fetch_matching_chunks(
#             audio_client, AUDIO_INDEX, video_id, start_time, end_time, "content_vector",
#             ["content", "segment_id"]
#         )
#         audio_content = audio_results[0]["content"] if audio_results else "No audio transcription available."

#         # Combine content
#         combined_content = (
#             f"Cumulative Story: {cum_doc['cumulative_story']}\n"
#             f"{gpt_content}\n"
#             f"Audio Transcription: {audio_content}"
#         )
#         candidates.append({
#             "combined_content": combined_content,
#             "start_time": start_time,
#             "end_time": end_time,
#             "cumulative_doc": cum_doc,
#             "source": "combined"
#         })

#     # Step 3: Select best chunk via LLM
#     selection = await select_best_chunk(query, candidates)
#     if not selection or not candidates:
#         logger.warning(f"No suitable chunk selected for query: {query}")
#         return {
#             "text_answer": "No relevant information found.",
#             "source": "none",
#             "start_time": None,
#             "end_time": None,
#             "video_clip": None
#         }

#     selected_chunk = candidates[selection["index"]]
#     logger.info(f"Selected chunk: {selected_chunk['combined_content'][:100]}... Reason: {selection['reason']}")

#     # Step 4: Generate final answer
#     final_answer = await generate_final_answer(query, selected_chunk)

#     # Step 5: Crop video clip
#     output_clip_path = f"output/clip_{int(time.time() * 1000)}.mp4"
#     clip_path = crop_video_clip(video_path, selected_chunk["start_time"], selected_chunk["end_time"], output_clip_path) if FFMPEG_AVAILABLE else None

#     return {
#         "text_answer": final_answer,
#         "source": selected_chunk["source"],
#         "start_time": selected_chunk["start_time"],
#         "end_time": selected_chunk["end_time"],
#         "video_clip": clip_path
#     }

# async def run_chatbot(video_id: str):
#     logger.info(f"Starting chatbot for video_id: {video_id}")
#     video_path = input("Enter the original video path: ")
#     if not os.path.exists(video_path):
#         logger.error(f"Invalid video path: {video_path}")
#         print("Error: Video file not found.")
#         return
#     while True:
#         query = input("Enter your query (e.g., 'Who took the third wicket?') or 'quit' to exit: ")
#         if query.lower() == 'quit':
#             break

#         try:
#             result = await retrieve_and_answer(query, video_id, video_path)
#             print("\nFinal Answer:")
#             print(result["text_answer"])
#             if result["video_clip"]:
#                 print(f"10-sec Video Clip: {result['video_clip']} (from {result['start_time']}s to {result['end_time']}s)")
#             else:
#                 print("No video clip available. Ensure ffmpeg-python and ffmpeg are installed.")
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
#             print("Error processing query. Please try again.")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Run chatbot for video queries")
#     parser.add_argument("--video-id", required=True, help="Video ID to query")
#     args = parser.parse_args()
#     asyncio.run(run_chatbot(args.video_id))
import os
import json
import asyncio
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import ffmpeg-python: {e}. Video clipping will be disabled.")
    ffmpeg = None
    FFMPEG_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chatbot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Azure Search clients
SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()
AUDIO_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME1", "audio_index")
GPT_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME3", "gpt_index")
CUMULATIVE_INDEX = "cumulative_story_index"

audio_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=AUDIO_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))
gpt_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=GPT_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))
cumulative_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=CUMULATIVE_INDEX, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

# Embedding client for query
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1")
)
EMBEDDING_MODEL = "text-embedding-3-small"

# Eval LLM client (use GPT-4o for evaluation)
eval_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY2"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_GPT4O", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT2")
)
EVAL_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_GPT4O", "gpt-4o")

def generate_embedding(text):
    try:
        time.sleep(0.1)  # Rate-limiting
        response = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        logger.info(f"Embedding generated for text: {text[:50]}... (length: {len(response.data[0].embedding)})")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Query embedding error: {e}")
        return None

async def search_index(client, index_name, query, video_id, vector_field, select_fields, top_k=5):
    """Vector search on an index with filter on video_id."""
    query_vector = generate_embedding(query)
    if not query_vector:
        logger.error(f"No embedding generated for query: {query}")
        return []

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field,
        kind="vector"
    )
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        #filter=f"video_id eq '{video_id}'",
        select=select_fields,
        top=top_k,
        semantic_configuration_name="default_semantic_config"
    )
    docs = list(results)
    logger.info(f"Search in {index_name}: {len(docs)} results for query: {query}")
    for doc in docs:
        logger.info(f"Result: {json.dumps(doc, indent=2)}")
    return docs

async def select_best_chunk(query, candidates):
    """Use GPT-4o to select the best combination of chunks based on query relevance."""
    eval_prompt = (
        f"Query: {query}\n\n"
        "Below are candidate chunks from three indexes: cumulative story, visual descriptions (gpt), and audio transcriptions. "
        "Each chunk includes its source, content, start_time, and end_time. Select the best combination of chunks (one or more) "
        "that best answers the query, prioritizing wicket-specific details (e.g., bowler, batsman, wicket number). "
        "Return a json response with the indexes of the selected chunks (0-based) and a brief reason. "
        "Use the timestamps from the most relevant chunk for video clipping.\n\n"
        + "\n".join([f"Chunk {i} (Source: {c['source']}):\nContent: {c['content']}\nStart Time: {c['start_time']}s\nEnd Time: {c['end_time']}s" for i, c in enumerate(candidates)])
        + "\n\nResponse format: {'indexes': [<int>], 'reason': '<string>', 'start_time': <float>, 'end_time': <float>}"
    )
    try:
        time.sleep(0.1)  # Rate-limiting
        response = eval_client.chat.completions.create(
            model=EVAL_DEPLOYMENT,
            messages=[{"role": "system", "content": "You are an evaluator returning a json response."}, {"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content.strip())
        logger.info(f"LLM selected chunks {result['indexes']}: {result['reason']}")
        return result
    except Exception as e:
        logger.error(f"LLM selection error: {e}")
        return {"indexes": [0], "reason": "Fallback to first chunk due to error", "start_time": candidates[0]["start_time"], "end_time": candidates[0]["end_time"]} if candidates else None

async def generate_final_answer(query, selected_chunks):
    """Generate a concise final answer based on the selected chunks."""
    combined_content = "\n".join([c["content"] for c in selected_chunks])
    prompt = (
        f"Query: {query}\n"
        f"Selected Chunks Content:\n{combined_content}\n\n"
        "Generate a concise answer to the query in json format, focusing on wicket-specific details if relevant. "
        "Combine information from the selected chunks to provide a complete answer. "
        "Use plain text, suitable for answering questions like 'Who took the third wicket?'"
        + "\n\nResponse format: {'answer': '<string>'}"
    )
    try:
        time.sleep(0.1)  # Rate-limiting
        response = eval_client.chat.completions.create(
            model=EVAL_DEPLOYMENT,
            messages=[{"role": "system", "content": "You are a concise answer generator returning a json response."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.5
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result['answer']
    except Exception as e:
        logger.error(f"Final answer generation error: {e}")
        return combined_content[:200] + "..."  # Fallback

def crop_video_clip(video_path, start_time, end_time, output_clip_path):
    """Crop a video clip using ffmpeg-python."""
    if not FFMPEG_AVAILABLE:
        logger.error("Cannot crop video: ffmpeg-python is not available.")
        return None
    try:
        # Validate inputs
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        if start_time < 0 or end_time <= start_time:
            logger.error(f"Invalid clip times: start_time={start_time}, end_time={end_time}")
            return None

        # Probe video to check duration
        probe = ffmpeg.probe(video_path)
        video_duration = float(probe['format']['duration'])
        if end_time > video_duration:
            logger.warning(f"End time {end_time}s exceeds video duration {video_duration}s. Adjusting to {video_duration}s.")
            end_time = video_duration

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)

        # Crop clip using ffmpeg
        stream = ffmpeg.input(video_path, ss=start_time, t=end_time - start_time)
        stream = ffmpeg.output(stream, output_clip_path, c='copy', loglevel='error')
        ffmpeg.run(stream)
        logger.info(f"Created video clip: {output_clip_path}")
        return output_clip_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Video crop error: {e}")
        return None

async def retrieve_and_answer(query, video_id, video_path):
    """Retrieve top 5 chunks from each index, select best combination via LLM, generate answer and clip."""
    logger.info(f"Processing query: {query} for video_id: {video_id}")

    # Check document counts for debugging
    async def count_docs(client, index_name, video_id):
        results = client.search(search_text="*", include_total_count=True)
        count = results.get_count()
        logger.info(f"Found {count} documents in {index_name} for video_id: {video_id}")
        return count

    await count_docs(cumulative_client, CUMULATIVE_INDEX, video_id)
    await count_docs(gpt_client, GPT_INDEX, video_id)
    await count_docs(audio_client, AUDIO_INDEX, video_id)

    # Step 1: Search top 5 chunks from each index
    cumulative_results = await search_index(
        cumulative_client, CUMULATIVE_INDEX, query, video_id, "story_vector",
        ["cumulative_story", "start_time", "end_time", "prev_id", "next_id"], top_k=5
    )
    gpt_results = await search_index(
        gpt_client, GPT_INDEX, query, video_id, "image_vector",
        ["story", "key_events", "texts", "objects", "actions", "scene", "attributes", "tags", "start_time", "end_time"], top_k=5
    )
    audio_results = await search_index(
        audio_client, AUDIO_INDEX, query, video_id, "content_vector",
        ["content", "segment_id", "start_time", "end_time"], top_k=5
    )

    # Step 2: Combine all chunks into candidates
    candidates = []
    
    for cum_doc in cumulative_results:
        content = f"Cumulative Story: {cum_doc['cumulative_story']}"
        candidates.append({
            "content": content,
            "start_time": cum_doc["start_time"],
            "end_time": cum_doc["end_time"],
            "source": "cumulative",
            "doc": cum_doc
        })
        logger.info(f"Cumulative chunk: start_time={cum_doc['start_time']}, end_time={cum_doc['end_time']}, content={content[:50]}...")

    for gpt_doc in gpt_results:
        content = (
            f"Visual Story: {gpt_doc['story']}\n"
            f"Key Events: {', '.join(gpt_doc['key_events'])}\n"
            f"Texts: {', '.join(gpt_doc['texts'])}\n"
            f"Objects: {', '.join(gpt_doc['objects'])}\n"
            f"Actions: {', '.join(gpt_doc['actions'])}\n"
            f"Scene: {gpt_doc['scene']}\n"
            f"Attributes: {', '.join(gpt_doc['attributes'])}\n"
            f"Tags: {', '.join(gpt_doc['tags'])}"
        )
        candidates.append({
            "content": content,
            "start_time": gpt_doc["start_time"],
            "end_time": gpt_doc["end_time"],
            "source": "gpt",
            "doc": gpt_doc
        })
        logger.info(f"GPT chunk: start_time={gpt_doc['start_time']}, end_time={gpt_doc['end_time']}, content={content[:50]}...")

    for audio_doc in audio_results:
        content = f"Audio Transcription: {audio_doc['content']}"
        candidates.append({
            "content": content,
            "start_time": audio_doc["start_time"],
            "end_time": audio_doc["end_time"],
            "source": "audio",
            "doc": audio_doc
        })
        logger.info(f"Audio chunk: start_time={audio_doc['start_time']}, end_time={audio_doc['end_time']}, content={content[:50]}...")

    if not candidates:
        logger.warning(f"No results found in any index for query: {query}")
        return {
            "text_answer": "No relevant information found.",
            "source": "none",
            "start_time": None,
            "end_time": None,
            "video_clip": None
        }

    # Step 3: Select best chunk combination via LLM
    selection = await select_best_chunk(query, candidates)
    if not selection or not candidates:
        logger.warning(f"No suitable chunks selected for query: {query}")
        return {
            "text_answer": "No relevant information found.",
            "source": "none",
            "start_time": None,
            "end_time": None,
            "video_clip": None
        }

    selected_chunks = [candidates[i] for i in selection["indexes"]]
    selected_source = ", ".join([c["source"] for c in selected_chunks])
    logger.info(f"Selected chunks: {[c['content'][:100] for c in selected_chunks]}... Reason: {selection['reason']}")

    # Step 4: Generate final answer
    final_answer = await generate_final_answer(query, selected_chunks)

    # Step 5: Crop video clip using selected timestamps
    output_clip_path = f"output/clip_{int(time.time() * 1000)}.mp4"
    clip_path = crop_video_clip(video_path, selection["start_time"], selection["end_time"], output_clip_path) if FFMPEG_AVAILABLE else None

    return {
        "text_answer": final_answer,
        "source": selected_source,
        "start_time": selection["start_time"],
        "end_time": selection["end_time"],
        "video_clip": clip_path
    }

async def run_chatbot(video_id: str):
    logger.info(f"Starting chatbot for video_id: {video_id}")
    video_path = input("Enter the original video path: ")
    if not os.path.exists(video_path):
        logger.error(f"Invalid video path: {video_path}")
        print("Error: Video file not found.")
        return
    while True:
        query = input("Enter your query (e.g., 'Who took the third wicket?') or 'quit' to exit: ")
        if query.lower() == 'quit':
            break

        try:
            result = await retrieve_and_answer(query, video_id, video_path)
            print("\nFinal Answer:")
            print(result["text_answer"])
            if result["video_clip"]:
                print(f"Video Clip: {result['video_clip']} (from {result['start_time']}s to {result['end_time']}s)")
            else:
                print("No video clip available. Ensure ffmpeg-python and ffmpeg are installed.")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("Error processing query. Please try again.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run chatbot for video queries")
    parser.add_argument("--video-id", required=True, help="Video ID to query")
    args = parser.parse_args()
    asyncio.run(run_chatbot(args.video_id))