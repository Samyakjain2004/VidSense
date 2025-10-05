import os
import json
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Azure Search clients
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT1")
azure_search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY1"))
#CUMULATIVE_INDEX_NAME = "cumulative_story_index"
CUMULATIVE_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME4")
search_index_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=azure_search_credential)
search_client = SearchClient(endpoint=azure_search_endpoint, index_name=CUMULATIVE_INDEX_NAME, credential=azure_search_credential)

# Azure OpenAI for embeddings
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT1", "")
openai_key = os.getenv("AZURE_OPENAI_API_KEY1", "")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1", "2024-12-01-preview")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
embed_client = AzureOpenAI(
    api_key=openai_key,
    api_version=openai_api_version,
    azure_endpoint=openai_endpoint
)

# Azure OpenAI for GPT-4o story generation
gpt_api_key = os.getenv("AZURE_OPENAI_API_KEY2", "")
gpt_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2", "")
gpt_api_version = os.getenv("AZURE_OPENAI_API_VERSION_GPT4O", "2024-12-01-preview")
gpt_deployment = os.getenv("AZURE_DEPLOYMENT_GPT4O", "")
gpt_client = AzureOpenAI(
    api_key=gpt_api_key,
    api_version=gpt_api_version,
    azure_endpoint=gpt_endpoint
)

# Input file from gpt_desc.py
GPT_OUTPUT_JSONL = "output/frame_descriptions_with_vecs.jsonl"

def create_or_update_cumulative_index():
    """Create or update the cumulative story index in Azure Search."""
    try:
        search_index_client.get_index(CUMULATIVE_INDEX_NAME)
        search_index_client.delete_index(CUMULATIVE_INDEX_NAME)
        print(f"Deleted existing index '{CUMULATIVE_INDEX_NAME}'.")
    except ResourceNotFoundError:
        print(f"Index '{CUMULATIVE_INDEX_NAME}' does not exist. Proceeding to create.")
    except Exception as e:
        print(f"Error deleting index: {e}")
        raise

    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="video_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchField(name="start_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="end_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="cumulative_story", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="story_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSION,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchField(name="prev_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="next_id", type=SearchFieldDataType.String, filterable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4, ef_construction=400, ef_search=1000, metric=VectorSearchAlgorithmMetric.COSINE
                ),
            ),
        ],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default_semantic_config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="cumulative_story")]
                ),
            )
        ]
    )

    index = SearchIndex(name=CUMULATIVE_INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    
    try:
        search_index_client.create_or_update_index(index)
        print(f"Index '{CUMULATIVE_INDEX_NAME}' created or updated.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def generate_embedding(text):
    """Generate embedding for text using Azure OpenAI."""
    try:
        response = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def generate_cumulative_story(previous_chunk: dict, current_chunk: dict) -> str:
    """Use GPT-4o to create a coherent cumulative story linking current chunk to previous context."""
    system_prompt = (
        "You are an expert visual storyteller specializing in cricket matches. Create a concise, flowing narrative "
        "for a 10-second video chunk that continues the story from the previous chunk. Focus on maintaining story flow "
        "without repeating the full previous narrative. Reference only relevant context from the previous chunk (e.g., "
        "recent events like the last wicket, score, or key moment) to ensure continuity. Incorporate the current chunk’s "
        "details (story, key events, texts like scores or wickets, objects, actions, scene, attributes, tags) into the "
        "narrative naturally. Return only the story as plain text, suitable for answering questions like 'Who took the third wicket?'"
    )
    previous_summary = (
        f"Previous chunk ({previous_chunk['start_time']}-{previous_chunk['end_time']}s):\n"
        f"Story: {previous_chunk['story']}\n"
        f"Key Events: {', '.join(previous_chunk['key_events'])}\n"
        f"Texts (OCR): {', '.join(previous_chunk['texts'])}\n"
        f"Objects: {', '.join(previous_chunk['objects'])}\n"
        f"Actions: {', '.join(previous_chunk['actions'])}\n"
        f"Scene: {previous_chunk['scene']}\n"
        f"Attributes: {', '.join(previous_chunk['attributes'])}\n"
        f"Tags: {', '.join(previous_chunk['tags'])}"
    ) if previous_chunk else "No previous chunk."
    current_summary = (
        f"Current chunk ({current_chunk['start_time']}-{current_chunk['end_time']}s):\n"
        f"Story: {current_chunk['story']}\n"
        f"Key Events: {', '.join(current_chunk['key_events'])}\n"
        f"Texts (OCR): {', '.join(current_chunk['texts'])}\n"
        f"Objects: {', '.join(current_chunk['objects'])}\n"
        f"Actions: {', '.join(current_chunk['actions'])}\n"
        f"Scene: {current_chunk['scene']}\n"
        f"Attributes: {', '.join(current_chunk['attributes'])}\n"
        f"Tags: {', '.join(current_chunk['tags'])}"
    )

    user_prompt = (
        f"Previous context:\n{previous_summary}\n\n"
        f"Current chunk details:\n{current_summary}\n\n"
        "Create a concise narrative that continues the story, linking to the previous chunk’s relevant events "
        "(e.g., last wicket, score update) without repeating its full story. Focus on cricket-specific details like "
        "wickets, overs, or scores for clarity."
    )

    try:
        response = gpt_client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating cumulative story: {e}")
        return current_chunk['story']  # Fallback to current chunk’s story

def process_cumulative_stories():
    """Load gpt_desc chunks, generate cumulative stories with GPT-4o, and index them."""
    if not os.path.exists(GPT_OUTPUT_JSONL):
        raise FileNotFoundError(f"GPT output not found: {GPT_OUTPUT_JSONL}")

    chunks = []
    with open(GPT_OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Sort by start_time to ensure sequential order
    chunks.sort(key=lambda x: x["start_time"])

    documents = []
    prev_doc_id = None
    video_id = chunks[0]["video_id"] if chunks else None

    for idx, chunk in enumerate(chunks):
        # Generate story: First chunk uses its own story; others link to previous chunk
        current_story = chunk["story"] if idx == 0 else generate_cumulative_story(chunks[idx-1], chunk)
        if not current_story:
            print(f"Warning: No story generated for chunk {idx + 1}")
            continue

        story_vector = generate_embedding(current_story)
        if not story_vector:
            print(f"Warning: No embedding generated for chunk {idx + 1}")
            continue

        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "video_id": chunk["video_id"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "cumulative_story": current_story,
            "story_vector": story_vector,
            "prev_id": prev_doc_id,
            "next_id": None,
        }
        documents.append(doc)
        if prev_doc_id:
            documents[-2]["next_id"] = doc_id
        prev_doc_id = doc_id

    if documents:
        create_or_update_cumulative_index()
        search_client.merge_or_upload_documents(documents)
        print(f"Indexed {len(documents)} cumulative stories for video_id: {video_id}")
        # Save locally for verification
        with open("output/cumulative_stories.json", "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2)
        print("Saved cumulative stories to output/cumulative_stories.json")
    else:
        print("No chunks found to process.")

    return {"status": "success", "documents_indexed": len(documents), "docs_path": "output/cumulative_stories.json"}

if __name__ == "__main__":
    process_cumulative_stories()