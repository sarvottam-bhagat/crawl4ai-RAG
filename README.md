# Pydantic AI Documentation RAG System

A Retrieval-Augmented Generation (RAG) system built with Crawl4AI, Supabase (with pgvector), and OpenAI. This system crawls the Pydantic AI documentation, processes the content, and creates a knowledge base that can be queried using natural language.

## Video Demo

<div align="center">
  <a href="https://youtu.be/iN_mLdVuIm4?si=r8dzMlthJv4XqjGY">
    <img src="https://img.youtube.com/vi/iN_mLdVuIm4/maxresdefault.jpg" alt="Pydantic AI Documentation RAG System Demo" width="100%">
  </a>
  <p>Click the image above to watch the demo video</p>
</div>

## Project Overview

This project provides a complete RAG pipeline for Pydantic AI documentation:

1. **Web Crawling**: Crawl the Pydantic AI documentation website using Crawl4AI
2. **Content Processing**: Process and chunk the content for efficient storage
3. **Embedding Generation**: Generate embeddings for semantic search using OpenAI
4. **Vector Storage**: Store content and embeddings in Supabase with pgvector
5. **Retrieval**: Retrieve relevant content based on user queries
6. **Generation**: Generate responses using OpenAI's models

## Key Components

### Core Files

- **crawl_pydantic_ai_docs.py**: Crawls the Pydantic AI documentation and stores it in Supabase
- **pydantic_ai_agent.py**: Defines the agent that handles user queries using the RAG system
- **streamlit_ui.py**: Streamlit user interface for interacting with the agent
- **site_pages.sql**: SQL schema for the Supabase database, including vector search functions

## Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account and project
- Crawl4AI library

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your-openai-api-key-here
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_KEY=your-supabase-service-key
LLM_MODEL=gpt-4o-mini  # or another OpenAI model
```

4. Set up your Supabase database:
   - Enable the pgvector extension in your Supabase project
   - Run the SQL commands in `site_pages.sql` to create the necessary tables and functions

## Usage

### Crawling the Documentation

First, crawl and store the Pydantic AI documentation:

```bash
python crawl_pydantic_ai_docs.py
```

This script:
- Fetches the sitemap from the Pydantic AI documentation site
- Crawls each page in the sitemap
- Processes the content (chunking, generating titles, summaries, and embeddings)
- Stores the processed content in Supabase

### Running the Streamlit UI

Once the crawling is complete, run the Streamlit interface:

```bash
streamlit run streamlit_ui.py
```

This will start a web interface where you can:
- Ask questions about Pydantic AI
- Get answers based on the official documentation
- See the conversation history

## How It Works

### Web Crawling

The system uses Crawl4AI to extract content from the Pydantic AI documentation:

```python
# Get URLs from Pydantic AI docs sitemap
urls = get_pydantic_ai_docs_urls()
await crawl_parallel(urls)
```

### Content Processing

The crawler processes web content by:
1. Extracting the raw HTML
2. Converting it to Markdown
3. Chunking the content into manageable pieces
4. Extracting titles and summaries using OpenAI
5. Generating embeddings for each chunk

### Storage in Supabase

Processed content is stored in Supabase with pgvector for efficient semantic search:

```python
async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    data = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "summary": chunk.summary,
        "content": chunk.content,
        "metadata": {
            "source": "pydantic_ai_docs",
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        },
        "embedding": chunk.embedding,
    }

    result = supabase.table("site_pages").insert(data).execute()
```

### Retrieval and Generation

When a user asks a question, the agent:
1. Converts the question to an embedding
2. Searches Supabase for relevant content using vector similarity
3. Uses the retrieved content as context for OpenAI to generate a response

```python
@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    # Get the embedding for the query
    query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

    # Query Supabase for relevant documents
    result = ctx.deps.supabase.rpc(
        'match_site_pages',
        {
            'query_embedding': query_embedding,
            'match_count': 5,
            'filter': {'source': 'pydantic_ai_docs'}
        }
    ).execute()
```

## Key Features

- **Documentation-specific**: Focused on providing accurate information about Pydantic AI
- **Vector Search**: Uses pgvector in Supabase for efficient semantic search
- **Streaming Responses**: Provides real-time streaming of responses in the UI
- **Conversation History**: Maintains context across multiple questions
- **User-friendly Interface**: Clean Streamlit UI for easy interaction

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service key
- `LLM_MODEL`: The OpenAI model to use (default: gpt-4o-mini)

