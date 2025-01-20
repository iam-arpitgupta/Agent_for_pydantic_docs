import os
import sys
import json
import asyncio
import requests
import numpy as np 
from xml.etree import ElementTree
from typing import List, Dict, Any
from datatclasses import dataclass 

from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic_ai.models.groq import GroqModel
from supabase import create_client, Client

from sentence_transformers import SentenceTransformer
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model = GroqModel("")


superbase :Client = create_client(
    os.getenv("SUPABASE_URL")
    os.getenv("SUPABASE_SERVICE_KEY")
)


@dataclass 
class ProcessedChunks:
    url : str 
    chunk_number : int 
    title : str 
    summary : str 
    content: str
    metadata: Dict[str, Any]
    embeddings : int 


def chunk_text(text:str , chunk_size : int  = 5000) -> List[str]:
    """split the text into chunks """
    chunks = []
    start = 0
    text_lenght = len(text)

    while start < text_lenght:
        #claculate the end position 
        end = start + chunk_size

        #if the text is at the end then append 
        if end >=  text_lenght:
            chunks.append(text[start:].split())
            break 

        #try to find the code vblock first 
        chunk = text[start:end]
        # rfind -> return the last index of the sub string 
        code_block = chunk.rfind('```')
        # This checks if the string "```" was found within the chunk. 
        # If it wasn't found, code_block will be -1, and the condition will be False.
        # This checks if the last occurrence of "```" is located more than 30% of the way through the chunk. 
        # This condition helps to ensure that the extracted code block is not too short or insignificant.
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        #break the paragraph 
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break


        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                # This line aims to correctly determine the ending index (end) of a sentence within the text chunk
                end = start + last_period + 1

        
        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)


        
# max(start + 1, end) ensures that the start position for the next chunk is always at least one character beyond the end position of the current chunk.
        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

#stroing the docs in the database 
async def get_title_and_summary(chunk : str ,url :str)-> Dict[str,str]:
    """ get the title and the summary from the llm """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = await groq_client.chat.completions.create(
            model = os.getenv("")
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_formate = {"type" ; "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}
    
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using sentence-transformers."""
    try:
        # Generate embedding
        embedding = embeddings_model.encode(text, convert_to_numpy=True)
        
        # Normalize the embedding vector
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 384  # Return zero vector matching model dimensions


async def process_chunk(chunk: str, chunk_number: int, url: str)-> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

    
async def insert_chunk():
    """Insert the Processed chunk  into superbase """
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        results = superbase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return results

    except Exception as e:
        print(f"error in inserting the data {e}")
        return None


async def process_and_store_documents(url:str , markdown: str):
    """
    Process the data and store into chunks in ||
    """
    # split into chunks 
    chunks = chunk_text(markdown)

    #process chunks in || 
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    #store them || 
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]

    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """CRAWL multiple urls in parallel with a concurrent limit 
    get all the data and then stores it into the process_and_store_documents function"""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )   

    crawl_config = CrawlerRunConfig(cache_mode = CacheMode.BYPASS)

    #create the crawler instance            
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_documents(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


def get_pydantic_ai_docs_urls()-> List[str];
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """  
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()


    root = ElementTree.fromstring(response.content)


    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    # findall() is a method of the Element class in lxml that searches the XML tree for all elements that match the given XPath expression
    urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

async def main():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())





