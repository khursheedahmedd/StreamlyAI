from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import shutil
import glob
import assemblyai as aai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pytubefix import YouTube
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import moviepy.editor as mp
import random
import string
import threading
import uuid
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from langchain_core.documents import Document
import whisper


app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize API keys and paths
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Cohere model configuration
COHERE_MODEL = "embed-english-v3.0"  # Default embedding model for Cohere

# Validate required API keys
missing_keys = []
if not ASSEMBLYAI_API_KEY:
    missing_keys.append("ASSEMBLYAI_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    print(f"[ERROR] Missing required environment variables: {', '.join(missing_keys)}")
    print("[INFO] Please run: python setup_env.py")
    print("[INFO] Or create a .env file with the required API keys")
    print("[INFO] See README.md for setup instructions")
    # Don't exit here, let the app start but warn about missing functionality

CHROMA_BASE_PATH = "./Chroma"
DATA_PATH = "."  # Directory for Markdown file

# Set AssemblyAI API key
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
else:
    print("[WARNING] ASSEMBLYAI_API_KEY not set - transcription features will not work")

# PROMPT_TEMPLATE for general QA with conversation context
PROMPT_TEMPLATE = """
You are an AI assistant helping users explore and understand video content. You have access to the video transcript and the conversation history.

CRITICAL INSTRUCTIONS:
- You MUST answer based on the video transcript provided below
- The transcript contains the actual content from the video
- Do NOT say you don't have access to the video or ask for the video link
- Use ONLY the information from the transcript to answer questions
- If the question is about the video content, reference specific parts of the transcript
- If information is not in the transcript, say "Based on the transcript, I don't see information about [topic]"
- Be direct and concise in your responses

VIDEO TRANSCRIPT:
{context}

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: {question}

Answer based on the video transcript above:"""

JOBS_DIR = os.path.join(os.path.dirname(__file__), 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Topic Segmentation and Labeling
segmenter_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq client only if API key is available
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("[WARNING] GROQ_API_KEY not found - some features may not work")

# Initialize Whisper model for fallback transcription
whisper_model = None

def get_whisper_model():
    """Get or initialize Whisper model"""
    global whisper_model
    if whisper_model is None:
        print("[INFO] Loading Whisper model for fallback transcription...")
        # Use base model for faster processing, can be changed to 'small', 'medium', 'large' for better accuracy
        whisper_model = whisper.load_model("base")
        print("[INFO] Whisper model loaded successfully")
    return whisper_model

def save_job(job_id, data):
    with open(os.path.join(JOBS_DIR, f'{job_id}.json'), 'w') as f:
        json.dump(data, f)

def load_job(job_id):
    try:
        with open(os.path.join(JOBS_DIR, f'{job_id}.json'), 'r') as f:
            return json.load(f)
    except Exception:
        return None

def download_audio_only(video_link, job_id):
    try:
        print(f"[{job_id}] Starting download for video: {video_link}")
        yt = YouTube(video_link)
        
        audio_stream = yt.streams.get_audio_only()
        audio_mp4_path = os.path.join(JOBS_DIR, f"{job_id}.mp4")
        audio_stream.download(output_path=JOBS_DIR, filename=f"{job_id}.mp4")
        print(f"[{job_id}] Audio downloaded successfully to {audio_mp4_path}.")

        audio_mp3_path = os.path.join(JOBS_DIR, f"{job_id}.mp3")
        clip = mp.AudioFileClip(audio_mp4_path)
        clip.write_audiofile(audio_mp3_path)
        print(f"[{job_id}] Audio converted to {audio_mp3_path}.")
        os.remove(audio_mp4_path)
        return audio_mp3_path
    except Exception as e:
        print(f"[{job_id}] [ERROR] Failed to download or convert audio: {e}")
        return None

def transcribe_audio_file(audio_path, job_id):
    try:
        print(f"[{job_id}] Starting transcription for {audio_path}")
        
        # Get audio duration to estimate processing time
        try:
            clip = mp.AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            print(f"[{job_id}] Audio duration: {duration:.2f} seconds")
            
            # Warn if video is very long (over 2 hours)
            if duration > 7200:  # 2 hours
                print(f"[{job_id}] [WARNING] Very long video detected ({duration:.2f}s). This may take a while.")
        except Exception as e:
            print(f"[{job_id}] [WARNING] Could not determine audio duration: {e}")
            duration = None
        
        # Try AssemblyAI first for shorter videos, or if it's a long video, try both
        use_assemblyai = duration is None or duration <= 1800  # Use AssemblyAI for videos under 30 minutes
        
        if use_assemblyai:
            print(f"[{job_id}] Attempting AssemblyAI transcription...")
            try:
                # Configure transcriber with basic settings
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(
                    speaker_labels=True,
                    auto_chapters=True,
                    punctuate=True
                )
                
                # Set timeout based on video duration
                timeout_seconds = 600 if duration and duration > 1800 else 300
                
                # Use threading to implement timeout
                result = [None]
                exception = [None]
                
                def transcribe_with_timeout():
                    try:
                        result[0] = transcriber.transcribe(audio_path, config=config)
                    except Exception as e:
                        exception[0] = e
                
                # Start transcription in a separate thread
                thread = threading.Thread(target=transcribe_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout_seconds)
                
                if thread.is_alive():
                    print(f"[{job_id}] [ERROR] AssemblyAI transcription timed out after {timeout_seconds} seconds")
                    raise Exception(f"AssemblyAI transcription timed out after {timeout_seconds} seconds")
                
                if exception[0]:
                    raise exception[0]
                
                transcript = result[0]
                
                # Check transcription status
                if transcript.status == aai.TranscriptStatus.error:
                    print(f"[{job_id}] [ERROR] AssemblyAI transcription failed: {transcript.error}")
                    raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
                
                if transcript.status == aai.TranscriptStatus.completed:
                    transcription_text = transcript.text
                    
                    timestamped_transcription = []
                    for word_info in transcript.words:
                        timestamped_transcription.append({
                            "start_time": word_info.start / 1000.0,
                            "end_time": word_info.end / 1000.0,
                            "text": word_info.text
                        })
                    print(f"[{job_id}] AssemblyAI transcription completed successfully.")
                    return transcription_text, timestamped_transcription
                else:
                    print(f"[{job_id}] [ERROR] AssemblyAI transcription status: {transcript.status}")
                    raise Exception(f"AssemblyAI transcription status: {transcript.status}")
                    
            except Exception as e:
                print(f"[{job_id}] [WARNING] AssemblyAI transcription failed: {e}")
                if duration and duration > 1800:
                    print(f"[{job_id}] [INFO] Long video detected, trying Whisper fallback...")
                else:
                    print(f"[{job_id}] [INFO] Trying Whisper fallback...")
        
        # Use Whisper as fallback or for long videos
        print(f"[{job_id}] Using Whisper for transcription...")
        return transcribe_with_whisper(audio_path, job_id)
            
    except Exception as e:
        print(f"[{job_id}] [ERROR] All transcription methods failed: {e}")
        return None, None

def transcribe_with_whisper(audio_path, job_id):
    """Transcribe audio using Whisper as fallback"""
    try:
        print(f"[{job_id}] Starting Whisper transcription for {audio_path}")
        
        # Update job status to show we're using Whisper
        job = load_job(job_id)
        if job:
            job['status'] = 'transcribing'
            job['progress'] = 35  # Slightly lower than AssemblyAI since Whisper takes longer
            save_job(job_id, job)
        
        model = get_whisper_model()
        
        # Update progress to show model is loaded
        if job:
            job['progress'] = 40
            save_job(job_id, job)
        
        # Transcribe with Whisper
        result = model.transcribe(audio_path, word_timestamps=True)
        
        # Update progress during transcription
        if job:
            job['progress'] = 55
            save_job(job_id, job)
        
        transcription_text = result["text"]
        
        # Extract word-level timestamps
        timestamped_transcription = []
        if "segments" in result:
            for segment in result["segments"]:
                for word_info in segment.get("words", []):
                    timestamped_transcription.append({
                        "start_time": word_info["start"],
                        "end_time": word_info["end"],
                        "text": word_info["word"]
                    })
        else:
            # Fallback if no word timestamps
            timestamped_transcription.append({
                "start_time": 0,
                "end_time": 0,
                "text": transcription_text
            })
        
        # Update progress after successful transcription
        if job:
            job['progress'] = 60
            save_job(job_id, job)
        
        print(f"[{job_id}] Whisper transcription completed successfully.")
        return transcription_text, timestamped_transcription
        
    except Exception as e:
        print(f"[{job_id}] [ERROR] Whisper transcription failed: {e}")
        return None, None

def embed_transcription(timestamps, job_id):
    try:
        print(f"[{job_id}] Starting embedding process.")
        # Create a unique ChromaDB path for this job
        chroma_path = os.path.join(CHROMA_BASE_PATH, job_id)
        os.makedirs(chroma_path, exist_ok=True)

        # Prepare documents for ChromaDB
        documents = []
        for i, ts in enumerate(timestamps):
            doc = Document(
                page_content=ts["text"],
                metadata={
                    "source": f"video_segment_{i}",
                    "start_time": ts["start_time"],
                    "end_time": ts["end_time"],
                    "job_id": job_id
                }
            )
            documents.append(doc)

        # Initialize embeddings (using CohereEmbeddings if COHERE_API_KEY is available, else HuggingFace)
        if COHERE_API_KEY:
            try:
                embeddings = CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model=COHERE_MODEL
                )
                print(f"[{job_id}] Using CohereEmbeddings for RAG.")
            except Exception as e:
                print(f"[{job_id}] [WARNING] CohereEmbeddings failed: {e}")
                print(f"[{job_id}] Falling back to HuggingFaceEmbeddings")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print(f"[{job_id}] Using HuggingFaceEmbeddings for RAG.")

        # Create and persist the Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_path
        )
        vectorstore.persist()
        print(f"[{job_id}] Embedding completed and stored in {chroma_path}.")
        return True
    except Exception as e:
        print(f"[{job_id}] [ERROR] Embedding failed: {e}")
        return False

# Helper to split word-level timestamps into sentences
def split_words_into_sentences(word_timestamps):
    sentences = []
    current_sentence_words = []
    current_sentence_start_time = None

    for i, word_info in enumerate(word_timestamps):
        word = word_info["text"]
        start_time = word_info["start_time"]
        end_time = word_info["end_time"]

        if not current_sentence_words:
            current_sentence_start_time = start_time

        current_sentence_words.append(word)

        # Simple heuristic for sentence end: punctuation or end of transcript
        # Also ensure sentence is not empty after stripping punctuation
        if (word.endswith('.') or word.endswith('?') or word.endswith('!') or i == len(word_timestamps) - 1) and current_sentence_words:
            full_sentence_text = " ".join(current_sentence_words)
            sentences.append({
                "text": full_sentence_text,
                "start_time": current_sentence_start_time,
                "end_time": end_time
            })
            current_sentence_words = []
            current_sentence_start_time = None
    
    # Add any remaining words as a sentence if the transcript ends without punctuation
    if current_sentence_words:
        full_sentence_text = " ".join(current_sentence_words)
        sentences.append({
            "text": full_sentence_text,
            "start_time": current_sentence_start_time or word_timestamps[0]["start_time"],
            "end_time": word_timestamps[-1]["end_time"]
        })
    return sentences

# Segment transcript into topics
def segment_transcript(sentences_data, threshold=0.5, min_size=10):
    sentence_texts = [s["text"] for s in sentences_data]
    embeddings = segmenter_model.encode(sentence_texts)
    similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold and (i + 1 - boundaries[-1]) >= min_size:
            boundaries.append(i + 1)
    boundaries.append(len(sentences_data))
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segments.append({
            "start_idx": start,
            "end_idx": end,
            "sentence_data_indices": list(range(start, end))
        })
    return segments

def label_topic_with_groq(text):
    if not GROQ_API_KEY:
        return "Topic"
    prompt = f"""
    Given the following transcript section, generate a concise, descriptive topic title (max 7 words):
    ---
    {text}
    ---
    Title:
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.3
        )
        title = completion.choices[0].message.content.strip().replace('"', '')
        return title
    except Exception as e:
        print(f"[ERROR] Groq topic labeling failed: {e}")
        return "Topic"

def segment_and_label_topics(transcription, timestamps):
    # First, split word-level timestamps into sentences
    sentence_data = split_words_into_sentences(timestamps)
    
    # Segment these sentences into topics
    segments = segment_transcript(sentence_data) # segment_transcript now operates on sentence_data
    
    topics = []
    for seg_info in segments:
        # Collect sentences for this topic
        topic_sentences_data = [sentence_data[i] for i in seg_info["sentence_data_indices"]]
        
        # Combine sentence texts for LLM labeling
        seg_text_for_llm = " ".join([s["text"] for s in topic_sentences_data])
        
        # Label the topic
        title = label_topic_with_groq(seg_text_for_llm)

        # Determine overall start/end time of the topic
        topic_start_time = topic_sentences_data[0]["start_time"]
        topic_end_time = topic_sentences_data[-1]["end_time"]

        topics.append({
            "title": title,
            "start_time": topic_start_time,
            "end_time": topic_end_time,
            "text": seg_text_for_llm, # The full text of the topic
            "segments": topic_sentences_data # These are now sentence-level segments
        })
    return topics

def transcribe_job_thread(job_id, video_url):
    job = load_job(job_id)
    
    try:
        # Download phase
        job['status'] = 'downloading'
        job['progress'] = 10
        save_job(job_id, job)
        print(f"[{job_id}] Starting download...")
        
        audio_path = download_audio_only(video_url, job_id)
        if not audio_path:
            job['status'] = 'failed'
            job['error'] = 'Failed to download audio. Please check the video URL and try again.'
            save_job(job_id, job)
            return
        
        # Transcription phase
        job['status'] = 'transcribing'
        job['progress'] = 30
        save_job(job_id, job)
        print(f"[{job_id}] Starting transcription...")
        
        transcription, timestamps = transcribe_audio_file(audio_path, job_id)
        if not transcription:
            job['status'] = 'failed'
            # Check if it's a timeout error
            if job.get('duration', 0) > 1800:  # Longer than 30 minutes
                job['error'] = 'Transcription timed out. This video is quite long. Please try a shorter video (under 30 minutes) for better results.'
            else:
                job['error'] = 'Transcription failed. This may be due to audio issues or network problems. Please try again or check the video URL.'
            save_job(job_id, job)
            return
        
        job['transcription'] = transcription
        job['timestamps'] = timestamps
        job['progress'] = 60
        save_job(job_id, job)
        print(f"[{job_id}] Transcription completed, starting topic segmentation...")

        # Topic segmentation and labeling
        job['status'] = 'segmenting'
        job['progress'] = 75
        save_job(job_id, job)
        
        topics = segment_and_label_topics(transcription, timestamps)
        job['topics'] = topics
        job['progress'] = 85
        save_job(job_id, job)
        print(f"[{job_id}] Topic segmentation completed, starting embedding...")

        # Embedding transcription into ChromaDB (for RAG)
        job['status'] = 'embedding'
        job['progress'] = 90
        save_job(job_id, job)
        
        # Pass the sentence-level timestamps for more granular embedding
        if not embed_transcription(split_words_into_sentences(timestamps), job_id):
            job['status'] = 'failed'
            job['error'] = 'Failed to embed transcription for Q&A functionality.'
            save_job(job_id, job)
            return
        
        # Update progress after embedding
        job['progress'] = 95
        save_job(job_id, job)

        # Completion
        job['status'] = 'completed'
        job['progress'] = 100
        job['completed_at'] = time.time()
        save_job(job_id, job)
        
        # Save transcription to text file
        with open(os.path.join(JOBS_DIR, f'{job_id}.txt'), 'w') as f:
            f.write(transcription)

        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        print(f"[{job_id}] Job completed successfully.")
        
    except Exception as e:
        print(f"[{job_id}] [ERROR] Job failed with exception: {e}")
        job['status'] = 'failed'
        job['error'] = f'Unexpected error: {str(e)}'
        save_job(job_id, job)

def load_documents():
    print("[INFO] Loading documents from directory.")
    markdown_path = os.path.join(DATA_PATH, "transcription.md")
    loader = UnstructuredMarkdownLoader(markdown_path)
    document = loader.load()
    print(f"[INFO] Loaded {len(document)} documents.")
    return document

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Generated {len(chunks)} chunks.")
    return chunks

def generate_video_reel(video_path, highlights, output_path):
    try:
        print(f"[INFO] Generating video reel from {video_path}")
        # Load the original video
        video = mp.VideoFileClip(video_path)
        clips = []
        
        # Create clips based on the highlights (start and end times)
        for highlight in highlights:
            start_time = highlight['start']
            end_time = highlight['end']
            
            # Ensure times are within video bounds
            start_time = max(0, min(start_time, video.duration))
            end_time = max(start_time + 1, min(end_time, video.duration))
            
            clip = video.subclip(start_time, end_time)
            
            # Add captions to this clip
            if highlight.get('captions'):
                caption_text = highlight['captions'][0][0] if highlight['captions'] else "Highlight"
                try:
                    # Create text clip with caption
                    text_clip = mp.TextClip(
                        caption_text, 
                        fontsize=24, 
                        color='white', 
                        bg_color='black',
                        size=(clip.w, None),
                        method='caption'
                    )
                    text_clip = text_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
                    
                    # Composite the video clip with text
                    clip = mp.CompositeVideoClip([clip, text_clip])
                except Exception as e:
                    print(f"[WARNING] Could not add caption to clip: {e}")
            
            clips.append(clip)

        if not clips:
            print("[ERROR] No valid clips to create reel")
            return None

        # Concatenate clips into one reel
        final_reel = mp.concatenate_videoclips(clips)
        
        # Write the final video reel
        final_reel.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # Clean up
        video.close()
        final_reel.close()
        for clip in clips:
            clip.close()
        
        print(f"[INFO] Video reel saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Failed to generate video reel: {e}")
        return None

def extract_highlights_from_transcription(transcription):
    """
    Extracts highlights from transcription based on NLP techniques and content analysis.
    Uses sentence splitting and importance scoring to identify key moments.
    """
    highlights = []
    
    # Split transcription into sentences
    sentences = []
    current_sentence = ""
    current_start_time = 0
    
    # Simple sentence splitting (can be improved with NLTK)
    words = transcription.split()
    for i, word in enumerate(words):
        current_sentence += word + " "
        
        # End sentence on punctuation or every 15-20 words
        if (word.endswith('.') or word.endswith('?') or word.endswith('!') or 
            len(current_sentence.split()) >= random.randint(15, 20)):
            
            if current_sentence.strip():
                sentences.append({
                    "text": current_sentence.strip(),
                    "start_time": current_start_time,
                    "end_time": current_start_time + len(current_sentence.split()) * 0.5  # Rough estimate
                })
                current_start_time = current_sentence.split()[-1].count('.') * 0.5 + len(current_sentence.split()) * 0.5
                current_sentence = ""
    
    # Add remaining sentence if any
    if current_sentence.strip():
        sentences.append({
            "text": current_sentence.strip(),
            "start_time": current_start_time,
            "end_time": current_start_time + len(current_sentence.split()) * 0.5
        })
    
    # Select important sentences for highlights
    important_keywords = [
        'introduction', 'important', 'key', 'main', 'primary', 'essential',
        'conclusion', 'summary', 'finally', 'therefore', 'however', 'but',
        'example', 'instance', 'demonstrate', 'show', 'prove', 'evidence',
        'research', 'study', 'analysis', 'result', 'finding', 'discovery'
    ]
    
    selected_sentences = []
    for sentence in sentences:
        # Score sentence based on keywords and length
        score = 0
        text_lower = sentence["text"].lower()
        
        # Check for important keywords
        for keyword in important_keywords:
            if keyword in text_lower:
                score += 2
        
        # Prefer medium-length sentences (not too short, not too long)
        word_count = len(sentence["text"].split())
        if 8 <= word_count <= 25:
            score += 1
        elif word_count > 25:
            score -= 1
        
        # Prefer sentences that start with capital letters (likely important)
        if sentence["text"][0].isupper():
            score += 1
        
        # Add sentence if it meets minimum score
        if score >= 1:
            selected_sentences.append((sentence, score))
    
    # Sort by score and take top highlights
    selected_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = selected_sentences[:min(5, len(selected_sentences))]
    
    # Create highlights from selected sentences
    for i, (sentence, score) in enumerate(top_sentences):
        # Ensure minimum duration and no overlap
        duration = max(3, sentence["end_time"] - sentence["start_time"])
        start_time = sentence["start_time"]
        end_time = start_time + duration
        
        # Create a meaningful caption
        caption = sentence["text"]
        if len(caption) > 100:
            caption = caption[:97] + "..."
        
        highlights.append({
            "start": start_time,
            "end": end_time,
            "captions": [(caption, start_time, end_time)]
        })
    
    # If no highlights found, create some basic ones
    if not highlights:
        total_duration = len(transcription.split()) * 0.5
        num_highlights = min(3, int(total_duration / 10))
        
        for i in range(num_highlights):
            start_time = i * (total_duration / num_highlights)
            end_time = start_time + 5
            highlights.append({
                "start": start_time,
                "end": end_time,
                "captions": [(f"Highlight {i + 1}", start_time, end_time)]
            })
    
    return highlights

# Function to process and query ChromaDB (now focused on querying)
# NOTE: This function's name (process_and_query_chroma) is misleading, 
# as embedding now happens in embed_transcription.
# We will refactor it to `retrieve_context_from_chroma`
def retrieve_context_from_chroma(job_id, query_text):
    try:
        chroma_path = os.path.join(CHROMA_BASE_PATH, job_id)
        if not os.path.exists(chroma_path):
            print(f"[ERROR] ChromaDB path not found for job {job_id}: {chroma_path}")
            return [], "", [] # Return empty results

        # Initialize embeddings (same as used in embed_transcription)
        if COHERE_API_KEY:
            try:
                embeddings = CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model=COHERE_MODEL
                )
            except Exception as e:
                print(f"[WARNING] CohereEmbeddings failed: {e}")
                print("[INFO] Falling back to HuggingFaceEmbeddings")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Try to load the existing Chroma vector store
        try:
            vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
            # Test if the embeddings work with the existing collection
            test_query = "test"
            vectorstore.similarity_search(test_query, k=1)
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"[{job_id}] [WARNING] Embedding dimension mismatch detected. Recreating ChromaDB collection...")
                # Remove the existing collection and recreate it
                import shutil
                try:
                    shutil.rmtree(chroma_path)
                    print(f"[{job_id}] Removed old ChromaDB collection")
                except Exception as cleanup_error:
                    print(f"[{job_id}] [WARNING] Failed to clean up old collection: {cleanup_error}")
                
                # Re-embed the transcription with the current embedding model
                job = load_job(job_id)
                if job and job.get('timestamps'):
                    print(f"[{job_id}] Re-embedding transcription with current model...")
                    success = embed_transcription(split_words_into_sentences(job['timestamps']), job_id)
                    if success:
                        vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
                    else:
                        print(f"[{job_id}] [ERROR] Failed to re-embed transcription")
                        return [], "", []
                else:
                    print(f"[{job_id}] [ERROR] No transcription data available for re-embedding")
                    return [], "", []
            else:
                # Re-raise the error if it's not a dimension issue
                raise e

        # Perform similarity search to retrieve relevant chunks
        # We'll retrieve top 8 relevant chunks (increased from 4)
        retrieved_docs = vectorstore.similarity_search(query_text, k=8)
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = [{
            "text": doc.page_content,
            "start_time": doc.metadata.get("start_time"),
            "end_time": doc.metadata.get("end_time")
        } for doc in retrieved_docs if doc.metadata.get("start_time") is not None]
        
        print(f"[{job_id}] Retrieved {len(retrieved_docs)} documents for query.")
        print(f"[{job_id}] Retrieved context length: {len(context_text)} characters")
        print(f"[{job_id}] Retrieved context preview: {context_text[:300]}...")
        return retrieved_docs, context_text, sources
    except Exception as e:
        print(f"[ERROR] Error retrieving from ChromaDB: {e}")
        return [], "", []

@app.route('/generate_highlights', methods=['POST'])
def generate_highlights():
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
    
    job = load_job(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404
    
    try:
        # Use existing transcription and timestamps to generate highlights
        transcription = job.get('transcription', '')
        timestamps = job.get('timestamps', [])
        
        if not transcription or not timestamps:
            return jsonify({"error": "No transcription data available"}), 400
        
        # Generate highlights using the existing function
        highlights = extract_highlights_from_transcription(transcription)
        
        # Convert to frontend format
        frontend_highlights = []
        for i, highlight in enumerate(highlights):
            frontend_highlights.append({
                "id": str(i + 1),
                "startTime": highlight["start"],
                "endTime": highlight["end"],
                "caption": highlight["captions"][0][0] if highlight["captions"] else f"Highlight {i + 1}",
                "importance": random.uniform(0.7, 0.9)  # Random importance score
            })
        
        return jsonify(frontend_highlights)
        
    except Exception as e:
        print(f"[ERROR] Error generating highlights for job {job_id}: {e}")
        return jsonify({"error": "Failed to generate highlights"}), 500

@app.route('/export_reel', methods=['POST'])
def export_reel():
    data = request.json
    job_id = data.get('job_id')
    highlights = data.get('highlights', [])
    
    if not job_id or not highlights:
        return jsonify({"error": "Job ID and highlights are required"}), 400
    
    job = load_job(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404
    
    try:
        # Download the full video if not already available
        video_path = os.path.join(JOBS_DIR, f"{job_id}_full.mp4")
        if not os.path.exists(video_path):
            print(f"[{job_id}] Downloading full video for reel generation...")
            yt = YouTube(job['video_url'])
            video_stream = yt.streams.get_highest_resolution()
            video_stream.download(output_path=JOBS_DIR, filename=f"{job_id}_full.mp4")
        
        # Convert frontend highlights to backend format
        backend_highlights = []
        for highlight in highlights:
            backend_highlights.append({
                "start": highlight["startTime"],
                "end": highlight["endTime"],
                "captions": [(highlight["caption"], highlight["startTime"], highlight["endTime"])]
            })
        
        # Generate video reel
        reel_filename = f"reel_{job_id}_{int(time.time())}.mp4"
        reel_output_path = os.path.join(JOBS_DIR, reel_filename)
        
        result = generate_video_reel(video_path, backend_highlights, reel_output_path)
        
        if result is None:
            return jsonify({"error": "Failed to generate video reel"}), 500
        
        # Return the URL to the generated reel
        reel_url = f"/download_reel/{reel_filename}"
        
        return jsonify({
            "message": "Video reel created successfully",
            "url": reel_url,
            "filename": reel_filename
        })
        
    except Exception as e:
        print(f"[ERROR] Error exporting reel for job {job_id}: {e}")
        return jsonify({"error": "Failed to export reel"}), 500

@app.route('/download_reel/<filename>', methods=['GET'])
def download_reel(filename):
    try:
        reel_path = os.path.join(JOBS_DIR, filename)
        if not os.path.exists(reel_path):
            return jsonify({"error": "Reel not found"}), 404
        
        return send_file(reel_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        print(f"[ERROR] Error downloading reel {filename}: {e}")
        return jsonify({"error": "Failed to download reel"}), 500

@app.route('/create_reel', methods=['POST'])
def create_reel():
    data = request.json
    video_url = data.get('videoUrl')

    try:
        # Download and transcribe video
        transcription = download_and_transcribe_audio(video_url)
        if transcription is None:
            return jsonify({"error": "Failed to transcribe video"}), 500

        # Save transcription to markdown
        transcription_path = os.path.join(DATA_PATH, "transcription.md")
        with open(transcription_path, "w") as f:
            f.write(transcription)

        # Extract highlights based on the transcription
        highlights = extract_highlights_from_transcription(transcription)

        # Download the full video to generate clips
        yt = YouTube(video_url)
        video_stream = yt.streams.get_highest_resolution()
        video_stream.download(filename="full_video.mp4")

        # Generate video reel
        reel_filename = f"reel_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.mp4"
        reel_output_path = os.path.join(DATA_PATH, reel_filename)

        generate_video_reel("full_video.mp4", highlights, reel_output_path)

        return jsonify({"message": "Video reel created successfully", "reel_url": reel_output_path})
    except Exception as e:
        print(f"Error creating video reel: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


            
# Endpoint to transcribe video
@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    video_url = data.get('videoUrl') or data.get('video_url')
    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400
    
    try:
        yt = YouTube(video_url)
        video_title = yt.title
        video_thumbnail = yt.thumbnail_url
        video_duration = yt.length
        
        # Check video duration and warn about very long videos
        if video_duration > 14400:  # 4 hours - still too long even for Whisper
            return jsonify({
                'error': 'Video too long',
                'message': f'This video is {video_duration // 60} minutes long. Videos longer than 4 hours may cause issues. Please try a shorter video.',
                'duration': video_duration
            }), 400
        elif video_duration > 7200:  # 2 hours
            print(f"[WARNING] Very long video detected: {video_duration // 60} minutes - will use local transcription")
        elif video_duration > 3600:  # 1 hour
            print(f"[WARNING] Long video detected: {video_duration // 60} minutes")
        elif video_duration > 1800:  # 30 minutes
            print(f"[INFO] Medium-length video detected: {video_duration // 60} minutes - may take longer to process")
            
    except Exception as e:
        print(f"[ERROR] Could not fetch YouTube metadata: {e}")
        return jsonify({'error': 'Invalid YouTube URL or could not fetch video metadata.'}), 400

    job_id = str(uuid.uuid4())
    job = {
        'id': job_id,
        'status': 'queued',
        'progress': 0,
        'video_url': video_url,
        'title': video_title,
        'thumbnail': video_thumbnail,
        'duration': video_duration,
        'created_at': time.time(),
        'transcription': None,
        'timestamps': [],
        'conversation_history': [],  # Add conversation history
        'error': None
    }
    save_job(job_id, job)
    threading.Thread(target=transcribe_job_thread, args=(job_id, video_url), daemon=True).start()
    print(f"Created job {job_id} for video {video_url}")
    return jsonify(job)

@app.route('/transcribe/<job_id>', methods=['GET'])
def get_transcription_job(job_id):
    job = load_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

# Endpoint to query transcription for answers
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    job_id = data.get('job_id')
    question = data.get('question')

    if not job_id or not question:
        return jsonify({"error": "Job ID and question are required"}), 400

    job = load_job(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    try:
        # 1. Retrieve context from job-specific ChromaDB
        retrieved_docs, context_text, sources = retrieve_context_from_chroma(job_id, question)
        
        # Fallback: If context is too short, use full transcript or concatenated topic texts
        if not context_text or len(context_text.split()) < 40:
            print(f"[{job_id}] Fallback: Using full transcript or topics as context.")
            print(f"[{job_id}] Original context length: {len(context_text.split()) if context_text else 0} words")
            
            # Prefer concatenated topic texts if available
            topics = job.get('topics')
            if topics and isinstance(topics, list):
                context_text = "\n\n---\n\n".join([t.get('text', '') for t in topics if t.get('text')])
                print(f"[{job_id}] Using topics as fallback, {len(topics)} topics found")
            else:
                context_text = job.get('transcription', '')
                print(f"[{job_id}] Using full transcription as fallback, length: {len(context_text)} characters")

        if not context_text:
            print(f"[{job_id}] ERROR: No context available at all!")
            return jsonify({"response": "I could not find relevant information in the transcript to answer your question."})
        
        print(f"[{job_id}] Final context length: {len(context_text.split())} words")

        # 2. Prepare conversation history
        conversation_history = job.get('conversation_history', [])
        conversation_text = ""
        
        if conversation_history:
            # Format conversation history for the prompt
            conversation_lines = []
            for i, msg in enumerate(conversation_history[-10:]):  # Keep last 10 messages to avoid token limits
                if msg.get('role') == 'user':
                    conversation_lines.append(f"User: {msg.get('content', '')}")
                elif msg.get('role') == 'assistant':
                    conversation_lines.append(f"Assistant: {msg.get('content', '')}")
            
            conversation_text = "\n".join(conversation_lines)
        else:
            conversation_text = "No previous conversation."

        # 3. Check if it's a simple greeting (but be more restrictive)
        greeting_keywords = ['hello', 'hey', 'hi', 'good morning', 'good afternoon', 'good evening', 'greetings']
        is_greeting = any(keyword in question.lower() for keyword in greeting_keywords) and len(question.split()) <= 3
        
        # 4. Prepare prompt for LLM (Groq)
        if is_greeting:
            # Use a simpler, more friendly prompt for greetings
            greeting_prompt = f"""You are a helpful AI assistant for video content. The user said: "{question}"

Respond briefly and warmly. Mention you're here to help with questions about the video content.

Response:"""
            prompt = greeting_prompt
        else:
            # Use the full prompt for content questions
            prompt = PROMPT_TEMPLATE.format(
                context=context_text, 
                conversation_history=conversation_text,
                question=question
            )
        
        # Debug logging to see what context is being passed
        context_length = len(context_text.split()) if context_text else 0
        context_sample = context_text[:200] + "..." if context_text and len(context_text) > 200 else context_text
        print(f"[{job_id}] Context length: {context_length} words")
        print(f"[{job_id}] Context sample: {context_sample}")
        print(f"[{job_id}] Question: {question}")
        print(f"[{job_id}] Is greeting: {is_greeting}")
        
        # Log the full prompt for debugging
        if not is_greeting:
            print(f"[{job_id}] Full prompt preview: {prompt[:500]}...")
        
        print(f"[{job_id}] Prompt sent to LLM with conversation history")

        # 4. Generate answer using Groq Llama-3
        if is_greeting:
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            # Use system message to reinforce instructions
            messages = [
                {"role": "system", "content": "You are an AI assistant that answers questions based on video transcripts. Always use the provided transcript content to answer questions."},
                {"role": "user", "content": prompt}
            ]
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # Using Groq's fastest model for quick responses
            messages=messages,
            max_tokens=1024, # Increased max tokens for fuller answers
            temperature=0.7
        )
        answer = completion.choices[0].message.content.strip()
        
        # 5. Update conversation history
        conversation_history.append({
            'role': 'user',
            'content': question,
            'timestamp': time.time()
        })
        conversation_history.append({
            'role': 'assistant',
            'content': answer,
            'timestamp': time.time()
        })
        
        # Keep only last 20 messages to prevent memory issues
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        job['conversation_history'] = conversation_history
        save_job(job_id, job)
        
        response_data = {"response": answer}
        if sources: # Include sources if available
            response_data["sources"] = sources

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] Error during query processing for job {job_id}: {e}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a specific job"""
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
    
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    try:
        # Clear conversation history
        job['conversation_history'] = []
        save_job(job_id, job)
        
        return jsonify({"message": "Conversation history cleared successfully"})
        
    except Exception as e:
        print(f"[ERROR] Error clearing conversation for job {job_id}: {e}")
        return jsonify({"error": "Failed to clear conversation"}), 500

@app.route('/library', methods=['GET'])
def get_library():
    """Get recent completed jobs for the library"""
    try:
        # Get all job files from the jobs directory
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.json"))
        jobs = []
        
        for job_file in job_files:
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                    
                # Only include completed jobs
                if job_data.get('status') == 'completed':
                    # Add file path for potential deletion
                    job_data['file_path'] = job_file
                    jobs.append(job_data)
            except Exception as e:
                print(f"[ERROR] Failed to load job file {job_file}: {e}")
                continue
        
        # Sort by creation time (newest first) and limit to 20 jobs
        jobs.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        jobs = jobs[:20]  # Limit to 20 most recent jobs
        
        # Remove file_path from response (it's internal)
        for job in jobs:
            job.pop('file_path', None)
        
        return jsonify(jobs)
        
    except Exception as e:
        print(f"[ERROR] Error fetching library: {e}")
        return jsonify({"error": "Failed to fetch library"}), 500

@app.route('/debug_job/<job_id>', methods=['GET'])
def debug_job(job_id):
    """Debug endpoint to check job data"""
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    debug_info = {
        "job_id": job_id,
        "status": job.get('status'),
        "has_transcription": bool(job.get('transcription')),
        "transcription_length": len(job.get('transcription', '')),
        "has_timestamps": bool(job.get('timestamps')),
        "timestamps_count": len(job.get('timestamps', [])),
        "has_topics": bool(job.get('topics')),
        "topics_count": len(job.get('topics', [])),
        "chroma_exists": os.path.exists(os.path.join(CHROMA_BASE_PATH, job_id)),
        "transcription_preview": job.get('transcription', '')[:500] + "..." if job.get('transcription') else None
    }
    
    return jsonify(debug_info)

@app.route('/delete_job', methods=['POST'])
def delete_job():
    """Delete a job and its associated files"""
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
    
    try:
        # Delete job file
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            os.remove(job_file)
        
        # Delete associated audio/video files
        audio_file = os.path.join(JOBS_DIR, f"{job_id}.mp3")
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        video_file = os.path.join(JOBS_DIR, f"{job_id}_full.mp4")
        if os.path.exists(video_file):
            os.remove(video_file)
        
        # Delete ChromaDB directory
        chroma_dir = os.path.join(CHROMA_BASE_PATH, job_id)
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        
        # Delete any reel files associated with this job
        reel_files = glob.glob(os.path.join(JOBS_DIR, f"reel_{job_id}_*.mp4"))
        for reel_file in reel_files:
            os.remove(reel_file)
        
        return jsonify({"message": "Job deleted successfully"})
        
    except Exception as e:
        print(f"[ERROR] Error deleting job {job_id}: {e}")
        return jsonify({"error": "Failed to delete job"}), 500

if __name__ == '__main__':
    # Ensure Chroma path exists
    if not os.path.exists(CHROMA_BASE_PATH):
        os.makedirs(CHROMA_BASE_PATH)

    app.run(port=5000)
