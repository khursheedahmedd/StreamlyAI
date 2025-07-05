# StreamlyAI

A powerful AI-powered video transcription and analysis platform that transforms YouTube videos into interactive, searchable content.

## Features

- 🎥 **Smart Transcription**: AI-powered transcription with speaker identification and timestamps
- 🔍 **Search & Navigate**: Find any moment in your video with full-text search capabilities
- 💬 **Ask Questions**: Chat with your video content using our AI assistant
- ✂️ **Auto Highlights**: Generate shareable highlight reels automatically
- 🎯 **Topic Segmentation**: Automatically identify and segment video topics
- 📊 **Interactive Library**: Manage all your transcribed videos in one place

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- API keys for the following services:
  - AssemblyAI (for transcription)
  - Groq (for topic labeling and Q&A)
  - Cohere (optional, for better embeddings)

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd StreamlyAI
   ```

2. **Set up environment variables**

   **Option A: Use the setup script (Recommended)**

   ```bash
   cd server
   python setup_env.py
   ```

   **Option B: Create manually**

   Create a `.env` file in the `server` directory with the following variables:

   ```env
   # Required API Keys
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here

   # Optional API Keys
   COHERE_API_KEY=your_cohere_api_key_here

   # Flask Configuration
   FLASK_ENV=development
   FLASK_DEBUG=True
   ```

3. **Get API Keys**

   - **AssemblyAI**: Sign up at [assemblyai.com](https://www.assemblyai.com/) and get your API key
   - **Groq**: Sign up at [console.groq.com](https://console.groq.com/) and get your API key
   - **Cohere**: Sign up at [cohere.ai](https://cohere.ai/) and get your API key (optional)

### Backend Setup

1. **Install Python dependencies**

   ```bash
   cd server
   pip install -r requirements.txt
   ```

2. **Run the backend server**

   ```bash
   python backend.py
   ```

   The server will start on `http://localhost:5000`

### Frontend Setup

1. **Install Node.js dependencies**

   ```bash
   cd client
   npm install
   ```

2. **Run the development server**

   ```bash
   npm run dev
   ```

   The frontend will start on `http://localhost:5173`

## Usage

1. **Add a YouTube Video**: Paste any YouTube URL in the input field
2. **Wait for Processing**: The system will download, transcribe, and analyze your video
3. **Explore Content**: Search through the transcript, ask questions, or create highlights
4. **Generate Reels**: Create shareable highlight reels automatically

## API Endpoints

### Core Endpoints

- `POST /transcribe` - Start transcription of a YouTube video
- `GET /transcribe/<job_id>` - Get transcription job status
- `POST /query` - Ask questions about video content
- `GET /library` - Get all transcribed videos
- `POST /generate_highlights` - Generate video highlights
- `POST /export_reel` - Export highlight reel as video

### Example Usage

```bash
# Start transcription
curl -X POST http://localhost:5000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=example"}'

# Ask a question
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"job_id": "your_job_id", "question": "What are the main topics discussed?"}'
```

## Architecture

### Backend (Python/Flask)

- **AssemblyAI**: High-quality speech-to-text transcription
- **Groq**: Fast LLM for topic labeling and Q&A
- **ChromaDB**: Vector database for semantic search
- **Whisper**: Fallback transcription for long videos
- **MoviePy**: Video processing and reel generation

### Frontend (React/TypeScript)

- **React 18**: Modern UI framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Framer Motion**: Smooth animations
- **Shadcn/ui**: Beautiful component library

## Security

- All API keys are stored in environment variables
- No sensitive data is committed to version control
- Large media files are excluded from git tracking
- Input validation and sanitization implemented

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License & Usage

**All code, models, and assets in this repository are the exclusive property of the Privify team.**

> **No one is permitted to use, copy, modify, distribute, or sublicense any part of this codebase, in whole or in part, for any purpose without the explicit written consent of the authors. Unauthorized use is strictly prohibited and may result in legal action.**

For licensing, partnership, or commercial inquiries, please contact:  
**[khursheed6577@gmail.com]**

## Troubleshooting

### ChromaDB Dimension Mismatch Error

If you see an error like `Embedding dimension 1024 does not match collection dimensionality 384`, this means you have old ChromaDB collections created with a different embedding model. The system will automatically handle this by:

1. **Automatic Recovery**: The system will detect the mismatch and automatically recreate the ChromaDB collection
2. **Re-embedding**: It will re-embed the transcription with the current embedding model
3. **Fallback**: If re-embedding fails, it will use the full transcript for Q&A

### Manual Cleanup (Optional)

If you want to start fresh with all ChromaDB collections:

```bash
cd server
python cleanup_chroma.py check    # Check current status
python cleanup_chroma.py cleanup  # Remove all collections
```

This will:

- Backup your job metadata
- Remove all ChromaDB collections
- Allow you to start fresh with the correct embedding dimensions

### Environment Variables Not Working

If you're getting API key errors:

1. **Check your `.env` file**:

   ```bash
   cd server
   python setup_env.py validate
   ```

2. **Recreate your environment**:

   ```bash
   cd server
   python setup_env.py
   ```

3. **Verify API keys are valid** by testing them in their respective dashboards

## Support

If you encounter any issues or have questions, please open an issue on GitHub or contact the development team.
