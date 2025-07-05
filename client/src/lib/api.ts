// Mock API for StreamlyAI

export type JobStatus = 'queued' | 'downloading' | 'transcribing' | 'segmenting' | 'embedding' | 'completed' | 'failed';

export interface TranscriptionJob {
  id: string;
  status: JobStatus;
  progress: number;
  title?: string;
  thumbnail?: string;
  duration?: number;
  video_url: string;
  transcription?: string;
  timestamps?: { start_time: number; end_time: number; text: string; }[];
  topics?: {
    title: string;
    start_time: number;
    end_time: number;
    text: string;
    segments: { start_time: number; end_time: number; text: string; }[];
  }[];
  created_at: number;
  error?: string | null;
}

export interface TranscriptSegment {
  id: string;
  text: string;
  start_time: number;
  end_time: number;
  speaker?: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  sources?: TranscriptSegment[];
}

export interface Highlight {
  id: string;
  startTime: number;
  endTime: number;
  caption: string;
  importance: number;
}

// Mock data store
const mockJobs: Record<string, TranscriptionJob> = {};
const mockTranscripts: Record<string, TranscriptSegment[]> = {};
const mockHighlights: Record<string, Highlight[]> = {};

// YouTube URL validation
export function isValidYouTubeUrl(url: string): boolean {
  const patterns = [
    /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
    /^https?:\/\/(www\.)?youtu\.be\/[\w-]+/,
    /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]+/
  ];
  return patterns.some(pattern => pattern.test(url));
}

// Extract video ID from YouTube URL
export function extractVideoId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/
  ];
  
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

// API Functions with mock implementations
export const api = {
  // Start transcription job (real backend call)
  async transcribe(videoUrl: string): Promise<TranscriptionJob> {
    const response = await fetch("/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoUrl })
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.message || errorData.error || "Failed to start transcription";
      throw new Error(errorMessage);
    }
    
    return await response.json();
  },

  // Get job status (real backend call)
  async getStatus(id: string): Promise<TranscriptionJob> {
    const response = await fetch(`/transcribe/${id}`);
    if (!response.ok) throw new Error("Failed to get job status");
    return await response.json();
  },

  // Get transcript
  async getTranscript(id: string): Promise<TranscriptSegment[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (!mockTranscripts[id]) {
      // Generate mock transcript
      mockTranscripts[id] = [
        {
          id: '1',
          text: 'Welcome to this amazing video where we explore the latest in AI technology.',
          start_time: 0,
          end_time: 4.5,
          speaker: 'Host'
        },
        {
          id: '2',
          text: 'Today we\'ll be discussing how artificial intelligence is transforming various industries.',
          start_time: 4.5,
          end_time: 9.2,
          speaker: 'Host'
        },
        {
          id: '3',
          text: 'From healthcare to finance, AI is making significant impacts everywhere.',
          start_time: 9.2,
          end_time: 13.8,
          speaker: 'Host'
        },
        {
          id: '4',
          text: 'Let\'s start with some key examples of successful AI implementations.',
          start_time: 13.8,
          end_time: 18.1,
          speaker: 'Host'
        },
        {
          id: '5',
          text: 'Machine learning algorithms are now capable of processing vast amounts of data.',
          start_time: 18.1,
          end_time: 23.5,
          speaker: 'Host'
        }
      ];
    }
    
    return mockTranscripts[id];
  },

  // Ask question about transcript (real backend call)
  async query(id: string, question: string): Promise<string> {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: id, question })
    });
    if (!response.ok) throw new Error("Failed to get answer");
    const data = await response.json();
    return data.response;
  },

  // Generate highlights (real backend call)
  async generateHighlights(id: string): Promise<Highlight[]> {
    const response = await fetch("http://localhost:5000/generate_highlights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: id })
    });
    if (!response.ok) throw new Error("Failed to generate highlights");
    return await response.json();
  },

  // Export reel (real backend call)
  async exportReel(id: string, highlights: Highlight[]): Promise<{ url: string }> {
    const response = await fetch("/export_reel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: id, highlights })
    });
    if (!response.ok) throw new Error("Failed to export reel");
    const data = await response.json();
    return { url: data.url };
  },

  // Clear conversation history
  async clearConversation(id: string): Promise<void> {
    const response = await fetch("http://localhost:5000/clear_conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: id })
    });
    if (!response.ok) throw new Error("Failed to clear conversation");
  },

  // Get user's video library (real backend call)
  async getLibrary(): Promise<TranscriptionJob[]> {
    const response = await fetch("/library", {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    });
    if (!response.ok) throw new Error("Failed to fetch library");
    return await response.json();
  },

  // Delete a job
  async deleteJob(id: string): Promise<void> {
    const response = await fetch("/delete_job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: id })
    });
    if (!response.ok) throw new Error("Failed to delete job");
  },

  // Simulate progress updates
  simulateProgress(id: string) {
    const job = mockJobs[id];
    if (!job) return;

    const stages = ['downloading', 'transcribing', 'embedding', 'completed'] as const;
    let currentStage = 0;
    let progress = 0;

    const updateProgress = () => {
      progress += Math.random() * 15 + 5;
      
      if (progress >= 100) {
        progress = 100;
        job.status = 'completed';
        job.progress = 100;
        return;
      }

      if (progress > (currentStage + 1) * 25 && currentStage < stages.length - 1) {
        currentStage++;
      }

      job.status = stages[currentStage];
      job.progress = progress;

      setTimeout(updateProgress, 1000 + Math.random() * 2000);
    };

    updateProgress();
  }
};
