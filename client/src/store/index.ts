
import { create } from 'zustand';
import { TranscriptionJob, ChatMessage } from '@/lib/api';

interface AppState {
  // Theme
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  
  // Current job
  currentJob: TranscriptionJob | null;
  setCurrentJob: (job: TranscriptionJob | null) => void;
  
  // Chat messages
  messages: Record<string, ChatMessage[]>;
  addMessage: (jobId: string, message: ChatMessage) => void;
  clearMessages: (jobId: string) => void;
  
  // UI state
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  theme: 'system',
  setTheme: (theme) => {
    set({ theme });
    // Apply theme to document
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      // System theme
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', prefersDark);
    }
  },
  
  currentJob: null,
  setCurrentJob: (job) => set({ currentJob: job }),
  
  messages: {},
  addMessage: (jobId, message) => {
    const { messages } = get();
    set({
      messages: {
        ...messages,
        [jobId]: [...(messages[jobId] || []), message]
      }
    });
  },
  clearMessages: (jobId) => {
    const { messages } = get();
    const newMessages = { ...messages };
    delete newMessages[jobId];
    set({ messages: newMessages });
  },
  
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading })
}));
