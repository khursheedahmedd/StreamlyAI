import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { ChatMessage } from '@/components/ChatMessage';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { toast } from '@/hooks/use-toast';
import { api, TranscriptSegment, ChatMessage as ChatMessageType, TranscriptionJob } from '@/lib/api';
import { useAppStore } from '@/store';
import { formatTime, cn } from '@/lib/utils';
import { 
  Search, 
  Send, 
  Clock, 
  User, 
  MessageSquare,
  Scissors,
  ArrowRight,
  Play,
  Loader2,
  Trash2
} from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';

const Viewer = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<TranscriptionJob | null>(null);
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [filteredTranscript, setFilteredTranscript] = useState<TranscriptSegment[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [activeSegment, setActiveSegment] = useState<string | null>(null);
  const [expandedTopics, setExpandedTopics] = useState<Set<number>>(new Set());
  
  const { messages, addMessage, clearMessages } = useAppStore();
  const chatMessages = messages[id || ''] || [];
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) {
      navigate('/');
      return;
    }

    // Clear previous chat messages for this ID
    clearMessages(id);

    const loadJobData = async () => {
      try {
        const jobData = await api.getStatus(id);
        setJob(jobData);
        if (jobData.status !== 'completed' || !jobData.topics) {
          toast({
            title: "Job not ready",
            description: "This transcription is still processing or has failed.",
            variant: "destructive"
          });
          navigate(`/transcribe/${id}`);
          return;
        }

        // Expand all topics by default
        setExpandedTopics(new Set(jobData.topics.map((_, i) => i)));

        addMessage(id, {
          id: 'initial-bot-message',
          content: 'Hello! I have analyzed this video. Feel free to ask me any questions about its content.',
          isUser: false,
          timestamp: new Date().toISOString()
        });

      } catch (err) {
        toast({
          title: "Error",
          description: "Failed to load transcription data",
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadJobData();
  }, [id, navigate, addMessage, clearMessages]);

  // Search topics and segments
  const filteredTopics = job?.topics
    ? job.topics
        .map((topic, idx) => {
          // Filter segments by search
          const filteredSegments = searchQuery.trim()
            ? topic.segments.filter(seg =>
                seg.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
                topic.title.toLowerCase().includes(searchQuery.toLowerCase())
              )
            : topic.segments;
          return { ...topic, filteredSegments, idx };
        })
        .filter(topic => topic.filteredSegments.length > 0)
    : [];

  const toggleTopic = (idx: number) => {
    setExpandedTopics(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  useEffect(() => {
    if (searchQuery.trim()) {
      const filtered = transcript.filter(segment =>
        segment.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (segment.speaker && segment.speaker.toLowerCase().includes(searchQuery.toLowerCase()))
      );
      setFilteredTranscript(filtered);
    } else {
      setFilteredTranscript(transcript);
    }
  }, [searchQuery, transcript]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleSegmentClick = (segment: TranscriptSegment) => {
    setActiveSegment(segment.id);
    // In a real app, this would seek to the video timestamp
    toast({
      title: "Video Seek",
      description: `Seeking to ${formatTime(segment.start_time)}`
    });
  };

  const handleSourceClick = (start_time: number) => {
    const segment = transcript.find(s => s.start_time === start_time);
    if (segment) {
      handleSegmentClick(segment);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || !id) return;

    const userMessage: ChatMessageType = {
      id: Date.now().toString(),
      content: chatInput,
      isUser: true,
      timestamp: new Date().toISOString()
    };

    addMessage(id, userMessage);
    setChatInput('');
    setIsSending(true);

    try {
      const responseText = await api.query(id, chatInput);
      const botMessage: ChatMessageType = {
        id: Date.now().toString(),
        content: responseText,
        isUser: false,
        timestamp: new Date().toISOString()
      };
      addMessage(id, botMessage);
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to get AI response",
        variant: "destructive"
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleClearConversation = async () => {
    if (!id) return;
    
    try {
      await api.clearConversation(id);
      clearMessages(id);
      toast({
        title: "Conversation cleared",
        description: "Chat history has been reset"
      });
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to clear conversation",
        variant: "destructive"
      });
    }
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">{job?.title || "Video Transcript"}</h1>
              <p className="text-muted-foreground">
                Search through the transcript or ask questions about the content
              </p>
            </div>
            <Button
              onClick={() => navigate(`/reel/${id}`)}
              className="flex items-center space-x-2"
            >
              <Scissors className="h-4 w-4" />
              <span>Create Highlight Reel</span>
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-[2fr_3fr] gap-8">
          {/* Transcript Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="space-y-6"
          >
            {/* Search */}
            <Card className="p-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search transcript..."
                  value={searchQuery}
                  onChange={handleSearch}
                  className="pl-10"
                />
              </div>
              {searchQuery && (
                <p className="text-sm text-muted-foreground mt-2">
                  Found {filteredTopics.length} result(s)
                </p>
              )}
            </Card>

            {/* Transcript */}
            <Card className="p-6">
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {isLoading ? (
                  // Loading skeletons
                  Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ))
                ) : filteredTopics.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    {searchQuery ? 'No results found' : 'No transcript available'}
                  </div>
                ) : (
                  filteredTopics.map((topic) => (
                    <div key={topic.idx} className="border rounded-lg mb-4">
                      <button
                        className="w-full flex items-center justify-between px-4 py-3 bg-muted/40 hover:bg-muted/60 rounded-t-lg focus:outline-none"
                        onClick={() => toggleTopic(topic.idx)}
                      >
                        <span className="font-semibold text-lg text-left">{topic.title}</span>
                        <span className="ml-2 text-xs text-muted-foreground">{formatTime(topic.start_time)} - {formatTime(topic.end_time)}</span>
                        <span className="ml-2">{expandedTopics.has(topic.idx) ? '▲' : '▼'}</span>
                      </button>
                      {expandedTopics.has(topic.idx) && (
                        <div className="p-4 space-y-3">
                          {topic.filteredSegments.map((segment, i) => (
                            <motion.div
                              key={segment.start_time + '-' + i}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.3 }}
                              className={cn(
                                "group cursor-pointer p-3 rounded-lg border transition-colors hover:bg-muted/50",
                                activeSegment === segment.start_time + '-' + i && "bg-primary/10 border-primary"
                              )}
                              onClick={() => handleSegmentClick({ ...segment, id: segment.start_time + '-' + i })}
                            >
                              <div className="flex items-start space-x-3">
                                <Badge
                                  variant="outline"
                                  className="flex items-center space-x-1 text-xs"
                                >
                                  <Clock className="h-3 w-3" />
                                  <span>{formatTime(segment.start_time)}</span>
                                </Badge>
                              </div>
                              <p className="mt-2 leading-relaxed group-hover:text-foreground transition-colors">
                                {segment.text}
                              </p>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </Card>
          </motion.div>

          {/* Chat Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="space-y-6"
          >
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <MessageSquare className="h-5 w-5 text-primary" />
                  <h3 className="text-lg font-semibold">Ask Questions</h3>
                </div>
                {chatMessages.length > 1 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearConversation}
                    className="flex items-center space-x-2 text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                    <span>Clear Chat</span>
                  </Button>
                )}
              </div>
              
              {/* Chat Messages */}
              <div className="space-y-4 max-h-[500px] overflow-y-auto mb-4">
                {chatMessages.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Ask any question about the video content</p>
                    <p className="text-sm mt-2">
                      Try: "What are the main topics discussed?"
                    </p>
                  </div>
                ) : (
                  <>
                    {chatMessages.length > 2 && (
                      <div className="text-xs text-muted-foreground text-center py-2 bg-muted/30 rounded-lg">
                        💬 AI remembers previous conversation context
                      </div>
                    )}
                    {chatMessages.map((message) => (
                      <ChatMessage
                        key={message.id}
                        message={message}
                        onSourceClick={handleSourceClick}
                      />
                    ))}
                  </>
                )}
                {isSending && (
                  <div className="flex items-center space-x-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>AI is thinking...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <Separator className="my-4" />

              {/* Chat Input */}
              <form onSubmit={handleChatSubmit} className="flex space-x-2">
                <Textarea
                  placeholder="Ask about the video content..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  disabled={isSending}
                  className="flex-1 min-h-[40px] max-h-[120px] resize-y"
                  rows={2}
                />
                <Button
                  type="submit"
                  disabled={!chatInput.trim() || isSending}
                  size="icon"
                >
                  {isSending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </form>
            </Card>
          </motion.div>
        </div>
      </div>
    </Layout>
  );
};

export default Viewer;
