import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AnimatedProgress } from '@/components/AnimatedProgress';
import { toast } from '@/hooks/use-toast';
import { api, TranscriptionJob } from '@/lib/api';
import { formatDuration } from '@/lib/utils';
import {
  Clock,
  Download,
  FileText,
  Database,
  CheckCircle,
  X,
  AlertCircle,
  Play,
  Sparkles
} from 'lucide-react';

const Transcribe = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<TranscriptionJob | null>(null);
  const [error, setError] = useState('');
  const [loadingMessage, setLoadingMessage] = useState(0);

  // Fun loading messages that cycle
  const loadingMessages = [
    "🎵 Converting audio to text...",
    "🧠 Analyzing speech patterns...",
    "📝 Creating detailed transcript...",
    "🔍 Finding key moments...",
    "💡 Identifying topics...",
    "🎯 Preparing for Q&A...",
    "⚡ Almost there...",
    "🚀 Finalizing your video analysis..."
  ];

  useEffect(() => {
    if (!id) {
      navigate('/');
      return;
    }

    const pollStatus = async () => {
      try {
        const jobStatus = await api.getStatus(id);
        setJob(jobStatus);
        if (jobStatus.status === 'failed') {
          const errorMsg = jobStatus.error || 'Transcription failed. Please try again.';
          setError(errorMsg);
          
          // Show specific toast for timeout errors
          if (errorMsg.includes('timed out') || errorMsg.includes('timeout')) {
            toast({
              title: "Transcription Timeout",
              description: "The video was too long. Please try a shorter video (under 30 minutes).",
              variant: "destructive"
            });
          }
        }
      } catch (err) {
        setError('Failed to get job status');
      }
    };

    pollStatus();
    const interval = setInterval(() => {
      setJob(prevJob => {
        if (prevJob?.status === 'completed' || prevJob?.status === 'failed') {
          clearInterval(interval);
          return prevJob;
        }
        pollStatus();
        return prevJob;
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [id, navigate]);
  
  useEffect(() => {
    if (job?.status === 'completed') {
      toast({
        title: "Transcription Complete!",
        description: "Your video has been successfully processed."
      });
      setTimeout(() => navigate(`/viewer/${id}`), 1500);
    }
  }, [job?.status, id, navigate]);

  useEffect(() => {
    // Cycle through loading messages every 3 seconds
    const messageInterval = setInterval(() => {
      setLoadingMessage(prev => (prev + 1) % loadingMessages.length);
    }, 3000);

    return () => clearInterval(messageInterval);
  }, []);

  const handleCancel = () => {
    toast({
      title: "Navigation Cancelled",
      description: "Returning to the home page."
    });
    navigate('/');
  };

  const STAGES = [
    { key: 'queued', title: 'Queued', icon: Clock, description: 'Preparing to process' },
    { key: 'downloading', title: 'Downloading', icon: Download, description: 'Getting video audio' },
    { key: 'transcribing', title: 'Transcribing', icon: FileText, description: 'Converting speech to text' },
    { key: 'segmenting', title: 'Analyzing', icon: Sparkles, description: 'Finding topics & highlights' },
    { key: 'embedding', title: 'Processing', icon: Database, description: 'Preparing for Q&A' },
  ];

  const currentStageIndex = STAGES.findIndex(s => s.key === job?.status);

  const getStageTitle = (status: string) => STAGES.find(s => s.key === status)?.title || 'Processing';

  const CurrentStageIcon = STAGES[currentStageIndex > -1 ? currentStageIndex : 0].icon;

  if (!job && !error) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-screen">
          <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full"></div>
        </div>
      </Layout>
    );
  }
  
  if (error) {
    return (
      <Layout>
        <div className="max-w-4xl mx-auto px-4 lg:px-8 py-12">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <div className="mt-4 text-center">
            <Button onClick={() => navigate('/')}>Return Home</Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 lg:px-8 py-12">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-4">Processing Your Video</h1>
            <p className="text-muted-foreground">
              We're transcribing and analyzing your content. 
              {job?.duration && job.duration > 1800 ? 
                ` This video is ${Math.round(job.duration / 60)} minutes long, so it may take 5-10 minutes. We'll use local transcription for better reliability.` : 
                ' This usually takes 1-3 minutes.'
              }
            </p>
          </div>

          <Card className="p-6 mb-8">
            <div className="flex items-start space-x-4">
              {job.thumbnail && <img src={job.thumbnail} alt="Video thumbnail" className="w-32 h-20 object-cover rounded-lg" />}
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-2">{job.title || 'YouTube Video'}</h3>
                <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                  {job.duration && <span className="flex items-center"><Play className="h-4 w-4 mr-1" />{formatDuration(job.duration)}</span>}
                  <span>Started {new Date(job.created_at * 1000).toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-8 mb-6">
            <motion.div 
              className="space-y-6"
              animate={job.status !== 'completed' && job.status !== 'failed' ? {
                boxShadow: [
                  "0 0 0 0 rgba(59, 130, 246, 0.4)",
                  "0 0 0 10px rgba(59, 130, 246, 0)",
                  "0 0 0 0 rgba(59, 130, 246, 0)"
                ]
              } : {}}
              transition={{
                duration: 2,
                repeat: job.status !== 'completed' && job.status !== 'failed' ? Infinity : 0,
                ease: "easeInOut"
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <motion.div 
                    className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center"
                    animate={job.status !== 'completed' && job.status !== 'failed' ? {
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0]
                    } : {}}
                    transition={{
                      duration: 2,
                      repeat: job.status !== 'completed' && job.status !== 'failed' ? Infinity : 0,
                      ease: "easeInOut"
                    }}
                  >
                    <CurrentStageIcon className="h-5 w-5 text-primary" />
                  </motion.div>
                  <span className="text-lg font-semibold">{getStageTitle(job.status)}</span>
                </div>
                <Badge variant="secondary" className="capitalize">{job.status}</Badge>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm font-medium">
                  <span>Progress</span>
                  <span>{Math.round(job.progress)}%</span>
                </div>
                <AnimatedProgress value={job.progress} />
                {job.status === 'transcribing' && job.duration && job.duration > 1800 && (
                  <p className="text-xs text-muted-foreground mt-2">
                    ⏱️ Long video detected ({Math.round(job.duration / 60)} min). Using local transcription for better reliability.
                  </p>
                )}
                
                {/* Animated status indicator */}
                <motion.div 
                  className="flex items-center space-x-2 mt-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <motion.div
                    animate={{ 
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0]
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="w-2 h-2 bg-primary rounded-full"
                  />
                  <span className="text-sm text-muted-foreground">
                    {job.status === 'transcribing' ? 'Processing audio...' : 
                     job.status === 'segmenting' ? 'Analyzing content...' :
                     job.status === 'embedding' ? 'Preparing for Q&A...' :
                     'Processing...'}
                  </span>
                </motion.div>
                
                {/* Animated loading message */}
                {/* {job.status !== 'completed' && job.status !== 'failed' && (
                  <motion.div
                    key={loadingMessage}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.5 }}
                    className="text-center mt-4"
                  >
                    <p className="text-sm text-muted-foreground font-medium">
                      {loadingMessages[loadingMessage]}
                    </p>
                  </motion.div>
                )} */}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {STAGES.map((stage, index) => {
                  const isActive = currentStageIndex === index;
                  const isCompleted = currentStageIndex > index;
                  const Icon = stage.icon;
                  return (
                    <motion.div 
                      key={stage.key} 
                      initial={{ opacity: 0, y: 20 }} 
                      animate={{ opacity: 1, y: 0 }} 
                      transition={{ delay: index * 0.1 }}
                      className={`text-center p-4 rounded-lg border transition-all duration-300 ${
                        isActive 
                          ? 'border-primary bg-primary/5 shadow-lg scale-105' 
                          : isCompleted 
                            ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20' 
                            : 'border-muted hover:border-muted-foreground/30'
                      }`}
                    >
                      <motion.div 
                        className={`w-12 h-12 rounded-full mx-auto mb-3 flex items-center justify-center transition-all duration-300 ${
                          isActive 
                            ? 'bg-primary text-primary-foreground shadow-lg' 
                            : isCompleted 
                              ? 'bg-green-500 text-white' 
                              : 'bg-muted text-muted-foreground'
                        }`}
                        animate={isActive ? {
                          scale: [1, 1.1, 1],
                          rotate: [0, 5, -5, 0]
                        } : {}}
                        transition={isActive ? {
                          duration: 2,
                          repeat: Infinity,
                          ease: "easeInOut"
                        } : {}}
                      >
                        {isCompleted ? (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 500, damping: 30 }}
                          >
                            <CheckCircle className="h-6 w-6" />
                          </motion.div>
                        ) : (
                          <Icon className="h-6 w-6" />
                        )}
                      </motion.div>
                      <p className="text-sm font-medium capitalize mb-1">{stage.title}</p>
                      <p className="text-xs text-muted-foreground">{stage.description}</p>
                      
                      {/* Progress indicator for active stage */}
                      {isActive && (
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: "100%" }}
                          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                          className="h-1 bg-primary/30 rounded-full mt-2"
                        />
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          </Card>

          <div className="flex justify-center space-x-4">
            <Button variant="outline" onClick={handleCancel}>
              <X className="h-4 w-4 mr-2" /> Cancel
            </Button>
          </div>
        </motion.div>
      </div>
    </Layout>
  );
};

export default Transcribe;
