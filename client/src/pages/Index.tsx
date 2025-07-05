import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';
import { api, isValidYouTubeUrl } from '@/lib/api';
import { useAppStore } from '@/store';
import { 
  Play, 
  Search, 
  MessageSquare, 
  Scissors, 
  ArrowRight,
  CheckCircle,
  Youtube
} from 'lucide-react';

const Index = () => {
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { setCurrentJob, setIsLoading } = useAppStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!url.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }

    if (!isValidYouTubeUrl(url)) {
      setError('Please enter a valid YouTube URL');
      return;
    }

    try {
      setIsLoading(true);
      const job = await api.transcribe(url);
      setCurrentJob(job);
      toast({
        title: "Transcription Started",
        description: "Your video is being processed. You'll be redirected to the progress page.",
      });
      navigate(`/transcribe/${job.id}`);
    } catch (err: any) {
      // Handle specific error cases
      let errorMessage = 'Failed to start transcription. Please try again.';
      let toastMessage = 'Failed to start transcription. Please try again.';
      
      if (err.message && err.message.includes('Video too long')) {
        errorMessage = 'This video is too long (over 2 hours). Please try a shorter video to avoid timeouts.';
        toastMessage = 'Video too long - please try a shorter video.';
      } else if (err.message && err.message.includes('timeout')) {
        errorMessage = 'Transcription timed out. This usually happens with very long videos. Please try a shorter video.';
        toastMessage = 'Transcription timed out - please try a shorter video.';
      } else if (err.message && err.message.includes('Invalid YouTube URL')) {
        errorMessage = 'Please check the YouTube URL and make sure the video is accessible.';
        toastMessage = 'Invalid YouTube URL - please check the link.';
      }
      
      setError(errorMessage);
      toast({
        title: "Error",
        description: toastMessage,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const features = [
    {
      icon: Youtube,
      title: 'Smart Transcription',
      description: 'AI-powered transcription with speaker identification and timestamps'
    },
    {
      icon: Search,
      title: 'Search & Navigate',
      description: 'Find any moment in your video with full-text search capabilities'
    },
    {
      icon: MessageSquare,
      title: 'Ask Questions',
      description: 'Chat with your video content using our AI assistant'
    },
    {
      icon: Scissors,
      title: 'Auto Highlights',
      description: 'Generate shareable highlight reels automatically'
    }
  ];

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-12">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-6xl font-bold mb-6 text-balance">
            Transform Your{' '}
            <span className="gradient-text">YouTube Videos</span>
            {' '}into Interactive Content
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto text-balance">
            AI-powered transcription, intelligent search, and automatic highlight generation 
            for any YouTube video. Make your content searchable, quotable, and shareable.
          </p>
          <p className="text-sm text-muted-foreground mb-8 max-w-2xl mx-auto">
            💡 <strong>Tip:</strong> For best results, use videos under 2 hours. Longer videos will use local transcription for better reliability.
          </p>

          {/* URL Input Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="max-w-2xl mx-auto"
          >
            <Card className="p-6 shadow-lg">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="youtube-url" className="text-left block">
                    YouTube URL
                  </Label>
                  <div className="flex space-x-2">
                    <Input
                      id="youtube-url"
                      type="url"
                      placeholder="https://youtube.com/watch?v=..."
                      value={url}
                      onChange={(e) => {
                        setUrl(e.target.value);
                        setError('');
                      }}
                      className="flex-1"
                    />
                    <Button 
                      type="submit" 
                      className="gradient-primary"
                      disabled={!url.trim()}
                    >
                      <Play className="h-4 w-4 mr-2" />
                      Transcribe
                    </Button>
                  </div>
                </div>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </form>
            </Card>
          </motion.div>

          {/* Demo Video */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mt-12"
          >
            <div className="relative max-w-4xl mx-auto">
              <div className="aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl">
                <video
                  className="w-full h-full object-cover"
                  controls
                  preload="metadata"
                  poster="/placeholder.svg"
                >
                  <source src="/demo.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
              <div className="absolute inset-0 pointer-events-none rounded-2xl ring-1 ring-inset ring-white/10"></div>
            </div>
          </motion.div>
        </motion.div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-center mb-12">
            Everything you need to unlock your video content
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.1 * index }}
                >
                  <Card className="p-6 h-full hover-scale">
                    <Icon className="h-12 w-12 text-primary mb-4" />
                    <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* How it works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold mb-12">How it works</h2>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {[
              { step: '1', title: 'Paste URL', description: 'Simply paste any YouTube URL' },
              { step: '2', title: 'AI Processing', description: 'Our AI transcribes and analyzes your video' },
              { step: '3', title: 'Explore & Share', description: 'Search, chat, and create highlight reels' }
            ].map((step, index) => (
              <div key={step.step} className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-full gradient-primary text-white flex items-center justify-center text-lg font-bold mb-4">
                  {step.step}
                </div>
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
                {index < 2 && (
                  <ArrowRight className="h-6 w-6 text-muted-foreground/50 mt-4 hidden md:block" />
                )}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </Layout>
  );
};

export default Index;
