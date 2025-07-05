
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { toast } from '@/hooks/use-toast';
import { api, Highlight } from '@/lib/api';
import { formatTime, cn } from '@/lib/utils';
import { 
  Play,
  Trash2,
  Plus,
  Download,
  Shuffle,
  Loader2,
  Sparkles,
  GripVertical,
  Edit3,
  Check,
  X
} from 'lucide-react';

const Reel = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editCaption, setEditCaption] = useState('');
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  useEffect(() => {
    if (!id) {
      navigate('/');
      return;
    }

    loadHighlights();
  }, [id, navigate]);

  const loadHighlights = async () => {
    try {
      setIsLoading(true);
      const data = await api.generateHighlights(id!);
      setHighlights(data);
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to generate highlights",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegenerateHighlights = async () => {
    setIsGenerating(true);
    try {
      const data = await api.generateHighlights(id!);
      setHighlights(data);
      toast({
        title: "Highlights Regenerated",
        description: "New highlights have been generated for your video"
      });
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to regenerate highlights",
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleExport = async () => {
    if (highlights.length === 0) return;
    
    setIsExporting(true);
    try {
      const result = await api.exportReel(id!, highlights);
      
      // Show success message
      toast({
        title: "Reel exported successfully!",
        description: "Your video reel has been created and is ready for download.",
      });
      
      // Create confetti effect
      createConfetti();
      
      // Trigger download
      const link = document.createElement('a');
      link.href = result.url;
      link.download = `reel_${id}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
    } catch (error) {
      console.error('Export failed:', error);
      toast({
        title: "Export failed",
        description: "Failed to export reel. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  const createConfetti = () => {
    const colors = ['#635BFF', '#00C6AE', '#FF6B6B', '#4ECDC4', '#FFD93D'];
    
    for (let i = 0; i < 50; i++) {
      const confetti = document.createElement('div');
      confetti.className = 'absolute w-2 h-2 animate-confetti pointer-events-none';
      confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
      confetti.style.left = Math.random() * 100 + '%';
      confetti.style.top = '100%';
      confetti.style.animationDelay = Math.random() * 2 + 's';
      confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
      document.body.appendChild(confetti);
      
      setTimeout(() => {
        confetti.remove();
      }, 4000);
    }
  };

  const handleDeleteHighlight = (highlightId: string) => {
    setHighlights(prev => prev.filter(h => h.id !== highlightId));
    toast({
      title: "Highlight Removed",
      description: "The highlight has been removed from your reel"
    });
  };

  const startEditing = (highlight: Highlight) => {
    setEditingId(highlight.id);
    setEditCaption(highlight.caption);
  };

  const saveEdit = () => {
    if (!editingId) return;
    
    setHighlights(prev => prev.map(h => 
      h.id === editingId ? { ...h, caption: editCaption } : h
    ));
    setEditingId(null);
    setEditCaption('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditCaption('');
  };

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    
    if (draggedIndex === null) return;
    
    const newHighlights = [...highlights];
    const draggedHighlight = newHighlights[draggedIndex];
    
    newHighlights.splice(draggedIndex, 1);
    newHighlights.splice(dropIndex, 0, draggedHighlight);
    
    setHighlights(newHighlights);
    setDraggedIndex(null);
  };

  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return 'bg-red-500';
    if (importance >= 0.6) return 'bg-orange-500';
    return 'bg-yellow-500';
  };

  return (
    <Layout>
      <div className="max-w-6xl mx-auto px-4 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">Highlight Reel Builder</h1>
              <p className="text-muted-foreground">
                Edit and arrange your video highlights to create the perfect reel
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button
                variant="outline"
                onClick={handleRegenerateHighlights}
                disabled={isGenerating}
                className="flex items-center space-x-2"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Shuffle className="h-4 w-4" />
                )}
                <span>Regenerate</span>
              </Button>
              <Button
                onClick={handleExport}
                disabled={isExporting || highlights.length === 0}
                className="flex items-center space-x-2 gradient-primary"
              >
                {isExporting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                <span>Export Reel</span>
              </Button>
            </div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Video Player Placeholder */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="lg:col-span-2"
          >
            <Card className="p-6">
              <div className="aspect-video bg-black rounded-lg flex items-center justify-center mb-4">
                <div className="text-center text-white">
                  <Play className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg">Video Player</p>
                  <p className="text-sm opacity-70">
                    Click on highlights to preview
                  </p>
                </div>
              </div>
              
              {highlights.length > 0 && (
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Total duration: {formatTime(highlights.reduce((acc, h) => acc + (h.endTime - h.startTime), 0))}</span>
                  <span>{highlights.length} highlight{highlights.length !== 1 ? 's' : ''}</span>
                </div>
              )}
            </Card>
          </motion.div>

          {/* Highlights Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold flex items-center">
                  <Sparkles className="h-5 w-5 mr-2 text-primary" />
                  Highlights
                </h3>
                {highlights.length > 0 && (
                  <Badge variant="secondary">
                    {highlights.length}
                  </Badge>
                )}
              </div>

              {isLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <div className="h-4 bg-muted rounded animate-pulse" />
                      <div className="h-16 bg-muted rounded animate-pulse" />
                    </div>
                  ))}
                </div>
              ) : highlights.length === 0 ? (
                <Alert>
                  <Sparkles className="h-4 w-4" />
                  <AlertDescription>
                    No highlights generated yet. Click "Regenerate" to create some!
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-4 max-h-[600px] overflow-y-auto">
                  {highlights.map((highlight, index) => (
                    <motion.div
                      key={highlight.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                      draggable
                      onDragStart={() => handleDragStart(index)}
                      onDragOver={(e) => handleDragOver(e, index)}
                      onDrop={(e) => handleDrop(e, index)}
                      className={cn(
                        "group relative border rounded-lg p-4 cursor-move transition-all hover:shadow-md",
                        draggedIndex === index && "opacity-50"
                      )}
                    >
                      {/* Drag Handle */}
                      <div className="absolute left-2 top-1/2 transform -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <GripVertical className="h-4 w-4 text-muted-foreground" />
                      </div>

                      <div className="ml-6">
                        {/* Header */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Badge
                              variant="outline"
                              className="text-xs"
                            >
                              {formatTime(highlight.startTime)} - {formatTime(highlight.endTime)}
                            </Badge>
                            <div
                              className={cn(
                                "w-2 h-2 rounded-full",
                                getImportanceColor(highlight.importance)
                              )}
                              title={`Importance: ${Math.round(highlight.importance * 100)}%`}
                            />
                          </div>
                          <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => startEditing(highlight)}
                              className="h-8 w-8 p-0"
                            >
                              <Edit3 className="h-3 w-3" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteHighlight(highlight.id)}
                              className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>

                        {/* Caption */}
                        {editingId === highlight.id ? (
                          <div className="space-y-2">
                            <Input
                              value={editCaption}
                              onChange={(e) => setEditCaption(e.target.value)}
                              placeholder="Enter caption..."
                              className="text-sm"
                            />
                            <div className="flex items-center space-x-2">
                              <Button
                                size="sm"
                                onClick={saveEdit}
                                className="h-7 px-2"
                              >
                                <Check className="h-3 w-3" />
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={cancelEdit}
                                className="h-7 px-2"
                              >
                                <X className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm leading-relaxed">
                            {highlight.caption}
                          </p>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </Card>
          </motion.div>
        </div>
      </div>
    </Layout>
  );
};

export default Reel;
