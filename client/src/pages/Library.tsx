
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { toast } from '@/hooks/use-toast';
import { api, TranscriptionJob } from '@/lib/api';
import { formatRelativeTime, formatDuration, truncateText } from '@/lib/utils';
import { 
  Search, 
  Play, 
  FileText, 
  Scissors, 
  Trash2, 
  Clock,
  Plus,
  Filter,
  Grid,
  List
} from 'lucide-react';

const Library = () => {
  const [jobs, setJobs] = useState<TranscriptionJob[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<TranscriptionJob[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  useEffect(() => {
    loadLibrary();
  }, []);

  useEffect(() => {
    if (searchQuery.trim()) {
      const filtered = jobs.filter(job =>
        job.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        job.video_url.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredJobs(filtered);
    } else {
      setFilteredJobs(jobs);
    }
  }, [searchQuery, jobs]);

  const loadLibrary = async () => {
    try {
      setIsLoading(true);
      const data = await api.getLibrary();
      setJobs(data);
      setFilteredJobs(data);
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to load library",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      setJobs(prev => prev.filter(job => job.id !== jobId));
      setFilteredJobs(prev => prev.filter(job => job.id !== jobId));
      toast({
        title: "Video Deleted",
        description: "The video has been removed from your library"
      });
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to delete video",
        variant: "destructive"
      });
    }
  };

  const JobCard = ({ job, isGridView }: { job: TranscriptionJob; isGridView: boolean }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="group"
    >
      <Card className={`overflow-hidden hover:shadow-lg transition-all ${
        isGridView ? 'h-full' : 'flex items-center p-4'
      }`}>
        {isGridView ? (
          <>
            {/* Thumbnail */}
            <div className="aspect-video relative overflow-hidden">
              {job.thumbnail ? (
                <img
                  src={job.thumbnail}
                  alt={job.title || 'Video thumbnail'}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="w-full h-full bg-muted flex items-center justify-center">
                  <Play className="h-12 w-12 text-muted-foreground" />
                </div>
              )}
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-300" />
            </div>

            {/* Content */}
            <div className="p-4">
              <h3 className="font-semibold mb-2 line-clamp-2">
                {job.title || 'YouTube Video'}
              </h3>
              
              <div className="flex items-center space-x-4 text-sm text-muted-foreground mb-4">
                {job.duration && (
                  <span className="flex items-center">
                    <Clock className="h-3 w-3 mr-1" />
                    {formatDuration(job.duration)}
                  </span>
                )}
                <span>{formatRelativeTime(new Date(job.created_at * 1000).toISOString())}</span>
              </div>

              <div className="flex items-center space-x-2">
                <Link to={`/viewer/${job.id}`} className="flex-1">
                  <Button variant="outline" size="sm" className="w-full">
                    <FileText className="h-4 w-4 mr-2" />
                    Chat
                  </Button>
                </Link>
                <Link to={`/reel/${job.id}`}>
                  <Button variant="outline" size="sm">
                    <Scissors className="h-4 w-4" />
                  </Button>
                </Link>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDelete(job.id)}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </>
        ) : (
          <>
            {/* List view */}
            <div className="flex items-center space-x-4 flex-1">
              {job.thumbnail ? (
                <img
                  src={job.thumbnail}
                  alt={job.title || 'Video thumbnail'}
                  className="w-20 h-12 object-cover rounded"
                />
              ) : (
                <div className="w-20 h-12 bg-muted rounded flex items-center justify-center">
                  <Play className="h-6 w-6 text-muted-foreground" />
                </div>
              )}
              
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold truncate">
                  {job.title || 'YouTube Video'}
                </h3>
                <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                  {job.duration && (
                    <span className="flex items-center">
                      <Clock className="h-3 w-3 mr-1" />
                      {formatDuration(job.duration)}
                    </span>
                  )}
                  <span>{formatRelativeTime(new Date(job.created_at * 1000).toISOString())}</span>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Link to={`/viewer/${job.id}`}>
                  <Button variant="outline" size="sm">
                    <FileText className="h-4 w-4 mr-2" />
                    View
                  </Button>
                </Link>
                <Link to={`/reel/${job.id}`}>
                  <Button variant="outline" size="sm">
                    <Scissors className="h-4 w-4" />
                  </Button>
                </Link>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDelete(job.id)}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
    </motion.div>
  );

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
              <h1 className="text-3xl font-bold mb-2">Your Library</h1>
              <p className="text-muted-foreground">
                Manage all your transcribed videos in one place
              </p>
            </div>
            <Link to="/">
              <Button className="flex items-center space-x-2">
                <Plus className="h-4 w-4" />
                <span>Add New Video</span>
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Controls */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8"
        >
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search your videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('grid')}
            >
              <Grid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('list')}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {isLoading ? (
            <div className={viewMode === 'grid' ? 'grid md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className={viewMode === 'grid' ? 'space-y-3' : 'h-20'}>
                  <div className={`bg-muted rounded animate-pulse ${
                    viewMode === 'grid' ? 'aspect-video' : 'h-full'
                  }`} />
                  {viewMode === 'grid' && (
                    <>
                      <div className="h-4 bg-muted rounded animate-pulse" />
                      <div className="h-3 bg-muted rounded animate-pulse w-2/3" />
                    </>
                  )}
                </div>
              ))}
            </div>
          ) : filteredJobs.length === 0 ? (
            <div className="text-center py-16">
              <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-lg font-semibold mb-2">
                {searchQuery ? 'No videos found' : 'Your library is empty'}
              </h3>
              <p className="text-muted-foreground mb-6">
                {searchQuery 
                  ? 'Try adjusting your search terms'
                  : 'Start by adding your first YouTube video to transcribe'
                }
              </p>
              {!searchQuery && (
                <Link to="/">
                  <Button>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First Video
                  </Button>
                </Link>
              )}
            </div>
          ) : (
            <div className={viewMode === 'grid' ? 'grid md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
              {filteredJobs.map((job, index) => (
                <motion.div
                  key={job.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <JobCard job={job} isGridView={viewMode === 'grid'} />
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        {/* Stats */}
        {!isLoading && filteredJobs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="mt-8 text-center text-sm text-muted-foreground"
          >
            {searchQuery ? (
              <p>Showing {filteredJobs.length} of {jobs.length} videos</p>
            ) : (
              <p>Total videos: {jobs.length}</p>
            )}
          </motion.div>
        )}
      </div>
    </Layout>
  );
};

export default Library;
