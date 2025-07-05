import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChatMessage as ChatMessageType } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import { User, Bot, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

interface ChatMessageProps {
  message: ChatMessageType;
  onSourceClick?: (startTime: number) => void;
}

export function ChatMessage({ message, onSourceClick }: ChatMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex items-start space-x-3 ${message.isUser ? 'flex-row-reverse space-x-reverse' : ''}`}
    >
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        message.isUser ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'
      }`}>
        {message.isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      
      <div className={`flex-1 max-w-[80%] ${message.isUser ? 'text-right' : ''}`}>
        <Card className={`p-3 ${
          message.isUser 
            ? 'bg-primary text-primary-foreground ml-auto' 
            : 'bg-muted'
        }`}>
          <div className="text-sm leading-relaxed">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
          
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 space-y-2">
              <p className="text-xs opacity-70">Related transcript segments:</p>
              <div className="flex flex-wrap gap-1">
                {message.sources.map((source) => (
                  <Badge
                    key={source.id}
                    variant="secondary"
                    className="cursor-pointer hover:bg-background/20 transition-colors"
                    onClick={() => onSourceClick?.(source.start_time)}
                  >
                    <Clock className="h-3 w-3 mr-1" />
                    {formatTime(source.start_time)}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </Card>
        
        <p className="text-xs text-muted-foreground mt-1 px-1">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </motion.div>
  );
}
