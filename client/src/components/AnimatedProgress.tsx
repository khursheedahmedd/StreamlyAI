import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Progress } from '@/components/ui/progress';

interface AnimatedProgressProps {
  value: number;
  className?: string;
  showPercentage?: boolean;
  animated?: boolean;
}

export const AnimatedProgress = ({ 
  value, 
  className = "", 
  showPercentage = true, 
  animated = true 
}: AnimatedProgressProps) => {
  const [displayValue, setDisplayValue] = useState(0);
  const [displayPercentage, setDisplayPercentage] = useState(0);

  useEffect(() => {
    if (animated) {
      // Animate the progress bar
      const progressTimer = setTimeout(() => {
        setDisplayValue(value);
      }, 100);

      // Animate the percentage number
      const startPercentage = displayPercentage;
      const endPercentage = Math.round(value);
      const duration = 800; // 800ms animation
      const steps = 60; // 60 steps for smooth animation
      const increment = (endPercentage - startPercentage) / steps;
      const stepDuration = duration / steps;

      let currentStep = 0;
      const percentageTimer = setInterval(() => {
        currentStep++;
        const newPercentage = Math.round(startPercentage + (increment * currentStep));
        setDisplayPercentage(newPercentage);

        if (currentStep >= steps) {
          setDisplayPercentage(endPercentage);
          clearInterval(percentageTimer);
        }
      }, stepDuration);

      return () => {
        clearTimeout(progressTimer);
        clearInterval(percentageTimer);
      };
    } else {
      setDisplayValue(value);
      setDisplayPercentage(Math.round(value));
    }
  }, [value, animated, displayPercentage]);

  return (
    <div className={`space-y-2 ${className}`}>
      {showPercentage && (
        <div className="flex justify-between text-sm font-medium">
          {/* <span>Progress</span> */}
          {/* <motion.span
            key={displayPercentage}
            initial={{ scale: 1.2, color: '#3b82f6' }}
            animate={{ scale: 1, color: 'inherit' }}
            transition={{ duration: 0.3 }}
            className="font-bold"
          >
            {displayPercentage}%
          </motion.span> */}
        </div>
      )}
      
      <div className="relative">
        <Progress 
          value={displayValue} 
          className="h-3 transition-all duration-300 ease-out" 
        />
        
        {/* Animated progress indicator */}
        <AnimatePresence>
          {displayValue > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="absolute -top-1 -right-1 w-5 h-5 bg-primary rounded-full flex items-center justify-center"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-3 h-3 border-2 border-white border-t-transparent rounded-full"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}; 