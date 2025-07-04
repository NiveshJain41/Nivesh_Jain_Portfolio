
import React, { useRef, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface SkillBarProps {
  skill: string;
  percentage: number;
  className?: string;
}

const SkillBar: React.FC<SkillBarProps> = ({ 
  skill, 
  percentage, 
  className = "" 
}) => {
  const progressRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
          } else {
            setIsVisible(false); // Reset when not visible
          }
        });
      },
      { threshold: 0.2 }
    );
    
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    
    return () => {
      if (containerRef.current) {
        observer.unobserve(containerRef.current);
      }
    };
  }, []);
  
  useEffect(() => {
    if (progressRef.current) {
      if (isVisible) {
        progressRef.current.style.width = `${percentage}%`;
      } else {
        progressRef.current.style.width = "0%";
      }
    }
  }, [isVisible, percentage]);
  
  return (
    <div ref={containerRef} className={cn("mb-6", className)}>
      <div className="flex justify-between mb-2">
        <span className="font-medium">{skill}</span>
        <span className="text-silver/70">{percentage}%</span>
      </div>
      <div className="h-2 bg-navy-600 rounded-full overflow-hidden">
        <div 
          ref={progressRef} 
          className="h-full bg-gradient-to-r from-purple to-teal rounded-full transition-all duration-1000 ease-out"
          style={{ width: "0%" }}
        ></div>
      </div>
    </div>
  );
};

export default SkillBar;
