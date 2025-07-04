
import React, { useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface InteractiveTextProps {
  children: React.ReactNode;
  className?: string;
  hoveredClassName?: string;
}

const InteractiveText: React.FC<InteractiveTextProps> = ({ 
  children, 
  className = "",
  hoveredClassName = "text-gradient"
}) => {
  const textRef = useRef<HTMLSpanElement>(null);
  
  useEffect(() => {
    const text = textRef.current;
    if (!text) return;
    
    const handleMouseEnter = () => {
      text.classList.add(...hoveredClassName.split(' '));
    };
    
    const handleMouseLeave = () => {
      text.classList.remove(...hoveredClassName.split(' '));
    };
    
    text.addEventListener('mouseenter', handleMouseEnter);
    text.addEventListener('mouseleave', handleMouseLeave);
    
    return () => {
      text.removeEventListener('mouseenter', handleMouseEnter);
      text.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [hoveredClassName]);
  
  return (
    <span 
      ref={textRef} 
      className={cn("interactive-element transition-all duration-300", className)}
    >
      {children}
    </span>
  );
};

export default InteractiveText;
