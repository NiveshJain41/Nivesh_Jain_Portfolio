
import React, { useRef, useEffect, ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ScrollRevealProps {
  children: ReactNode;
  effect?: 'fade-in' | 'slide-right' | 'slide-left' | 'scale';
  delay?: number;
  className?: string;
  threshold?: number;
  once?: boolean;
}

const ScrollReveal: React.FC<ScrollRevealProps> = ({
  children,
  effect = 'fade-in',
  delay = 0,
  className = '',
  threshold = 0.2,
  once = true
}) => {
  const elementRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    
    // Set the initial state based on the effect
    let effectClass = '';
    switch (effect) {
      case 'fade-in':
        effectClass = 'reveal-fade-in';
        break;
      case 'slide-right':
        effectClass = 'reveal-slide-right';
        break;
      case 'slide-left':
        effectClass = 'reveal-slide-left';
        break;
      case 'scale':
        effectClass = 'reveal-scale';
        break;
    }
    
    element.classList.add(effectClass);
    element.style.transitionDelay = `${delay}ms`;
    
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            element.classList.add('revealed');
            if (once) {
              observer.unobserve(element);
            }
          } else if (!once) {
            element.classList.remove('revealed');
          }
        });
      },
      { threshold }
    );
    
    observer.observe(element);
    
    return () => {
      if (element) {
        observer.unobserve(element);
      }
    };
  }, [effect, delay, threshold, once]);
  
  return (
    <div 
      ref={elementRef} 
      className={cn('reveal-on-scroll hw-accelerated', className)}
    >
      {children}
    </div>
  );
};

export default ScrollReveal;
