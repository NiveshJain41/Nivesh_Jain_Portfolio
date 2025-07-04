import React, { useEffect, useState, useRef } from 'react';
import { HandIcon } from 'lucide-react';

const CustomCursor = () => {
  const cursorRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const handRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isPointer, setIsPointer] = useState(false);
  const [isClicking, setIsClicking] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);
  const particlesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Check if device is desktop (not mobile/tablet)
    const checkDevice = () => {
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const hasTouch = 'ontouchstart' in window;
      const isLargeScreen = window.innerWidth > 768;
      
      setIsDesktop(!isMobile && !hasTouch && isLargeScreen);
    };

    checkDevice();
    window.addEventListener('resize', checkDevice);

    return () => {
      window.removeEventListener('resize', checkDevice);
    };
  }, []);

  useEffect(() => {
    if (!isDesktop) return;

    // Only hide default cursor on desktop when custom cursor is active
    document.body.classList.add('custom-cursor-active');

    const handleMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
      
      // Create particle effect on mouse move - less frequent for a more subtle effect
      if (Math.random() > 0.95 && particlesRef.current) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        const size = Math.random() * 3 + 1; // Smaller particles
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.background = Math.random() > 0.5 ? '#ffffff' : '#f3f3f3'; // More normal colors
        particle.style.left = `${e.clientX}px`;
        particle.style.top = `${e.clientY}px`;
        
        particlesRef.current.appendChild(particle);
        
        setTimeout(() => {
          if (particlesRef.current && particlesRef.current.contains(particle)) {
            particlesRef.current.removeChild(particle);
          }
        }, 1000);
      }
      
      // Check if cursor is over interactive elements
      const target = e.target as HTMLElement;
      const isInteractive = 
        target.tagName === 'BUTTON' || 
        target.tagName === 'A' || 
        target.closest('.interactive-element') ||
        target.closest('.hover-card') ||
        target.closest('[role="button"]') ||
        target.closest('input') ||
        target.closest('textarea') ||
        target.style.cursor === 'pointer';
      
      setIsPointer(!!isInteractive);
    };
    
    const handleMouseDown = () => setIsClicking(true);
    const handleMouseUp = () => setIsClicking(false);
    
    const handleMouseLeave = () => {
      // Show default cursor when mouse leaves window
      document.body.classList.remove('custom-cursor-active');
    };
    
    const handleMouseEnter = () => {
      // Hide default cursor when mouse enters window
      document.body.classList.add('custom-cursor-active');
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
      
      // Restore default cursor when component unmounts
      document.body.classList.remove('custom-cursor-active');
    };
  }, [isDesktop]);
  
  useEffect(() => {
    if (!isDesktop) return;

    const updateCursorPosition = () => {
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate(${position.x}px, ${position.y}px)`;
        cursorRef.current.style.width = isPointer ? '1.75rem' : '1.5rem'; // Smaller overall
        cursorRef.current.style.height = isPointer ? '1.75rem' : '1.5rem';
        cursorRef.current.style.opacity = isClicking ? '0.7' : '0.4'; // More transparent
      }
      
      if (dotRef.current) {
        dotRef.current.style.transform = `translate(${position.x}px, ${position.y}px)`;
      }
      
      if (handRef.current) {
        handRef.current.style.transform = `translate(${position.x}px, ${position.y}px)`;
        handRef.current.style.opacity = isPointer ? '1' : '0';
        handRef.current.style.scale = isClicking ? '0.9' : '1';
      }
    };
    
    // Use requestAnimationFrame for smoother animation
    const frame = requestAnimationFrame(updateCursorPosition);
    return () => cancelAnimationFrame(frame);
  }, [position, isPointer, isClicking, isDesktop]);

  // Don't render custom cursor on mobile devices
  if (!isDesktop) {
    return null;
  }

  return (
    <>
      <div ref={cursorRef} className="custom-cursor"></div>
      <div ref={dotRef} className="cursor-dot"></div>
      <div ref={handRef} className="cursor-hand">
        <HandIcon className="w-4 h-4 text-white" />
      </div>
      <div ref={particlesRef} className="particles-container"></div>
    </>
  );
};

export default CustomCursor;