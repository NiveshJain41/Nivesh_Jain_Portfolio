
import React, { useState, useEffect } from 'react';
import InteractiveText from './InteractiveText';
import { motion } from 'framer-motion';

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('home');
  const [menuOpen, setMenuOpen] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => {
      const offset = window.scrollY;
      if (offset > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
      
      // Update active section based on scroll position
      const sections = document.querySelectorAll('section[id]');
      sections.forEach(section => {
        const sectionTop = (section as HTMLElement).offsetTop - 100;
        const sectionHeight = (section as HTMLElement).offsetHeight;
        const sectionId = section.getAttribute('id') || '';
        
        if (offset >= sectionTop && offset < sectionTop + sectionHeight) {
          setActiveSection(sectionId);
        }
      });
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const navLinks = [
    { name: 'Home', id: 'home' },
    { name: 'About', id: 'about' },
    { name: 'Portfolio', id: 'portfolio' },
    { name: 'Skills', id: 'skills' },
    { name: 'Contact', id: 'contact' },
  ];
  
  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const navbarVariants = {
    hidden: { y: -100, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { 
        type: 'spring', 
        stiffness: 100, 
        damping: 20 
      }
    }
  };
  
  return (
    <motion.header 
      className={`fixed top-0 left-0 w-full z-40 ${
        scrolled ? 'header-solid py-3 shadow-md' : 'header-transparent py-6'
      }`}
      initial="hidden"
      animate="visible"
      variants={navbarVariants}
    >
      <div className="container-custom flex justify-between items-center">
        <a href="#home" className="text-3xl font-bold text-gradient hw-accelerated">
          Portfolio
        </a>
        
        <nav className="hidden md:block">
          <ul className="flex gap-8">
            {navLinks.map(link => (
              <li key={link.id}>
                <a 
                  href={`#${link.id}`}
                  className={`interactive-element ${
                    activeSection === link.id ? 'text-purple' : 'text-silver/80'
                  }`}
                >
                  <InteractiveText>{link.name}</InteractiveText>
                </a>
              </li>
            ))}
          </ul>
        </nav>
        
        <button 
          className={`md:hidden p-2 rounded-md transition-all z-50 ${menuOpen ? 'text-purple' : 'text-silver'}`}
          aria-label="Menu"
          onClick={toggleMenu}
        >
          <div className="w-6 h-5 relative flex flex-col justify-between">
            <span className={`w-full h-0.5 bg-current transform transition-all duration-300 ${menuOpen ? 'rotate-45 translate-y-2' : ''}`}></span>
            <span className={`w-full h-0.5 bg-current transition-all duration-300 ${menuOpen ? 'opacity-0' : 'opacity-100'}`}></span>
            <span className={`w-full h-0.5 bg-current transform transition-all duration-300 ${menuOpen ? '-rotate-45 -translate-y-2' : ''}`}></span>
          </div>
        </button>
        
        {/* Mobile menu */}
        <div 
          className={`fixed md:hidden inset-0 bg-navy/95 z-40 flex flex-col items-center justify-center transition-all duration-500 ${
            menuOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
          }`}
        >
          <ul className="flex flex-col gap-6 text-center">
            {navLinks.map(link => (
              <li key={link.id} className="text-2xl py-2">
                <a 
                  href={`#${link.id}`}
                  className={`interactive-element ${
                    activeSection === link.id ? 'text-purple' : 'text-silver/80'
                  }`}
                  onClick={() => setMenuOpen(false)}
                >
                  <InteractiveText>{link.name}</InteractiveText>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </motion.header>
  );
};

export default Navbar;
