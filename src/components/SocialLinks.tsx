
import React from 'react';
import { Github, Linkedin, Instagram, Mail } from 'lucide-react';

const SocialLinks = () => {
  const socialLinks = [
    { 
      icon: <Linkedin className="w-6 h-6" />, 
      url: 'https://www.linkedin.com/in/niveshjain41/', 
      label: 'LinkedIn'
    },
    { 
      icon: <Mail className="w-6 h-6" />, 
      url: 'niveshjain41@gmail.com', 
      label: 'Email'
    },
    { 
      icon: <Github className="w-6 h-6" />, 
      url: 'https://github.com/NiveshJain41', 
      label: 'GitHub'
    },
    { 
      icon: <Instagram className="w-6 h-6" />, 
      url: 'https://www.instagram.com/_jain_niv_/', 
      label: 'Instagram'
    },

  ];
  
  return (
    <div className="flex justify-center md:justify-start space-x-6">
      {socialLinks.map((link, index) => (
        <a
          key={index}
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-silver/70 hover:text-purple transform transition-all duration-300 hover:scale-110"
          aria-label={link.label}
        >
          {link.icon}
        </a>
      ))}
    </div>
  );
};

export default SocialLinks;
