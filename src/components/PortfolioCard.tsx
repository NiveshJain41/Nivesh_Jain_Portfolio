
import React from 'react';
import { cn } from '@/lib/utils';
import { Link } from 'react-router-dom';
import TiltCard from './TiltCard';

interface PortfolioCardProps {
  title: string;
  description: string;
  image: string;
  tags: string[];
  link?: string;
  className?: string;
  id: string; // Add id for the project
}

const PortfolioCard: React.FC<PortfolioCardProps> = ({
  title,
  description,
  image,
  tags,
  link,
  className = "",
  id
}) => {
  return (
    <Link to={`/projects/${id}`} className={cn("block", className)}>
      <TiltCard className="hover-card group" tiltIntensity={5}>
        <img 
          src={image}
          alt={title}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
        />
        <div className="hover-card-content">
          <h3 className="text-xl font-bold mb-2 text-white">{title}</h3>
          <p className="text-silver/80 mb-4 text-sm">{description}</p>
          <div className="flex flex-wrap gap-2 mb-4">
            {tags.map((tag, index) => (
              <span key={index} className="text-xs px-2 py-1 bg-navy/50 text-silver/70 rounded-full">
                {tag}
              </span>
            ))}
          </div>
          <span className="inline-block text-teal underline-animation">
            View Project Details
          </span>
        </div>
      </TiltCard>
    </Link>
  );
};

export default PortfolioCard;
