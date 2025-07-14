import React, { useEffect } from 'react';
import Navbar from '@/components/Navbar';
import Section from '@/components/Section';
import PortfolioCard from '@/components/PortfolioCard';
import SkillBar from '@/components/SkillBar';
import ContactForm from '@/components/ContactForm';
import SocialLinks from '@/components/SocialLinks';
import CustomCursor from '@/components/CustomCursor';
import AnimatedBackground from '@/components/AnimatedBackground';
import InteractiveText from '@/components/InteractiveText';
import TiltCard from '@/components/TiltCard';
import ScrollReveal from '@/components/ScrollReveal';
// Importing icons from lucide-react for various sections
import { ArrowDown, User, Briefcase, Award, Code, Palette, Layout, Mail } from 'lucide-react';
// Importing motion from framer-motion for animations
import { motion } from 'framer-motion';
// Importing useQuery from @tanstack/react-query for data fetching
import { useQuery } from '@tanstack/react-query';
// Importing the API function to fetch projects
import { fetchAllProjects } from '@/lib/api';

const Index = () => {
  // Fetching projects data using react-query
  const { data: projects = [], isLoading } = useQuery({
    queryKey: ['projects'], // Unique key for this query
    queryFn: fetchAllProjects // Function to call to fetch data
  });

  // Defining an array of skills with their percentages
  const skills = [
    { skill: 'Data Structures and Algorithms', percentage: 90 },
    { skill: 'AI/ML', percentage: 85 },
    { skill: 'Deep Learning', percentage: 75 },
    { skill: 'Web Development', percentage: 75 },
    { skill: 'Cloud Services', percentage: 60 },
    { skill: 'DevOps', percentage: 50 },
  ];

  // useEffect hook for managing side effects like custom cursor and parallax
  useEffect(() => {
    // Disable default cursor
    document.body.style.cursor = 'none';
    // Check if the device is a touch device, if so, use auto cursor
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (isTouchDevice) document.body.style.cursor = 'auto';

    // Function to handle mouse movement for parallax effect
    const handleMouseMove = (e: MouseEvent) => {
      // Select all elements with the 'parallax-layer' class
      const parallaxElements = document.querySelectorAll('.parallax-layer');
      parallaxElements.forEach(element => {
        // Get the speed from data-speed attribute or default to 0.9
        const speed = (element as HTMLElement).dataset.speed || "0.9";
        // Calculate translation based on mouse position and speed
        const x = (window.innerWidth - e.pageX * parseFloat(speed)) / 100;
        const y = (window.innerHeight - e.pageY * parseFloat(speed)) / 100;
        // Apply transform to create the parallax effect
        (element as HTMLElement).style.transform = `translateX(${x}px) translateY(${y}px)`;
      });
    };

    // Add mousemove event listener
    document.addEventListener('mousemove', handleMouseMove);
    // Cleanup function: revert cursor and remove event listener on unmount
    return () => {
      document.body.style.cursor = 'auto';
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []); // Empty dependency array means this effect runs once on mount and cleans up on unmount

  return (
    <div className="min-h-screen">
      {/* Custom Cursor component */}
      <CustomCursor />
      {/* Animated Background component */}
      <AnimatedBackground />
      {/* Navbar component */}
      <Navbar />

      {/* Hero Section */}
      <section id="home" className="min-h-screen flex items-center justify-center relative parallax-section">
        {/* Overlay for hero section background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="hero-overlay"></div>
        </div>
        {/* Content for the hero section */}
        <div className="container-custom text-center relative z-10">
          {/* Animated main title */}
          <motion.div initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <h1 className="text-6xl md:text-8xl font-bold mb-6">
              <span className="title-gradient">Nivesh Jain</span>
            </h1>
          </motion.div>
          {/* Animated subtitle */}
<motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.3 }}>
{/* Removed max-w-2xl */}
<p className="text-xl md:text-2xl text-silver/80 mb-10 mx-auto whitespace-nowrap text-center">
  Amazon Hackon 5.0 || President- College Society || Machine Learning || Deep Learning
</p>
</motion.div>
          {/* Animated scroll down arrow */}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1, delay: 0.6 }} className="mt-12">
            <a href="#about" className="inline-block animate-float">
              <ArrowDown className="w-10 h-10 text-purple" />
            </a>
          </motion.div>
        </div>
      </section>

      {/* About Section */}
      <Section id="about" title="About Me">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Scroll reveal for profile card */}
          <ScrollReveal effect="slide-right">
            <TiltCard className="bg-navy-800/50 p-6 rounded-xl backdrop-blur-sm" tiltIntensity={7}>
              <div className="flex flex-col items-center text-center">
                <div className="w-48 h-48 rounded-full overflow-hidden mb-6 border-4 border-purple/20">
                  <img src="public\ProfilePhoto.jpg" alt="Profile" className="w-full h-full object-cover" />
                </div>
                <h3 className="text-2xl font-bold mb-2">Nivesh Jain</h3>
                <p className="text-silver/70 mb-4">Software Engineer</p>
              </div>
            </TiltCard>
          </ScrollReveal>
          {/* Scroll reveal for about text */}
          <ScrollReveal effect="slide-left" delay={200}>
            <div>
              <h3 className="text-2xl font-bold mb-4">
                I create <InteractiveText className="text-gradient">digital experiences</InteractiveText> that people love
              </h3>
              <p className="text-silver/80 mb-6">
              As a passionate engineering student with hands-on experience in AI/ML, web development, and system design, I specialize in building intelligent, user-centric digital solutions that combine performance with creativity.
              </p>
              <p className="text-silver/80 mb-8">
              Whether it's developing intelligent systems, building responsive web applications, or experimenting with AI-driven features, I approach each project with curiosity, precision, and a drive for impactful innovation.
              </p>
              {/* Feature list */}
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center"><Code className="w-6 h-6 text-purple mr-2" /><span>Clean Code</span></div>
                <div className="flex items-center"><Palette className="w-6 h-6 text-purple mr-2" /><span>Creative Design</span></div>
                <div className="flex items-center"><Layout className="w-6 h-6 text-purple mr-2" /><span>Intelligent Systems</span></div>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </Section>

      {/* Portfolio Section */}
      <Section id="portfolio" title="My Projects">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {isLoading // Show loading pulsate effect while projects are loading
            ? Array(6).fill(0).map((_, index) => (
                <div key={index} className="animate-pulse bg-navy-800/50 rounded-xl h-64"></div>
              ))
            : projects.map((project, index) => (
                // Scroll reveal for each project card
                <ScrollReveal key={project.id} effect={index % 2 === 0 ? 'slide-right' : 'slide-left'} delay={index * 100}>
                  <PortfolioCard
                    id={project.id}
                    title={project.title}
                    description={project.description}
                    image={project.image}
                    tags={project.tags}
                  />
                </ScrollReveal>
              ))
          }
        </div>
      </Section>

      {/* Skills Section */}
      <Section id="skills" title="My Skills">
        <div className="grid md:grid-cols-2 gap-12">
          {/* Scroll reveal for technical expertise */}
          <ScrollReveal effect="fade-in">
            <h3 className="text-2xl font-bold mb-6">Technical Expertise</h3>
            {skills.map((skill, index) => (
              // Scroll reveal for each skill bar
              <ScrollReveal key={index} delay={index * 100} effect="slide-right">
                <SkillBar skill={skill.skill} percentage={skill.percentage} />
              </ScrollReveal>
            ))}
          </ScrollReveal>
          {/* Scroll reveal for professional experience */}
          <ScrollReveal effect="scale" delay={300}>
            <TiltCard className="bg-navy-800/50 p-6 rounded-xl backdrop-blur-sm h-full" tiltIntensity={5}>
              <h3 className="text-2xl font-bold mb-6">College Professional Experience</h3>
              <div className="space-y-6">
                {/* Experience entries */}
                <div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 rounded-full bg-purple mr-2"></div>
                    <h4 className="font-bold">President (2025-Present)</h4>
                  </div>
                  <p className="text-md text-silver/90 mb-1">Rangmanch – Arts & Performing Society, GEU</p>
                  <p className="text-sm text-silver/90">
                  Led 170+ students in cultural events and performances. <br/>Handled creative direction and team coordination.
                  </p>
                </div>
                <div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 rounded-full bg-teal mr-2"></div>
                    <h4 className="font-bold">Technical Head</h4>
                  </div>
                  <p className="text-sm text-silver/70 mb-1">IEEE – Technical Society, GEU</p>
                  <p className="text-sm text-silver/80">
                  Led technical activities and coding events. <br/>
                  Managed project builds and tech setup.
                  </p>
                </div>
                <div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 rounded-full bg-purple mr-2"></div>
                    <h4 className="font-bold">Management Heade</h4>
                  </div>
                  <p className="text-sm text-silver/70 mb-1">ACM – Technical Society, GEU</p>
                  <p className="text-sm text-silver/80">
                  Planned and managed tech events and workshops. <br/>
                  Ensured smooth execution and resource handling.
                  </p>
                </div>
              </div>
            </TiltCard>
          </ScrollReveal>
        </div>
      </Section>

      {/* Education Section */}
      <Section id="education" title="Education">
        <div className="grid md:grid-cols-2 gap-12">
          {/* Scroll reveal for academic background */}
          <ScrollReveal effect="slide-right">
            <TiltCard className="bg-navy-800/50 p-6 rounded-xl backdrop-blur-sm h-full" tiltIntensity={5}>
              <h3 className="text-2xl font-bold mb-6">Academic Background</h3>
              <div className="space-y-6">
                {/* Education entries */}
                <div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 rounded-full bg-purple mr-2"></div>
                    <h4 className="font-bold">B.Tech in Computer Science</h4>
                  </div>
                  <p className="text-sm text-silver/70 mb-1">Graphic Era Deemed to be University(2022 - Present)</p>
                  <p className="text-sm text-silver/80">
                    Relevant Subjects: DSA, AI/ML, OS, Web Dev, Compiler Design, DBMS.
                  </p>
                </div>
                <br />
                <div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 rounded-full bg-teal mr-2"></div>
                    <h4 className="font-bold">Senior Secondary Education</h4>
                  </div>
                  <p className="text-sm text-silver/70 mb-1">MD International School,(2019 - 2021)</p>
                  <p className="text-sm text-silver/80">
                    PCM Stream. Secured 83% in CBSE Board Exams.
                  </p>
                </div>
                <br />
                <div>
                <div className="flex items-center mb-2">
                  <div className="w-2 h-2 rounded-full bg-purple mr-2"></div> {/* You can choose a different color if you like, e.g., bg-teal */}
                  <h4 className="font-bold">Secondary School Education</h4>
                </div>
                <p className="text-sm text-silver/70 mb-1">St. Peter's School, Kiratpur (2018 - 2019)</p> {/* Assuming the same school and year */}
                <p className="text-sm text-silver/80">
                  Secured 89% in ICSE Board Exams.
                </p>
              </div>
              </div>
            </TiltCard>
          </ScrollReveal>
          {/* Scroll reveal for education image */}
          <ScrollReveal effect="fade-in" delay={300}>
            <div className="flex items-center justify-center h-50">
              <img
                src="educationBuildings.jpg"
                alt="Education"
                className="rounded-xl shadow-lg w-full max-w-xs"
              />
            </div>
          </ScrollReveal>
        </div>
      </Section>

      {/* Contact Section */}
      <Section id="contact" title="Get In Touch">
        <div className="grid md:grid-cols-2 gap-12">
          {/* Scroll reveal for contact information */}
          <ScrollReveal effect="slide-right">
            <div>
              <h3 className="text-2xl font-bold mb-6">Let's Connect</h3>
              <p className="text-silver/80 mb-8">
                I'm always open to discussing new projects, creative ideas or job opportunities to be part of your vision.
              </p>
              <div className="mb-8">
                <h4 className="text-lg font-bold mb-4">Find me on</h4>
                {/* Social Links component */}
                <SocialLinks />
              </div> 
              <TiltCard className="bg-navy-800/50 p-6 rounded-xl backdrop-blur-sm" tiltIntensity={3}>
                <div className="flex items-center mb-4">
                  <Mail className="w-6 h-6 text-purple mr-3" />
                  <div>
                    <h4 className="font-bold">Email</h4>
                    <p className="text-sm text-silver/80">niveshjain41@gmail.com</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <Briefcase className="w-6 h-6 text-purple mr-3" />
                  <div>
                    <h4 className="font-bold">Availability</h4>
                    <p className="text-sm text-silver/80">Open to new job opportunities</p>
                  </div>
                </div>
              </TiltCard>
            </div>
          </ScrollReveal>
          {/* Scroll reveal for contact form */}
          <ScrollReveal effect="slide-left" delay={200}>
            <ContactForm />
          </ScrollReveal>
        </div>
      </Section>

      {/* Footer */}
      <footer className="py-8 border-t border-navy-600/30">
        <div className="container-custom text-center">
          <p className="text-silver/60">
            &copy; {new Date().getFullYear()} Nivesh Jain's Portfolio.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;