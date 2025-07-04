import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, Github, ExternalLink, FileCode, Code, FileText, Video } from 'lucide-react';
import { fetchProjectDetails } from '@/lib/api';
import { motion } from 'framer-motion';
import ScrollReveal from '@/components/ScrollReveal';
import TiltCard from '@/components/TiltCard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

interface ProjectFile {
  name: string;
  content: string;
}

interface Project {
  id: string;
  title: string;
  description: string;
  longDescription: string;
  image: string;
  tags: string[];
  demoUrl: string;
  repoUrl: string;
  videoUrl: string;
  files: ProjectFile[];
  readme: string;
  features: string[];
}

const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const { data: project, isLoading, error } = useQuery({
    queryKey: ['project', id],
    queryFn: () => fetchProjectDetails(id as string),
  });

  useEffect(() => {
    if (project) {
      document.title = `${project.title} | Project Detail`;
    }
    window.scrollTo(0, 0);
  }, [project]);

  useEffect(() => {
    if (project && project.files?.length > 0) {
      setSelectedFile(project.files[0].name);
    }
  }, [project]);

  const getFileType = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'js':
      case 'jsx': return 'JavaScript';
      case 'ts':
      case 'tsx': return 'TypeScript';
      case 'css': return 'CSS';
      case 'html': return 'HTML';
      case 'json': return 'JSON';
      default: return 'Code';
    }
  };

  const getFileContent = (fileName: string) => {
    return project?.files?.find(f => f.name === fileName)?.content || '';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-navy-900/80 backdrop-blur-md rounded-lg p-8">
          <div className="w-16 h-16 border-4 border-t-purple rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center">
        <div className="bg-navy-900/80 backdrop-blur-md rounded-lg p-8 text-center">
          <h2 className="text-2xl font-bold mb-4">Project not found</h2>
          <Button asChild><Link to="/">Back to Home</Link></Button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen py-16">
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: "url('/images/cosmic-bg.jpg')" }}
      ></div>
      <div className="absolute inset-0 bg-black opacity-30"></div>

      <div className="relative z-10 container-custom">
        <motion.div 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ duration: 0.5 }}
          className="bg-navy-900/70 backdrop-blur-md rounded-2xl p-8 shadow-2xl"
        >
          <div className="bg-navy-800/60 backdrop-blur-sm rounded-lg p-3 inline-block mb-8">
            <Link to="/" className="flex items-center text-silver/70 hover:text-purple transition-colors">
              <ArrowLeft className="w-4 h-4 mr-2" /> Back to Projects
            </Link>
          </div>

          <div className="bg-navy-800/40 backdrop-blur-sm rounded-xl p-6 mb-8">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 text-white">{project.title}</h1>
            <div className="flex flex-wrap gap-2 mb-4">
              {project.tags.map((tag, i) => (
                <Badge key={i} variant="outline" className="bg-navy-800/70 text-silver border-purple/30">{tag}</Badge>
              ))}
            </div>
            <p className="text-xl text-silver/90 mb-0">{project.description}</p>
          </div>

          <Tabs defaultValue="overview" className="w-full mt-12">
            <TabsList className="grid grid-cols-4 mb-8 bg-navy-800/70 backdrop-blur-md p-1 rounded-lg border border-purple/20">
              <TabsTrigger value="overview" className="data-[state=active]:bg-purple/30 data-[state=active]:text-white">Overview</TabsTrigger>
              <TabsTrigger value="code" className="data-[state=active]:bg-purple/30 data-[state=active]:text-white">Code Examples</TabsTrigger>
              <TabsTrigger value="docs" className="data-[state=active]:bg-purple/30 data-[state=active]:text-white">Documentation</TabsTrigger>
              <TabsTrigger value="video" className="data-[state=active]:bg-purple/30 data-[state=active]:text-white">Demo Video</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-8">
              <ScrollReveal effect="fade-in">
                <div className="grid md:grid-cols-2 gap-8 items-center bg-navy-800/50 backdrop-blur-sm rounded-xl p-6">
                  <div className="bg-navy-900/50 rounded-lg p-4">
                    <img src={project.image} alt={project.title} className="rounded-xl w-full object-cover" />
                  </div>
                  <div className="space-y-6 bg-navy-900/30 backdrop-blur-sm rounded-lg p-6">
                    <h2 className="text-2xl font-bold text-white">Project Overview</h2>
                    <p className="text-silver/90">{project.longDescription}</p>
                    <div className="flex flex-col sm:flex-row gap-4">
                      <Button asChild className="bg-purple hover:bg-purple/80">
                        <a href={project.demoUrl} target="_blank">
                          <ExternalLink className="w-4 h-4 mr-2" /> Live Demo
                        </a>
                      </Button>
                      <Button asChild variant="outline" className="border-purple/50 hover:bg-purple/20">
                        <a href={project.repoUrl} target="_blank">
                          <Github className="w-4 h-4 mr-2" /> Source Code
                        </a>
                      </Button>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
              
              <ScrollReveal effect="fade-in" delay={100}>
                <div className="grid md:grid-cols-2 gap-6">
                  {project.features.map((feature, index) => (
                    <TiltCard key={index} className="bg-navy-800/60 backdrop-blur-sm p-6 rounded-xl border border-purple/20">
                      <div className="flex gap-3 items-start">
                        <div className="w-2 h-2 mt-2 rounded-full bg-purple"></div>
                        <span className="text-silver/90">{feature}</span>
                      </div>
                    </TiltCard>
                  ))}
                </div>
              </ScrollReveal>
            </TabsContent>

            <TabsContent value="code">
              <div className="grid grid-cols-12 h-96 gap-4 bg-navy-800/50 backdrop-blur-sm rounded-xl p-4">
                <div className="col-span-4 bg-navy-900/80 backdrop-blur-md rounded-lg p-3 overflow-y-auto border border-purple/20">
                  <h4 className="text-sm uppercase text-silver/70 mb-3 bg-navy-800/50 p-2 rounded">Project Files</h4>
                  <div className="space-y-1">
                    {project.files.map((file, index) => (
                      <button
                        key={index}
                        onClick={() => setSelectedFile(file.name)}
                        className={`flex items-center gap-2 text-left w-full px-3 py-2 rounded-md transition-colors ${
                          selectedFile === file.name 
                            ? 'bg-purple/30 text-purple border border-purple/50' 
                            : 'hover:bg-navy-800/70 text-silver/80'
                        }`}
                      >
                        <FileCode className="w-4 h-4" /> 
                        <span className="truncate">{file.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="col-span-8 bg-navy-900/80 backdrop-blur-md rounded-lg p-4 overflow-hidden border border-purple/20">
                  {selectedFile && (
                    <div className="h-full flex flex-col">
                      <div className="flex items-center justify-between mb-3 bg-navy-800/50 p-3 rounded">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-white">{selectedFile}</span>
                          <Badge variant="outline" className="text-xs bg-purple/20 border-purple/50 text-purple">{getFileType(selectedFile)}</Badge>
                        </div>
                      </div>
                      <Separator className="mb-3 bg-purple/30" />
                      <div className="bg-black/50 rounded p-3 flex-1 overflow-auto">
                        <pre className="text-sm font-mono text-green-400">
                          <code>{getFileContent(selectedFile)}</code>
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="docs">
              <div className="bg-navy-900/80 backdrop-blur-md rounded-lg p-4 h-96 overflow-y-auto border border-purple/20">
                <div className="prose prose-invert max-w-none">
                  {project.readme.split('\n').map((line, i) => {
                    if (line.startsWith('# ')) return <h1 key={i} className="text-white bg-navy-800/50 p-3 rounded">{line.substring(2)}</h1>;
                    if (line.startsWith('## ')) return <h2 key={i} className="text-purple bg-navy-800/30 p-2 rounded">{line.substring(3)}</h2>;
                    if (line.startsWith('### ')) return <h3 key={i} className="text-silver bg-navy-800/20 p-2 rounded">{line.substring(4)}</h3>;
                    if (line.startsWith('```')) return <pre key={i} className="bg-black/60 p-3 rounded"><code className="text-green-400">{line.substring(3)}</code></pre>;
                    if (line.startsWith('- ')) return <li key={i} className="text-silver/90 bg-navy-800/20 p-1 rounded mb-1">{line.substring(2)}</li>;
                    if (line === '') return <br key={i} />;
                    return <p key={i} className="text-silver/90 bg-navy-800/10 p-2 rounded mb-2">{line}</p>;
                  })}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="video">
              <div className="aspect-video relative rounded-xl overflow-hidden bg-navy-900/80 backdrop-blur-md p-4 border border-purple/20">
                <iframe 
                  src={project.videoUrl} 
                  title="Project Demo Video" 
                  className="absolute inset-4 w-[calc(100%-2rem)] h-[calc(100%-2rem)] rounded-lg" 
                  allowFullScreen
                ></iframe>
              </div>
            </TabsContent>
          </Tabs>

          <div className="text-center mt-16 bg-navy-800/50 backdrop-blur-sm rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 text-white">Want to see more projects?</h3>
            <Button asChild size="lg" className="bg-purple hover:bg-purple/80">
              <Link to="/">Back to Portfolio</Link>
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ProjectDetail;