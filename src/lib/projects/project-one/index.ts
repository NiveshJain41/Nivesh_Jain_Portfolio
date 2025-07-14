import image from './image.jpg';
import readme from './readme.md?raw';

// Dynamically import all code files from /files
const fileImports = import.meta.glob("./files/*", { as: "raw", eager: true });

const files = Object.entries(fileImports).map(([path, content]) => {
  const parts = path.split("/");
  const fileName = parts[parts.length - 1];
  return {
    name: fileName,
    content: content as string
  };
});

export const projectOne = {
  id: "project-one",
  title: "Project One",
  description: "A responsive web application with modern UI/UX.",
  longDescription:
    "This project showcases a responsive web application built using React and TailwindCSS. It features clean UI patterns, animations, and cross-device compatibility for seamless user experiences.",
  image,
  tags: ["React", "TailwindCSS", "TypeScript"],
  demoUrl: "https://example.com/demo",
  repoUrl: "https://github.com/username/project-one",
  videoUrl: "https://www.youtube.com/embed/dQw4w9WgXcQ",
  readme,
  files,
  features: [
    "Responsive design that adapts across devices",
    "Modern UI/UX with interactive animations",
    "Optimized performance and accessibility",
    "Created for a demo project 1"
  ]
};
