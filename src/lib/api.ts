// Mock API service for project data

export const fetchProjectDetails = async (id: string) => {
  
  const allProjects = [
    {
      id: "project-one",
      title: "E-Commerce Trust",
      description: "A responsive web application with modern UI/UX",
      longDescription: "This project showcases a responsive web application built with React and TailwindCSS. It features modern UI/UX patterns, animations, and a seamless user experience across devices.",
      image: "https://images.unsplah.com/photo-1498050108023-c5249f4df085",
      tags: ["React", "TailwindCSS", "TypeScript"],
      demoUrl: "https://example.com/demo",
      repoUrl: "https://github.com/username/project-one",
      videoUrl: "https://www.youtube.com/embed/dQw4w9WgXcQ",
      files: [
        { name: "App.tsx", content: "import React from 'react';\n\nconst App = () => {\n  return <div>Hello World</div>;\n};\n\nexport default App;" },
        { name: "index.tsx", content: "import React from 'react';\nimport ReactDOM from 'react-dom';\nimport App from './App';\n\nReactDOM.render(<App />, document.getElementById('root'));" },
        { name: "styles.css", content: "body {\n  font-family: sans-serif;\n  margin: 0;\n  padding: 0;\n}" }
      ],
      readme: "# Project One\n\nThis is a responsive web application with modern UI/UX.\n\n## Installation\n\n```bash\nnpm install\nnpm start\n```\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n\n## Demo Credentials\n\nUsername: demo\nPassword: demo123",
      features: [
        "Responsive design that works on all devices",
        "Modern UI/UX with animations",
        "Optimized performance",
        "Accessibility compliant"
      ]
    },
    {
      id: "project-two",
      title: "Project Two",
      description: "Mobile app design with intuitive interaction patterns",
      longDescription: "A mobile application designed with user experience in mind. This project focuses on intuitive interaction patterns and smooth animations to provide a delightful user experience.",
      image: "https://images.unsplash.com/photo-1551650975-87deedd944c3",
      tags: ["UI/UX", "Mobile Design", "Figma"],
      demoUrl: "https://example.com/demo2",
      repoUrl: "https://github.com/username/project-two",
      videoUrl: "https://www.youtube.com/embed/dQw4w9WgXcQ",
      files: [
        { name: "App.js", content: "import React from 'react';\nimport { View, Text } from 'react-native';\n\nconst App = () => {\n  return (\n    <View>\n      <Text>Hello World</Text>\n    </View>\n  );\n};\n\nexport default App;" },
        { name: "styles.js", content: "import { StyleSheet } from 'react-native';\n\nexport default StyleSheet.create({\n  container: {\n    flex: 1,\n    backgroundColor: '#fff',\n  },\n});" }
      ],
      readme: "# Project Two\n\nMobile app design with intuitive interaction patterns.\n\n## Installation\n\n```bash\nnpm install\nnpm run ios # or npm run android\n```\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n\n## Demo Credentials\n\nUsername: demo\nPassword: demo123",
      features: [
        "Intuitive navigation",
        "Smooth animations",
        "Offline capabilities",
        "Push notifications"
      ]
    },
    // More projects can be added here
  ];
  
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));
  
  const project = allProjects.find(project => project.id === id);
  
  if (!project) {
    throw new Error('Project not found');
  }
  
  return project;
};

export const fetchAllProjects = async () => {
  

  //this deflect changes on home page
  const projects = [
    {
      id: "project-one",
      title: "Project One",
      description: "A responsive web application with modern UI/UX",
      image: "https://images.unsplash.com/photo-1498050108023-c5249f4df085",
      tags: ["React", "TailwindCSS", "TypeScript"],
    },
    {
      id: "project-two",
      title: "Project Two",
      description: "Mobile app design with intuitive interaction patterns",
      image: "https://images.unsplash.com/photo-1551650975-87deedd944c3",
      tags: ["UI/UX", "Mobile Design", "Figma"],
    },
    {
      id: "project-three",
      title: "Project Three",
      description: "E-commerce platform with seamless checkout experience",
      image: "https://images.unsplash.com/photo-1460925895917-afdab827c52f",
      tags: ["E-commerce", "Reac]fasjjt", "Node.js"],
    }
  ];
  
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));
  
  return projects;
};

// Mock project data for demo purposes
const projectsData = [
  {
    id: "1",
    title: "E-Commerce Platform",
    description: "A full-featured e-commerce platform with product management, cart functionality, and secure payment processing.",
    image: "https://source.unsplash.com/random/800x600?ecommerce",
    tags: ["React", "Node.js", "MongoDB", "Strigvhgvghpe"],
    url: "https://example.com/project1"
  },
  {
    id: "2",
    title: "Task Management App",
    description: "A collaborative task management application with real-time updates, task assignment, and progress tracking.",
    image: "https://source.unsplash.com/random/800x600?app",
    tags: ["React", "Firebase", "Material UI", "Redux"],
    url: "https://example.com/project2"
  },
  {
    id: "3",
    title: "AI Content Generator",
    description: "An AI-powered application that generates various types of content including articles, social media posts, and marketing copy.",
    image: "https://source.unsplash.com/random/800x600?ai",
    tags: ["Python", "TensorFlow", "React", "Flask"],
    url: "https://example.com/project3"
  }
];

// Fetch a specific project by ID
export const fetchProjectById = (id: string) => {
  // Simulate API request delay
  return new Promise((resolve) => {
    setTimeout(() => {
      const project = projectsData.find(project => project.id === id);
      resolve(project || null);
    }, 500);
  });
};
