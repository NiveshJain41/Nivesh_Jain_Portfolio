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
  title: "E-Commerce Trust",
  description: "Simulates and detects bot behavior and fake reviews in e-commerce using machine learning.",
  longDescription:
  "E-Commerce Trust is a comprehensive fraud detection system designed for e-commerce platforms like Amazon. It simulates human and bot purchases, captures behavioral metrics like session timing, IP reuse, and coupon patterns, and detects fake reviews using ML. Built with Python and Streamlit, it features real-time data logging, Excel integration, and supports multiple classification models including Random Forest, XGBoost, LightGBM, and ensemble methods.",
  image,
  tags: ["Machine Learning", "XGBoost", "LightGBM"],
  demoUrl: "https://example.com/demo",
  repoUrl: "https://github.com/NiveshJain41/AmazonTrust",
  videoUrl: "https://www.youtube.com/embed/dQ9WgXcQ",
  readme,
  files,
  features: [
    "Fake review detection for scripted, AI-generated, and hijacked content",
    "Multi-model ML pipeline with support for Random Forest, XGBoost, LightGBM, and ensemble classification",
    "Excel-based logging of user sessions, IPs, timestamps, and account activity",
    "TF-IDF and custom feature extraction from review text",
  ]
};

