import { projectOne } from './projects/project-one';
// import { projectTwo } from './projects/project-two';

const allProjects = [projectOne];

export const fetchAllProjects = async () => {
  await new Promise(res => setTimeout(res, 500));
  return allProjects.map(({ id, title, description, image, tags }) => ({
    id, title, description, image, tags
  }));
};

export const fetchProjectDetails = async (id: string) => {
  await new Promise(res => setTimeout(res, 500));
  const project = allProjects.find(p => p.id === id);
  if (!project) throw new Error("Project not found");
  return project;
};
