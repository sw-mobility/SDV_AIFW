import ProjectHomePage from '../pages/project_home/ProjectHomePage.jsx';

export default {
  title: 'Pages/ProjectHomePage',
  component: ProjectHomePage,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {
  render: () => <ProjectHomePage />,
}; 