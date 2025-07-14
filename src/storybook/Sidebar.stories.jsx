import { BrowserRouter } from 'react-router-dom';
import Sidebar from '../components/layout/Sidebar.jsx';

export default {
  title: 'Layout/Sidebar',
  component: Sidebar,
  parameters: {
    layout: 'fullscreen',
  },
  decorators: [
    (Story) => (
      <BrowserRouter>
        <Story />
      </BrowserRouter>
    ),
  ],
};

export const Default = {
  render: () => <Sidebar />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123'],
    },
  },
};

export const ActiveHome = {
  render: () => <Sidebar />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123'],
    },
  },
};

export const ActiveLabelling = {
  render: () => <Sidebar />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123/labelling'],
    },
  },
};

export const ActiveTraining = {
  render: () => <Sidebar />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123/training'],
    },
  },
}; 