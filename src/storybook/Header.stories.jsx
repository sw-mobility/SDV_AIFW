import { BrowserRouter } from 'react-router-dom';
import Header from '../components/layout/Header.jsx';

export default {
  title: 'Layout/Header',
  component: Header,
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
  render: () => <Header />,
};

export const WithDatasetButton = {
  render: () => <Header />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123'],
    },
  },
};

export const WithoutDatasetButton = {
  render: () => <Header />,
  parameters: {
    reactRouter: {
      initialEntries: ['/'],
    },
  },
}; 