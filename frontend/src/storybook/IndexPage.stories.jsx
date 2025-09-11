import { BrowserRouter } from 'react-router-dom';
import IndexPage from '../pages/index_page/IndexPage.jsx';

export default {
  title: 'Pages/IndexPage',
  component: IndexPage,
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
  render: () => <IndexPage />,
};

export const Loading = {
  render: () => <IndexPage mockState={{ loading: true }} />,
};

export const Error = {
  render: () => <IndexPage mockState={{ error: true }} />,
};

export const Empty = {
  render: () => <IndexPage mockState={{ empty: true }} />,
};

export const ProjectsTab = {
  render: () => <IndexPage />,
  parameters: {
    reactRouter: {
      initialEntries: ['/'],
    },
  },
};

export const DataTab = {
  render: () => <IndexPage />,
  parameters: {
    reactRouter: {
      initialEntries: ['/'],
    },
  },
}; 