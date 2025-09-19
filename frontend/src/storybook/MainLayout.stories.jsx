import { BrowserRouter } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout.jsx';

export default {
  title: 'Layout/MainLayout',
  component: MainLayout,
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
  render: () => <MainLayout />,
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123'],
    },
  },
};

export const WithContent = {
  render: () => (
    <MainLayout>
      <div style={{ padding: '20px' }}>
        <h1>Main Content Area</h1>
        <p>This is the main content that would be rendered by the Outlet component.</p>
        <p>You can see how the layout components work together.</p>
      </div>
    </MainLayout>
  ),
  parameters: {
    reactRouter: {
      initialEntries: ['/projects/123'],
    },
  },
}; 