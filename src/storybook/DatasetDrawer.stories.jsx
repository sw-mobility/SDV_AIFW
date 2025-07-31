import { useState } from 'react';
import DatasetDrawer from '../components/features/dataset/DatasetDrawer.jsx';

export default {
  title: 'Dataset/DatasetDrawer',
  component: DatasetDrawer,
  parameters: {
    layout: 'centered',
  },
};

const DatasetDrawerWrapper = ({ open, onClose }) => {
  return (
    <div>
      <DatasetDrawer open={open} onClose={onClose} />
    </div>
  );
};

export const Default = {
  render: () => {
    const [open, setOpen] = useState(false);
    return (
      <div>
        <button onClick={() => setOpen(true)}>Open Dataset Drawer</button>
        <DatasetDrawerWrapper open={open} onClose={() => setOpen(false)} />
      </div>
    );
  },
};

export const Open = {
  render: () => {
    const [open, setOpen] = useState(true);
    return (
      <div>
        <button onClick={() => setOpen(!open)}>Toggle Drawer</button>
        <DatasetDrawerWrapper open={open} onClose={() => setOpen(false)} />
      </div>
    );
  },
};

export const Controlled = {
  args: {
    open: true,
    onClose: () => console.log('Drawer closed'),
  },
  render: (args) => <DatasetDrawer {...args} />,
}; 