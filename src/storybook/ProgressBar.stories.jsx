import ProgressBar from '../shared/common/ProgressBar.jsx';

export default {
  title: 'UI/ProgressBar',
  component: ProgressBar,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    percentage: {
      control: { type: 'range', min: 0, max: 100, step: 1 },
    },
  },
};

export const Default = {
  args: {
    label: 'Upload Progress',
    percentage: 50,
  },
};

export const Complete = {
  args: {
    label: 'Download Complete',
    percentage: 100,
  },
};

export const Starting = {
  args: {
    label: 'Processing',
    percentage: 10,
  },
};

export const NoLabel = {
  args: {
    percentage: 75,
  },
};

export const AllStates = {
  render: () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '400px' }}>
      <ProgressBar label="Starting" percentage={10} />
      <ProgressBar label="In Progress" percentage={45} />
      <ProgressBar label="Almost Done" percentage={80} />
      <ProgressBar label="Complete" percentage={100} />
    </div>
  ),
}; 