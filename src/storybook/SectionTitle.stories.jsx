import SectionTitle from '../components/ui/SectionTitle.jsx';

export default {
  title: 'UI/SectionTitle',
  component: SectionTitle,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    size: {
      control: { type: 'select' },
      options: ['sm', 'md', 'lg'],
    },
  },
};

export const Default = {
  args: {
    children: 'Section Title',
    size: 'lg',
  },
};

export const Small = {
  args: {
    children: 'Small Section Title',
    size: 'sm',
  },
};

export const Medium = {
  args: {
    children: 'Medium Section Title',
    size: 'md',
  },
};

export const Large = {
  args: {
    children: 'Large Section Title',
    size: 'lg',
  },
};

export const AllSizes = {
  render: () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <SectionTitle size="sm">Small Section Title</SectionTitle>
      <SectionTitle size="md">Medium Section Title</SectionTitle>
      <SectionTitle size="lg">Large Section Title</SectionTitle>
    </div>
  ),
}; 