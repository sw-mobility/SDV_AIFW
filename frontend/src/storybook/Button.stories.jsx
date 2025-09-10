import Button from '../components/ui/atoms/Button.jsx';

export default {
  title: 'UI/Button',
  component: Button,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    variant: {
      control: { type: 'select' },
      options: ['primary', 'secondary', 'outline', 'ghost', 'danger'],
    },
    size: {
      control: { type: 'select' },
      options: ['small', 'medium', 'large'],
    },
    disabled: {
      control: { type: 'boolean' },
    },
    onClick: { action: 'clicked' },
  },
};

export const Primary = {
  args: {
    children: 'Primary Button',
    variant: 'primary',
  },
};

export const Secondary = {
  args: {
    children: 'Secondary Button',
    variant: 'secondary',
  },
};

export const Outline = {
  args: {
    children: 'Outline Button',
    variant: 'outline',
  },
};

export const Ghost = {
  args: {
    children: 'Ghost Button',
    variant: 'ghost',
  },
};

export const Danger = {
  args: {
    children: 'Danger Button',
    variant: 'danger',
  },
};

export const Small = {
  args: {
    children: 'Small Button',
    size: 'small',
  },
};

export const Large = {
  args: {
    children: 'Large Button',
    size: 'large',
  },
};

export const Disabled = {
  args: {
    children: 'Disabled Button',
    disabled: true,
  },
};

export const AllVariants = {
  render: () => (
    <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', padding: '20px', background: '#f5f5f5' }}>
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="danger">Danger</Button>
    </div>
  ),
};

export const AllSizes = {
  render: () => (
    <div style={{ display: 'flex', gap: '10px', alignItems: 'center', padding: '20px', background: '#f5f5f5' }}>
      <Button size="small">Small</Button>
      <Button size="medium">Medium</Button>
      <Button size="large">Large</Button>
    </div>
  ),
};

export const CSSModuleTest = {
  render: () => (
    <div style={{ padding: '20px', background: '#f5f5f5' }}>
      <h3>CSS Module Test</h3>
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '10px' }}>
        <Button variant="primary">Primary (Blue)</Button>
        <Button variant="secondary">Secondary (Outline)</Button>
        <Button variant="outline">Outline (Gray)</Button>
        <Button variant="ghost">Ghost (Transparent)</Button>
        <Button variant="danger">Danger (Red)</Button>
      </div>
    </div>
  ),
}; 