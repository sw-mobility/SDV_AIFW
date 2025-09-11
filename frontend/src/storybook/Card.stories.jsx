import Card, { CardGrid } from '../components/ui/atoms/Card.jsx';
import Button from '../components/ui/atoms/Button.jsx';

export default {
  title: 'UI/Card',
  component: Card,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    onClick: { action: 'clicked' },
  },
};

export const Default = {
  args: {
    children: (
      <div>
        <h3>Card Title</h3>
        <p>This is a basic card with some content.</p>
      </div>
    ),
  },
};

export const WithButton = {
  args: {
    children: (
      <div>
        <h3>Card with Button</h3>
        <p>This card contains a button.</p>
        <Button variant="primary">Action Button</Button>
      </div>
    ),
  },
};

export const Clickable = {
  args: {
    children: (
      <div>
        <h3>Clickable Card</h3>
        <p>Click this card to trigger an action.</p>
      </div>
    ),
    onClick: () => alert('Card clicked!'),
  },
};

export const CardGridExample = {
  render: () => (
    <CardGrid gap="1rem">
      <Card>
        <h3>Card 1</h3>
        <p>First card in the grid.</p>
      </Card>
      <Card>
        <h3>Card 2</h3>
        <p>Second card in the grid.</p>
      </Card>
      <Card>
        <h3>Card 3</h3>
        <p>Third card in the grid.</p>
      </Card>
    </CardGrid>
  ),
};

export const ComplexContent = {
  args: {
    children: (
      <div>
        <h3>Complex Card</h3>
        <p>This card has more complex content with multiple elements.</p>
        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
          <Button variant="primary" size="small">Save</Button>
          <Button variant="outline" size="small">Cancel</Button>
        </div>
      </div>
    ),
  },
}; 