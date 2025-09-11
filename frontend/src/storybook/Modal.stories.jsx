import { useState } from 'react';
import Modal from '../components/ui/modals/Modal.jsx';
import Button from '../components/ui/atoms/Button.jsx';

export default {
  title: 'UI/Modal',
  component: Modal,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    isOpen: {
      control: { type: 'boolean' },
    },
    onClose: { action: 'closed' },
  },
};

const ModalWrapper = ({ isOpen, onClose, ...props }) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} {...props}>
      <p>This is the modal content. You can put any content here.</p>
      <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
        <Button variant="primary" onClick={onClose}>Confirm</Button>
        <Button variant="outline" onClick={onClose}>Cancel</Button>
      </div>
    </Modal>
  );
};

export const Default = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <div>
        <Button onClick={() => setIsOpen(true)}>Open Modal</Button>
        <ModalWrapper 
          isOpen={isOpen} 
          onClose={() => setIsOpen(false)}
          title="Default Modal"
        />
      </div>
    );
  },
};

export const WithCustomContent = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <div>
        <Button onClick={() => setIsOpen(true)}>Open Custom Modal</Button>
        <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Custom Content Modal">
          <div>
            <h4>Custom Modal Content</h4>
            <p>This modal has custom content with different styling.</p>
            <ul>
              <li>Feature 1</li>
              <li>Feature 2</li>
              <li>Feature 3</li>
            </ul>
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <Button variant="primary" onClick={() => setIsOpen(false)}>Save</Button>
              <Button variant="outline" onClick={() => setIsOpen(false)}>Cancel</Button>
            </div>
          </div>
        </Modal>
      </div>
    );
  },
};

export const LargeContent = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <div>
        <Button onClick={() => setIsOpen(true)}>Open Large Modal</Button>
        <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Large Content Modal">
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            <h4>Large Content Modal</h4>
            <p>This modal contains a lot of content to demonstrate scrolling.</p>
            {Array.from({ length: 20 }, (_, i) => (
              <p key={i}>This is paragraph {i + 1} of the large content modal.</p>
            ))}
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <Button variant="primary" onClick={() => setIsOpen(false)}>Confirm</Button>
              <Button variant="outline" onClick={() => setIsOpen(false)}>Cancel</Button>
            </div>
          </div>
        </Modal>
      </div>
    );
  },
};
