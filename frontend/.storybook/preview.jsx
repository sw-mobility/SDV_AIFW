import '../src/index.css';

/** @type { import('@storybook/react-vite').Preview } */
const preview = {
  decorators: [
    (Story) => (
      <div style={{ fontFamily: 'var(--font-family)', color: 'var(--color-text-main)' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    controls: {
      matchers: {
       color: /(background|color)$/i,
       date: /Date$/i,
      },
    },

    a11y: {
      // 'todo' - show a11y violations in the test UI only
      // 'error' - fail CI on a11y violations
      // 'off' - skip a11y checks entirely
      test: "todo"
    }
  },
};

export default preview;