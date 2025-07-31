import CodeEditor from '../components/ui/CodeEditor.jsx';

export default {
  title: 'UI/CodeEditor',
  component: CodeEditor,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {
  render: () => <CodeEditor />,
};

export const WithCustomCode = {
  render: () => {
    // CodeEditor 컴포넌트는 내부적으로 상태를 관리하므로
    // 기본 렌더링만 제공합니다
    return <CodeEditor />;
  },
};

export const FullScreen = {
  parameters: {
    layout: 'fullscreen',
  },
  render: () => <CodeEditor />,
}; 