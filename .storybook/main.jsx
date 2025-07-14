

/** @type { import('@storybook/react-vite').StorybookConfig } */
const config = {
  "stories": [
    "../src/**/*.mdx",
    "../src/**/*.stories.@(js|jsx|mjs|ts|tsx)"
  ],
  "addons": [
    "@chromatic-com/storybook",
    "@storybook/addon-docs",
    "@storybook/addon-a11y",
    "@storybook/addon-vitest"
  ],
  "framework": {
    "name": "@storybook/react-vite",
    "options": {}
  },
  "viteFinal": async (config) => {
    // CSS 모듈 설정
    config.css = {
      modules: {
        localsConvention: 'camelCase',
      },
    };
    
    // CSS 파일 처리 설정
    config.define = {
      ...config.define,
      'process.env': {},
    };
    
    return config;
  },
};
export default config;