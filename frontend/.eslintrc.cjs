module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  settings: {
    react: {
      version: 'detect',
    },
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended-type-checked',
    'plugin:@typescript-eslint/stylistic-type-checked',
    'plugin:react-hooks/recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'prettier',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    // TypeScript validates component props; the React rule only understands
    // runtime PropTypes and reports false positives for typed Radix wrappers.
    'react/prop-types': 'off',
    // Shared hooks and style helpers are intentionally exported beside the
    // small UI components that consume them.
    'react-refresh/only-export-components': 'off',
  },
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json', './tsconfig.vite.json'],
    tsconfigRootDir: __dirname,
  },
};
