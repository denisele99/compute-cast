/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        ink: '#050509',
        panel: '#101018',
        line: '#2a2a39',
        violet: '#9275ff',
        cyan: '#39d9e8',
      },
      boxShadow: {
        glow: '0 0 50px rgba(76, 42, 255, 0.22)',
      },
    },
  },
  plugins: [],
};
