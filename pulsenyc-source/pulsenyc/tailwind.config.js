// Palette and type taken from the pitch deck (Grease DISPATCH Pitch Deck.pdf): navy, signal red, gold, light grey; Oswald headings, Inter body.
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {navy: '#0b192c', signal: '#cc0000', gold: '#ffcc00', amber: '#f5a623', paper: '#f4f5f6', rule: '#e2e8f0', ink: '#334155'},
      fontFamily: {display: ['Oswald', 'Arial Narrow', 'Impact', 'sans-serif'], sans: ['Inter', 'Helvetica Neue', 'Arial', 'sans-serif']},
    },
  },
  plugins: [],
};
