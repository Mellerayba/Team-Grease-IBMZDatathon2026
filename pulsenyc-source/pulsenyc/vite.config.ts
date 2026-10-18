import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// base './' lets the built site work from any sub-path, such as https://<user>.github.io/<repo>/
export default defineConfig({plugins:[react()],base:'./'});
