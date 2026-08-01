# oss-bugbot web frontend

React + Vite + Tailwind + Framer Motion. See the repo root [README](../../README.md#web-ui)
for the full picture (what this is, the bring-your-own-key security model, deploy steps).

```bash
npm install
npm run dev      # local dev, expects the backend on VITE_BACKEND_URL (default localhost:8000)
npm run build     # type-check + production build
npm run test      # vitest - the lib/ws.ts message reducer, the one non-trivial piece of logic here
```
