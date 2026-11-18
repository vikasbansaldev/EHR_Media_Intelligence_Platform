// Copy the self-hosted Public Sans variable font into assets/ so the UI works offline.
import { copyFileSync, mkdirSync } from "node:fs";

const src = "node_modules/@fontsource-variable/public-sans/files/public-sans-latin-wght-normal.woff2";
mkdirSync("assets/fonts", { recursive: true });
copyFileSync(src, "assets/fonts/public-sans-latin-wght-normal.woff2");
console.log("fonts copied");
