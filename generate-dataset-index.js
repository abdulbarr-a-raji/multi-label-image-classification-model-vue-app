const fs = require("fs");
const path = require("path");

const dir = path.join(__dirname, "public/assets/training-images");

const files = fs.readdirSync(dir).filter(f => {
  return !f.toLowerCase().endsWith(".json") && (f != ".DS_Store");
});

const outPath = path.join(dir, "dataset-index.json");
fs.writeFileSync(outPath, JSON.stringify(files, null, 2));

console.log(`✅ Generated dataset-index.json with ${files.length} items`);
