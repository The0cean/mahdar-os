// Set Mahdar's version everywhere it is stored: package.json (+ lock), Cargo.toml (workspace),
// src-tauri/tauri.conf.json and the workspace crates in Cargo.lock.
//
//   node tools/bump.mjs 0.2.1
//
// Then commit, tag and push (see README › Releasing).
import { readFileSync, writeFileSync } from "node:fs";

const v = process.argv[2];
if (!/^\d+\.\d+\.\d+$/.test(v ?? "")) {
  console.error("usage: node tools/bump.mjs <major.minor.patch>");
  process.exit(1);
}

const edit = (file, re, to) => {
  const s = readFileSync(file, "utf8");
  if (!re.test(s)) throw new Error(`version not found in ${file}`);
  writeFileSync(file, s.replace(re, to));
};

edit("package.json", /("version":\s*")[^"]+(")/, `$1${v}$2`);
const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
lock.version = lock.packages[""].version = v;
writeFileSync("package-lock.json", JSON.stringify(lock, null, 2) + "\n");
edit("src-tauri/tauri.conf.json", /("version":\s*")[^"]+(")/, `$1${v}$2`);
edit("Cargo.toml", /(\[workspace\.package\]\s*\nversion\s*=\s*")[^"]+(")/, `$1${v}$2`);
for (const name of ["mahdar", "mahdar-core", "mahdar-brain"]) {
  edit("Cargo.lock", new RegExp(`(name = "${name}"\\r?\\nversion = ")[^"]+(")`), `$1${v}$2`);
}
console.log(`Mahdar ${v}. Next: git commit -am "v${v}" && git tag v${v} && git push && git push origin v${v}`);
