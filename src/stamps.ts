// Time stamps of notes typed while recording ("[12:34] text"), kept apart from the text so stealth
// can hide them: one stamp (or none) per line, and the text without them.

const STAMPED = /^(\[\d{1,2}:\d{2}(?::\d{2})?\]) (.*)$/;

export type Stamps = (string | null)[];

/** Notes → the text without its leading line stamps, and each line's stamp. */
export function splitStamps(notes: string): { text: string; stamps: Stamps } {
  const found = notes.split("\n").map((l) => STAMPED.exec(l));
  return {
    text: notes.split("\n").map((l, i) => found[i]?.[2] ?? l).join("\n"),
    stamps: found.map((m) => m?.[1] ?? null),
  };
}

/** The other way: put each line's stamp back in front of it. */
export function joinStamps(text: string, stamps: Stamps): string {
  return text.split("\n").map((l, i) => (stamps[i] ? `${stamps[i]} ${l}` : l)).join("\n");
}

/**
 * The stamps for `next` after the text changed from `prev`: lines that didn't change keep theirs
 * (also when lines were added or removed above them); in the changed part, edited lines keep
 * their stamp in order and new lines have none. A line emptied loses its stamp, so typing into
 * it again stamps it afresh.
 */
export function realignStamps(prev: string[], next: string[], stamps: Stamps): Stamps {
  let p = 0;
  while (p < prev.length && p < next.length && prev[p] === next[p]) p++;
  let s = 0;
  while (s < prev.length - p && s < next.length - p && prev[prev.length - 1 - s] === next[next.length - 1 - s]) s++;
  const changed = prev.length - s - p;
  const mid = next.slice(p, next.length - s).map((l, i) => (l.trim() && i < changed ? stamps[p + i] ?? null : null));
  return [...stamps.slice(0, p), ...mid, ...stamps.slice(prev.length - s)];
}
