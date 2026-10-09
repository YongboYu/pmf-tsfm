// Derive the public (HF Space) variant of the deck from slides.md.
//
// The talk deck keeps speaker notes and a trailing block of backup slides; the public build drops
// both. slides.md stays the single source of truth — the public variant is generated at build time
// (see make-public-deck.mjs) and never committed.
//
// - Backup slide: any slide whose frontmatter `class` contains `backup`.
// - Speaker notes: Slidev reads the last HTML comment of a slide as its note. We strip every HTML
//   comment, which also removes layout comments that never render anyway.

const SEP = '---'
const YAML_KEY = /^[A-Za-z_][\w-]*:/

// Split a Slidev markdown file into slides: [{ frontmatter: string|null, body: string }].
// The first slide's frontmatter is the deck headmatter.
export function splitSlides(md) {
  const lines = md.split('\n')
  const slides = []
  let i = 0
  let current = { frontmatter: null, body: [] }

  const readFrontmatter = () => {
    const start = i + 1
    const end = lines.indexOf(SEP, start)
    if (end === -1 || !YAML_KEY.test(lines[start] ?? '')) return null
    i = end + 1
    return lines.slice(start, end).join('\n')
  }

  if (lines[0] === SEP) current.frontmatter = readFrontmatter()

  while (i < lines.length) {
    if (lines[i] === SEP) {
      slides.push(current)
      current = { frontmatter: null, body: [] }
      const fm = readFrontmatter()
      if (fm === null) i += 1 // bare separator, no frontmatter
      else current.frontmatter = fm
      continue
    }
    current.body.push(lines[i])
    i += 1
  }
  slides.push(current)
  return slides.map(s => ({ frontmatter: s.frontmatter, body: s.body.join('\n') }))
}

export function joinSlides(slides) {
  return slides
    .map(({ frontmatter, body }) =>
      frontmatter === null ? `${SEP}\n${body}` : `${SEP}\n${frontmatter}\n${SEP}\n${body}`,
    )
    .join('\n')
}

export function isBackup({ frontmatter }) {
  const m = frontmatter?.match(/^class:\s*(.*)$/m)
  return Boolean(m && /\bbackup\b/.test(m[1]))
}

// A comment on its own line goes with its line. Removing only the comment would leave an indented
// whitespace-only line — a blank line to markdown — which ends the enclosing HTML block, so the
// rest of the block (indented 4+ spaces) renders as a code block.
export function stripComments(body) {
  return body
    .replace(/^[ \t]*<!--[\s\S]*?-->[ \t]*(\n|$)/gm, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\n{3,}/g, '\n\n')
}

export function toPublicDeck(md) {
  const slides = splitSlides(md)
    .filter(s => !isBackup(s))
    .map(s => ({ ...s, body: stripComments(s.body) }))
  return joinSlides(slides)
}
