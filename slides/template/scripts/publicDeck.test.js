import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { isBackup, joinSlides, splitSlides, stripComments, toPublicDeck } from './publicDeck.js'

const deck = `---
theme: default
layout: cover
---

# Title

<!-- cover note -->

---
layout: center
---

# Main

<div>
  <!-- inline comment -->
  body
</div>

<!--
multi-line
speaker note
-->

---
layout: end
---

# Thanks

---
layout: assertion-evidence
class: backup
hideInToc: true
---

# Backup one

<!-- backup note -->
`

describe('splitSlides', () => {
  it('splits headmatter and per-slide frontmatter', () => {
    const slides = splitSlides(deck)
    expect(slides).toHaveLength(4)
    expect(slides[0].frontmatter).toBe('theme: default\nlayout: cover')
    expect(slides[3].frontmatter).toContain('class: backup')
  })

  it('round-trips through joinSlides', () => {
    expect(joinSlides(splitSlides(deck))).toBe(deck)
  })

  it('handles a bare separator without frontmatter', () => {
    const slides = splitSlides('---\ntheme: default\n---\n\n# A\n\n---\n\n# B\n')
    expect(slides).toHaveLength(2)
    expect(slides[1].frontmatter).toBeNull()
    expect(slides[1].body).toContain('# B')
  })
})

describe('stripComments', () => {
  it('removes an own-line comment with its line, keeping the HTML block unbroken', () => {
    const body = '<div>\n  <div>\n    <!-- note -->\n    <div>x</div>\n  </div>\n</div>'
    expect(stripComments(body)).toBe('<div>\n  <div>\n    <div>x</div>\n  </div>\n</div>')
  })

  it('removes an inline comment but keeps the surrounding text', () => {
    expect(stripComments('a <!-- c --> b\n')).toBe('a  b\n')
  })
})

describe('toPublicDeck', () => {
  const out = toPublicDeck(deck)

  it('drops backup slides', () => {
    expect(out).not.toContain('Backup one')
    expect(splitSlides(out).some(isBackup)).toBe(false)
    expect(splitSlides(out)).toHaveLength(3)
  })

  it('strips every HTML comment (speaker notes included)', () => {
    expect(out).not.toContain('<!--')
    expect(out).toContain('body')
    expect(out).toContain('# Thanks')
  })

  it('keeps headmatter intact', () => {
    expect(out.startsWith('---\ntheme: default\nlayout: cover\n---\n')).toBe(true)
  })
})

describe('real deck', () => {
  const md = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'slides.md'), 'utf8')
  const out = toPublicDeck(md)

  it('has no notes or backups, and keeps every main slide', () => {
    const main = splitSlides(md).filter(s => !isBackup(s))
    expect(out).not.toContain('<!--')
    expect(splitSlides(out)).toHaveLength(main.length)
    expect(splitSlides(md).length).toBeGreaterThan(main.length)
  })

  it('leaves no whitespace-only lines behind (they would break HTML blocks)', () => {
    expect(out).not.toMatch(/^[ \t]+$/m)
  })
})
