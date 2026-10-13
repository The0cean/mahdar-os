//! Word (.docx) export of Markdown outputs and transcripts.

use anyhow::Result;
use docx_rs::{
    AlignmentType, BorderType, Docx, LineSpacing, Paragraph, Run, RunFonts, Shading, SpecialIndentType, Table,
    TableBorder, TableBorderPosition, TableBorders, TableCell, TableRow, WidthType,
};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

const FONT: &str = "Arial";
const BODY: usize = 20; // half-points: 10 pt

fn run(text: &str, bold: bool, italic: bool, size: usize) -> Run {
    let mut r = Run::new()
        .add_text(text)
        .size(size)
        .fonts(RunFonts::new().ascii(FONT).hi_ansi(FONT).east_asia(FONT).cs(FONT));
    if bold {
        r = r.bold();
    }
    if italic {
        r = r.italic();
    }
    r
}

fn is_arabic(c: char) -> bool {
    ('\u{0600}'..='\u{06FF}').contains(&c) || ('\u{0750}'..='\u{077F}').contains(&c)
}

/// Letter counts (arabic, latin), to decide a paragraph's alignment.
fn script_counts(s: &str) -> (usize, usize) {
    s.chars().fold((0, 0), |(a, l), c| {
        if is_arabic(c) {
            (a + 1, l)
        } else if c.is_ascii_alphabetic() {
            (a, l + 1)
        } else {
            (a, l)
        }
    })
}

/// Paragraph under construction. Mostly-Arabic paragraphs are right-aligned.
#[derive(Default)]
struct Para {
    runs: Vec<Run>,
    arabic: usize,
    latin: usize,
    indent: Option<i32>,
}

impl Para {
    fn push(&mut self, text: &str, bold: bool, italic: bool, size: usize) {
        let (a, l) = script_counts(text);
        self.arabic += a;
        self.latin += l;
        self.runs.push(run(text, bold, italic, size));
    }
    fn build(self) -> Paragraph {
        let mut p = Paragraph::new().line_spacing(LineSpacing::new().after(80));
        if self.arabic > self.latin {
            p = p.align(AlignmentType::Right);
        }
        if let Some(ind) = self.indent {
            p = p.indent(Some(ind), Some(SpecialIndentType::Hanging(240)), None, None);
        }
        for r in self.runs {
            p = p.add_run(r);
        }
        p
    }
}

fn header(title: &str, subtitle: &str) -> Docx {
    Docx::new()
        .add_paragraph(Paragraph::new().add_run(run(title, true, false, 32)))
        .add_paragraph(Paragraph::new().add_run(run(subtitle, false, false, 18).color("666666")))
        .add_paragraph(Paragraph::new())
}

fn pack(doc: Docx) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    doc.build().pack(&mut buf)?;
    Ok(buf.into_inner())
}

/// Render Markdown (headings, paragraphs, lists, bold/italic, task lists, tables) to a .docx.
pub fn markdown_to_docx(title: &str, subtitle: &str, markdown: &str) -> Result<Vec<u8>> {
    let mut doc = header(title, subtitle);
    let parser = Parser::new_ext(markdown, Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS);

    let mut para: Option<Para> = None;
    let (mut bold, mut italic, mut size) = (false, false, BODY);
    let mut lists: Vec<Option<u64>> = Vec::new(); // per nesting level: next number, or None for bullets
    let (mut rows, mut cells): (Vec<TableRow>, Vec<TableCell>) = (Vec::new(), Vec::new());
    let (mut in_table, mut in_head) = (false, false);

    fn flush(para: &mut Option<Para>, doc: Docx) -> Docx {
        match para.take() {
            Some(p) if !p.runs.is_empty() => doc.add_paragraph(p.build()),
            _ => doc,
        }
    }

    for ev in parser {
        match ev {
            Event::Start(Tag::Heading { level, .. }) => {
                doc = flush(&mut para, doc);
                size = match level {
                    HeadingLevel::H1 => 28,
                    HeadingLevel::H2 => 24,
                    _ => 22,
                };
                bold = true;
                para = Some(Para::default());
            }
            Event::End(TagEnd::Heading(_)) => {
                doc = flush(&mut para, doc);
                size = BODY;
                bold = false;
            }
            Event::Start(Tag::Paragraph) if !in_table && lists.is_empty() => {
                doc = flush(&mut para, doc);
                para = Some(Para::default());
            }
            Event::End(TagEnd::Paragraph) if !in_table && lists.is_empty() => doc = flush(&mut para, doc),
            Event::Start(Tag::List(start)) => {
                doc = flush(&mut para, doc);
                lists.push(start);
            }
            Event::End(TagEnd::List(_)) => {
                doc = flush(&mut para, doc);
                lists.pop();
            }
            Event::Start(Tag::Item) => {
                doc = flush(&mut para, doc);
                let marker = match lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => "• ".to_string(),
                };
                let mut p = Para { indent: Some(360 * lists.len() as i32), ..Default::default() };
                p.push(&marker, false, false, size);
                para = Some(p);
            }
            Event::End(TagEnd::Item) => doc = flush(&mut para, doc),
            Event::TaskListMarker(done) => {
                if let Some(p) = para.as_mut() {
                    p.runs.pop(); // swap the bullet for a checkbox
                    p.push(if done { "☑ " } else { "☐ " }, false, false, size);
                }
            }
            Event::Start(Tag::Table(_)) => {
                doc = flush(&mut para, doc);
                in_table = true;
                rows.clear();
            }
            Event::End(TagEnd::Table) => {
                in_table = false;
                doc = doc.add_table(Table::new(std::mem::take(&mut rows)).set_borders(borders()));
                doc = doc.add_paragraph(Paragraph::new());
            }
            Event::Start(Tag::TableHead) => {
                in_head = true;
                cells.clear();
            }
            Event::End(TagEnd::TableHead) => {
                in_head = false;
                rows.push(TableRow::new(std::mem::take(&mut cells)));
            }
            Event::Start(Tag::TableRow) => cells.clear(),
            Event::End(TagEnd::TableRow) => rows.push(TableRow::new(std::mem::take(&mut cells))),
            Event::Start(Tag::TableCell) => para = Some(Para::default()),
            Event::End(TagEnd::TableCell) => {
                let mut cell = TableCell::new().add_paragraph(para.take().unwrap_or_default().build());
                if in_head {
                    cell = cell.shading(Shading::new().fill("F2F2F2"));
                }
                cells.push(cell);
            }
            Event::Start(Tag::Strong) => bold = true,
            Event::End(TagEnd::Strong) => bold = false,
            Event::Start(Tag::Emphasis) => italic = true,
            Event::End(TagEnd::Emphasis) => italic = false,
            Event::Text(t) | Event::Code(t) => para.get_or_insert_with(Para::default).push(&t, bold || in_head, italic, size),
            Event::SoftBreak | Event::HardBreak => para.get_or_insert_with(Para::default).push(" ", false, false, size),
            Event::Rule => doc = flush(&mut para, doc).add_paragraph(Paragraph::new()),
            _ => {}
        }
    }
    pack(flush(&mut para, doc))
}

fn borders() -> TableBorders {
    let border = |pos| TableBorder::new(pos).border_type(BorderType::Single).size(4).color("BFBFBF");
    TableBorders::new()
        .set(border(TableBorderPosition::Top))
        .set(border(TableBorderPosition::Bottom))
        .set(border(TableBorderPosition::Left))
        .set(border(TableBorderPosition::Right))
        .set(border(TableBorderPosition::InsideH))
        .set(border(TableBorderPosition::InsideV))
}

/// Transcript lines: (timestamp, speaker name or "", text), as a Time | Speaker | Text table
/// (a table keeps timestamps in place when the text is right-to-left).
pub fn transcript_to_docx(title: &str, subtitle: &str, lines: &[(String, String, String)]) -> Result<Vec<u8>> {
    let with_speakers = lines.iter().any(|(_, s, _)| !s.is_empty());
    let widths: Vec<usize> = if with_speakers { vec![1000, 1800, 6400] } else { vec![1000, 8200] };
    let cell = |text: &str, bold: bool, w: usize, shade: bool| {
        let mut p = Para::default();
        p.push(text, bold, false, if w == 1000 { 16 } else { BODY });
        let mut c = TableCell::new().add_paragraph(p.build()).width(w, WidthType::Dxa);
        if shade {
            c = c.shading(Shading::new().fill("F2F2F2"));
        }
        c
    };
    let head: &[&str] = if with_speakers { &["Time", "Speaker", "Text"] } else { &["Time", "Text"] };
    let mut rows = vec![TableRow::new(head.iter().zip(&widths).map(|(h, w)| cell(h, true, *w, true)).collect())];
    for (ts, speaker, text) in lines {
        let mut cells = vec![cell(ts, false, widths[0], false)];
        if with_speakers {
            cells.push(cell(speaker, true, widths[1], false));
        }
        cells.push(cell(text, false, *widths.last().unwrap(), false));
        rows.push(TableRow::new(cells));
    }
    pack(header(title, subtitle).add_table(Table::new(rows).set_grid(widths).set_borders(borders())))
}
