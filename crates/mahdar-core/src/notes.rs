//! The user's own notes for a meeting: reading them from a file.

use std::path::Path;

use anyhow::{bail, Context, Result};
use docx_rs::{DocumentChild, Paragraph, Table, TableCellContent, TableChild, TableRowChild};

/// Text of a notes file: .docx, or plain text (.txt, .md, anything else readable).
pub fn read_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let text = match ext.as_str() {
        "docx" => docx_text(&bytes)?,
        "doc" | "pdf" | "pages" | "rtf" => bail!("can't read .{ext} notes: save them as .docx or .txt"),
        _ => String::from_utf8_lossy(&bytes).trim_start_matches('\u{feff}').replace("\r\n", "\n"),
    };
    Ok(text.trim().to_string())
}

/// Plain text of a Word document: one line per paragraph, "- " for list items,
/// table cells joined with " | ".
pub fn docx_text(bytes: &[u8]) -> Result<String> {
    let doc = docx_rs::read_docx(bytes).map_err(|e| anyhow::anyhow!("not a readable .docx: {e}"))?;
    let mut lines = Vec::new();
    for child in &doc.document.children {
        match child {
            DocumentChild::Paragraph(p) => lines.push(paragraph(p)),
            DocumentChild::Table(t) => table(t, &mut lines),
            _ => {}
        }
    }
    Ok(lines.join("\n"))
}

fn paragraph(p: &Paragraph) -> String {
    let text = p.raw_text();
    if p.property.numbering_property.is_some() && !text.trim().is_empty() {
        format!("- {text}")
    } else {
        text
    }
}

fn table(t: &Table, lines: &mut Vec<String>) {
    for TableChild::TableRow(row) in &t.rows {
        let cells: Vec<String> = row
            .cells
            .iter()
            .map(|TableRowChild::TableCell(c)| {
                c.children
                    .iter()
                    .filter_map(|x| match x {
                        TableCellContent::Paragraph(p) => Some(p.raw_text()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
                    .trim()
                    .to_string()
            })
            .collect();
        if cells.iter().any(|c| !c.is_empty()) {
            lines.push(cells.join(" | "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_docx_notes() {
        let doc = docx_rs::Docx::new()
            .add_paragraph(Paragraph::new().add_run(docx_rs::Run::new().add_text("Pricing call")))
            .add_paragraph(
                Paragraph::new()
                    .add_run(docx_rs::Run::new().add_text("@8 missed the price"))
                    .numbering(docx_rs::NumberingId::new(1), docx_rs::IndentLevel::new(0)),
            );
        let mut buf = std::io::Cursor::new(Vec::new());
        doc.build().pack(&mut buf).unwrap();
        assert_eq!(docx_text(buf.get_ref()).unwrap(), "Pricing call\n- @8 missed the price");
    }

    #[test]
    fn reads_text_notes() {
        let dir = std::env::temp_dir().join(format!("mahdar-notes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("n.md");
        std::fs::write(&p, "\u{feff}line one\r\nline two\r\n").unwrap();
        assert_eq!(read_file(&p).unwrap(), "line one\nline two");
        assert!(read_file(&dir.join("n.pdf")).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
