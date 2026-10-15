//! Mahdar Brain: a local index over meetings for questions, people/organisation pages and
//! topic timelines across all of them.
//!
//! - Passages (~30–60 s of speech) with a keyword index (SQLite FTS5 over Arabic-folded text)
//!   and a meaning index (local multilingual embeddings). Nothing leaves the machine to build it.
//! - Tags per meeting from the writer model: entities with what was said about them, decisions
//!   and action items. Entities are merged across meetings.
//! - Questions use hybrid retrieval (keywords + meaning, fused with RRF); only the best excerpts
//!   go to the writer.
//!
//! Documents are generic (id, title, date, timestamped lines), so other apps can feed it too.
//! One file: `<data>/brain.db`.

pub mod ask;
pub mod embed;
pub mod tags;
pub mod text;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

pub use tags::{DocTags, Entity};
pub use text::{Line, Passage};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meetings_indexed (
  meeting_id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL,
  transcript_hash TEXT NOT NULL, indexed_at TEXT NOT NULL, tagged INTEGER NOT NULL DEFAULT 0, tag_error TEXT
);
CREATE TABLE IF NOT EXISTS passages (
  id INTEGER PRIMARY KEY, meeting_id TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL,
  speaker TEXT, text TEXT NOT NULL, embedding BLOB
);
CREATE INDEX IF NOT EXISTS passages_meeting ON passages(meeting_id);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(text, passage_id UNINDEXED, tokenize = 'unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS entities (key TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]', role TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS mentions (
  entity_key TEXT NOT NULL, meeting_id TEXT NOT NULL, summary TEXT NOT NULL, timestamps TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY (entity_key, meeting_id)
);
CREATE TABLE IF NOT EXISTS items (
  meeting_id TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, owner TEXT NOT NULL DEFAULT '', due TEXT NOT NULL DEFAULT '',
  entity_keys TEXT NOT NULL DEFAULT '[]', ts REAL
);
CREATE TABLE IF NOT EXISTS asks (
  id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, citations TEXT NOT NULL, provider TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '', created TEXT NOT NULL, thread INTEGER
);
"#;

/// Bring databases from older versions up to [`SCHEMA`].
fn migrate(c: &Connection) -> Result<()> {
    let has_thread = c.prepare("SELECT thread FROM asks LIMIT 0").is_ok();
    if !has_thread {
        c.execute_batch("ALTER TABLE asks ADD COLUMN thread INTEGER; UPDATE asks SET thread = id;")?;
    }
    Ok(())
}

/// A meeting (or other document) to index.
#[derive(Clone, Debug)]
pub struct Doc {
    pub id: String,
    pub title: String,
    /// RFC 3339
    pub created: String,
    /// Hash of the transcript, to notice edits later ([`text::hash`]).
    pub hash: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct IndexedDoc {
    pub meeting_id: String,
    pub title: String,
    pub created: String,
    pub transcript_hash: String,
    pub indexed_at: String,
    pub tagged: bool,
    pub tag_error: Option<String>,
}

/// A search result: one passage.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub passage_id: i64,
    pub doc_id: String,
    pub title: String,
    pub created: String,
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: Option<String>,
    pub text: String,
    pub score: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// Only meetings where this entity was tagged.
    pub entity: Option<String>,
    /// Only meetings on or after / before these dates ("2026-09-01").
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EntityRow {
    pub key: String,
    pub kind: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub role: String,
    pub meetings: usize,
    pub last: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MentionRow {
    pub meeting_id: String,
    pub title: String,
    pub created: String,
    pub summary: String,
    /// Seconds
    pub timestamps: Vec<f64>,
    pub items: Vec<ItemRow>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ItemRow {
    pub kind: String,
    pub text: String,
    pub owner: String,
    pub due: String,
    pub ts: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EntityPage {
    pub entity: EntityRow,
    /// Oldest first: the timeline.
    pub mentions: Vec<MentionRow>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AskRow {
    pub id: i64,
    pub question: String,
    pub answer: String,
    pub citations: Vec<ask::Citation>,
    pub provider: String,
    pub scope: String,
    pub created: String,
    /// Id of the conversation's first question (follow-ups share it).
    pub thread: i64,
}

pub struct Brain {
    conn: Mutex<Connection>,
}

fn now() -> String {
    chrono::Local::now().to_rfc3339()
}

impl Brain {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path).with_context(|| format!("could not open {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        migrate(&conn)?;
        Ok(Brain { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Brain { conn: Mutex::new(conn) })
    }

    pub fn indexed(&self) -> Result<Vec<IndexedDoc>> {
        let c = self.conn.lock().unwrap();
        let mut st = c.prepare("SELECT meeting_id, title, created, transcript_hash, indexed_at, tagged, tag_error FROM meetings_indexed")?;
        let rows = st.query_map([], |r| {
            Ok(IndexedDoc {
                meeting_id: r.get(0)?,
                title: r.get(1)?,
                created: r.get(2)?,
                transcript_hash: r.get(3)?,
                indexed_at: r.get(4)?,
                tagged: r.get::<_, i64>(5)? != 0,
                tag_error: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Keep titles current (meetings get renamed after indexing).
    pub fn sync_titles(&self, titles: &[(String, String)]) -> Result<()> {
        let c = self.conn.lock().unwrap();
        for (id, title) in titles {
            c.execute("UPDATE meetings_indexed SET title = ?2 WHERE meeting_id = ?1 AND title <> ?2", params![id, title])?;
        }
        Ok(())
    }

    /// Store a document's passages and their embeddings, replacing any earlier index of it.
    /// Its tags are cleared (tag it again with [`Brain::set_tags`]).
    pub fn index(&self, doc: &Doc, passages: &[Passage], embeddings: &[Vec<f32>]) -> Result<()> {
        anyhow::ensure!(passages.len() == embeddings.len(), "passages and embeddings differ in length");
        let mut c = self.conn.lock().unwrap();
        let tx = c.transaction()?;
        remove_rows(&tx, &doc.id)?;
        {
            let mut ins = tx.prepare("INSERT INTO passages (meeting_id, start_s, end_s, speaker, text, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")?;
            let mut fts = tx.prepare("INSERT INTO passages_fts (text, passage_id) VALUES (?1, ?2)")?;
            for (p, e) in passages.iter().zip(embeddings) {
                ins.execute(params![doc.id, p.start_s, p.end_s, p.speaker, p.text, embed::to_blob(e)])?;
                let id = tx.last_insert_rowid();
                fts.execute(params![text::index_text(&p.text), id])?;
            }
        }
        tx.execute(
            "INSERT OR REPLACE INTO meetings_indexed (meeting_id, title, created, transcript_hash, indexed_at, tagged, tag_error) VALUES (?1, ?2, ?3, ?4, ?5, 0, NULL)",
            params![doc.id, doc.title, doc.created, doc.hash, now()],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn remove(&self, doc_id: &str) -> Result<()> {
        let mut c = self.conn.lock().unwrap();
        let tx = c.transaction()?;
        remove_rows(&tx, doc_id)?;
        tx.execute("DELETE FROM meetings_indexed WHERE meeting_id = ?1", [doc_id])?;
        tx.execute("DELETE FROM entities WHERE key NOT IN (SELECT entity_key FROM mentions)", [])?;
        tx.commit()?;
        Ok(())
    }

    pub fn entities_all(&self) -> Result<Vec<Entity>> {
        let c = self.conn.lock().unwrap();
        load_entities(&c)
    }

    /// Record a document's tags: merge its entities into the known ones, replace its mentions,
    /// decisions and actions.
    pub fn set_tags(&self, doc_id: &str, t: &DocTags) -> Result<()> {
        let mut c = self.conn.lock().unwrap();
        let tx = c.transaction()?;
        let mut known = load_entities(&tx)?;
        tx.execute("DELETE FROM mentions WHERE meeting_id = ?1", [doc_id])?;
        tx.execute("DELETE FROM items WHERE meeting_id = ?1", [doc_id])?;
        // name (normalized, incl. aliases) → key, for linking items to entities
        let mut by_name: HashMap<String, String> = HashMap::new();
        let mut mentions: HashMap<String, (String, Vec<f64>)> = HashMap::new();
        let mut touched: HashSet<String> = HashSet::new();
        for e in &t.entities {
            let idx = match tags::resolve(&known, e) {
                Some(i) => {
                    tags::absorb(&mut known[i], e);
                    i
                }
                None => {
                    let kind = tags::kind(&e.kind).to_string();
                    let mut new = Entity { key: tags::key_for(&kind, &e.name), kind, name: e.name.trim().to_string(), aliases: Vec::new(), role: String::new() };
                    // A different name that folds to an existing key: treat as the same.
                    if let Some(i) = known.iter().position(|k| k.key == new.key) {
                        tags::absorb(&mut known[i], e);
                        i
                    } else {
                        tags::absorb(&mut new, e);
                        known.push(new);
                        known.len() - 1
                    }
                }
            };
            let key = known[idx].key.clone();
            touched.insert(key.clone());
            for n in std::iter::once(&e.name).chain(&e.aliases) {
                by_name.insert(tags::norm(n), key.clone());
            }
            let m = mentions.entry(key).or_default();
            if !e.summary.trim().is_empty() {
                if !m.0.is_empty() {
                    m.0.push(' ');
                }
                m.0.push_str(e.summary.trim());
            }
            for ts in e.timestamps.iter().filter_map(|s| text::parse_ts(s)) {
                if !m.1.contains(&ts) {
                    m.1.push(ts);
                }
            }
        }
        for e in known.iter().filter(|e| touched.contains(&e.key)) {
            tx.execute(
                "INSERT INTO entities (key, type, name, aliases, role) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(key) DO UPDATE SET aliases = excluded.aliases, role = excluded.role",
                params![e.key, e.kind, e.name, serde_json::to_string(&e.aliases)?, e.role],
            )?;
        }
        for (key, (summary, mut ts)) in mentions {
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tx.execute(
                "INSERT INTO mentions (entity_key, meeting_id, summary, timestamps) VALUES (?1, ?2, ?3, ?4)",
                params![key, doc_id, summary, serde_json::to_string(&ts)?],
            )?;
        }
        for (kind, list) in [("decision", &t.decisions), ("action", &t.actions)] {
            for it in list.iter().filter(|i| !i.text.trim().is_empty()) {
                let mut keys: Vec<String> = it.entities.iter().filter_map(|n| by_name.get(&tags::norm(n)).cloned()).collect();
                if let Some(k) = by_name.get(&tags::norm(&it.owner)) {
                    keys.push(k.clone());
                }
                keys.dedup();
                tx.execute(
                    "INSERT INTO items (meeting_id, kind, text, owner, due, entity_keys, ts) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![doc_id, kind, it.text.trim(), it.owner.trim(), it.due.trim(), serde_json::to_string(&keys)?, text::parse_ts(&it.timestamp)],
                )?;
            }
        }
        tx.execute("UPDATE meetings_indexed SET tagged = 1, tag_error = NULL WHERE meeting_id = ?1", [doc_id])?;
        tx.execute("DELETE FROM entities WHERE key NOT IN (SELECT entity_key FROM mentions)", [])?;
        tx.commit()?;
        Ok(())
    }

    pub fn set_tag_error(&self, doc_id: &str, err: &str) -> Result<()> {
        let c = self.conn.lock().unwrap();
        c.execute("UPDATE meetings_indexed SET tag_error = ?2 WHERE meeting_id = ?1", params![doc_id, err])?;
        Ok(())
    }

    /// All entities with how many meetings mention them, most mentioned first.
    pub fn entities(&self) -> Result<Vec<EntityRow>> {
        let c = self.conn.lock().unwrap();
        let mut st = c.prepare(
            "SELECT e.key, e.type, e.name, e.aliases, e.role, COUNT(m.meeting_id), COALESCE(MAX(d.created), '')
             FROM entities e JOIN mentions m ON m.entity_key = e.key LEFT JOIN meetings_indexed d ON d.meeting_id = m.meeting_id
             GROUP BY e.key ORDER BY COUNT(m.meeting_id) DESC, e.name COLLATE NOCASE",
        )?;
        let rows = st.query_map([], entity_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn entity_page(&self, key: &str) -> Result<Option<EntityPage>> {
        let c = self.conn.lock().unwrap();
        let entity = c
            .query_row(
                "SELECT e.key, e.type, e.name, e.aliases, e.role, COUNT(m.meeting_id), COALESCE(MAX(d.created), '')
                 FROM entities e JOIN mentions m ON m.entity_key = e.key LEFT JOIN meetings_indexed d ON d.meeting_id = m.meeting_id
                 WHERE e.key = ?1 GROUP BY e.key",
                [key],
                entity_row,
            )
            .optional()?;
        let Some(entity) = entity else { return Ok(None) };
        let mut st = c.prepare(
            "SELECT m.meeting_id, d.title, d.created, m.summary, m.timestamps FROM mentions m
             JOIN meetings_indexed d ON d.meeting_id = m.meeting_id WHERE m.entity_key = ?1 ORDER BY d.created",
        )?;
        let mut mentions: Vec<MentionRow> = st
            .query_map([key], |r| {
                Ok(MentionRow {
                    meeting_id: r.get(0)?,
                    title: r.get(1)?,
                    created: r.get(2)?,
                    summary: r.get(3)?,
                    timestamps: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_default(),
                    items: Vec::new(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let pattern = format!("%{}%", serde_json::to_string(key)?);
        let mut st = c.prepare("SELECT meeting_id, kind, text, owner, due, ts FROM items WHERE entity_keys LIKE ?1 ORDER BY ts")?;
        let items = st.query_map([pattern], |r| {
            Ok((r.get::<_, String>(0)?, ItemRow { kind: r.get(1)?, text: r.get(2)?, owner: r.get(3)?, due: r.get(4)?, ts: r.get(5)? }))
        })?;
        for it in items {
            let (mid, item) = it?;
            if let Some(m) = mentions.iter_mut().find(|m| m.meeting_id == mid) {
                m.items.push(item);
            }
        }
        Ok(Some(EntityPage { entity, mentions }))
    }

    /// Hybrid search: keyword (FTS5 bm25) and meaning (cosine), fused with RRF. `extra` adds
    /// keyword terms (e.g. the filtered entity's names).
    pub fn search(&self, query: &str, query_vec: Option<&[f32]>, filter: &Filter, k: usize) -> Result<Vec<Hit>> {
        let c = self.conn.lock().unwrap();
        // Which meetings are in scope
        let mut sql = String::from("SELECT meeting_id FROM meetings_indexed WHERE 1=1");
        let mut args: Vec<String> = Vec::new();
        if let Some(from) = &filter.from {
            args.push(from.clone());
            sql.push_str(&format!(" AND substr(created, 1, 10) >= ?{}", args.len()));
        }
        if let Some(to) = &filter.to {
            args.push(to.clone());
            sql.push_str(&format!(" AND substr(created, 1, 10) <= ?{}", args.len()));
        }
        let mut extra: Vec<String> = Vec::new();
        if let Some(key) = &filter.entity {
            args.push(key.clone());
            sql.push_str(&format!(" AND meeting_id IN (SELECT meeting_id FROM mentions WHERE entity_key = ?{})", args.len()));
            if let Some((name, aliases)) = c
                .query_row("SELECT name, aliases FROM entities WHERE key = ?1", [key], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .optional()?
            {
                extra.push(name);
                extra.extend(serde_json::from_str::<Vec<String>>(&aliases).unwrap_or_default());
            }
        }
        let scope: HashSet<String> = {
            let mut st = c.prepare(&sql)?;
            let rows = st.query_map(rusqlite::params_from_iter(args.iter()), |r| r.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        if scope.is_empty() {
            return Ok(Vec::new());
        }

        // Keyword ranking
        let mut kw: Vec<i64> = Vec::new();
        if let Some(q) = text::fts_query(query, &extra) {
            let mut st = c.prepare(
                "SELECT f.passage_id, p.meeting_id FROM passages_fts f JOIN passages p ON p.id = f.passage_id
                 WHERE passages_fts MATCH ?1 ORDER BY bm25(passages_fts) LIMIT 200",
            )?;
            let rows = st.query_map([q], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            for row in rows {
                let (id, mid) = row?;
                if scope.contains(&mid) && kw.len() < 50 {
                    kw.push(id);
                }
            }
        }
        // Meaning ranking (brute force: a few thousand passages)
        let mut vec_rank: Vec<i64> = Vec::new();
        if let Some(qv) = query_vec {
            let mut st = c.prepare("SELECT id, meeting_id, embedding FROM passages WHERE embedding IS NOT NULL")?;
            let mut scored: Vec<(i64, f32)> = st
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?)))?
                .filter_map(|r| r.ok())
                .filter(|(_, mid, _)| scope.contains(mid))
                .map(|(id, _, b)| (id, embed::dot(qv, &embed::from_blob(&b))))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            vec_rank = scored.into_iter().take(50).map(|x| x.0).collect();
        }
        let fused = ask::rrf(&[kw, vec_rank], 60.0);
        let mut hits = Vec::new();
        let mut st = c.prepare(
            "SELECT p.id, p.meeting_id, d.title, d.created, p.start_s, p.end_s, p.speaker, p.text
             FROM passages p JOIN meetings_indexed d ON d.meeting_id = p.meeting_id WHERE p.id = ?1",
        )?;
        for (id, score) in fused.into_iter().take(k) {
            let h = st.query_row([id], |r| {
                Ok(Hit {
                    passage_id: r.get(0)?,
                    doc_id: r.get(1)?,
                    title: r.get(2)?,
                    created: r.get(3)?,
                    start_s: r.get(4)?,
                    end_s: r.get(5)?,
                    speaker: r.get(6)?,
                    text: r.get(7)?,
                    score,
                })
            })?;
            hits.push(h);
        }
        Ok(hits)
    }

    /// Save a question and its answer. `thread`: the conversation it follows up, or None to start one.
    pub fn save_ask(&self, question: &str, answer: &str, citations: &[ask::Citation], provider: &str, scope: &str, thread: Option<i64>) -> Result<i64> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "INSERT INTO asks (question, answer, citations, provider, scope, created, thread) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![question, answer, serde_json::to_string(citations)?, provider, scope, now(), thread],
        )?;
        let id = c.last_insert_rowid();
        if thread.is_none() {
            c.execute("UPDATE asks SET thread = id WHERE id = ?1", [id])?;
        }
        // Keep the last 200 questions.
        c.execute("DELETE FROM asks WHERE id NOT IN (SELECT id FROM asks ORDER BY id DESC LIMIT 200)", [])?;
        Ok(id)
    }

    const ASK_COLS: &'static str = "id, question, answer, citations, provider, scope, created, COALESCE(thread, id)";

    fn ask_rows(c: &Connection, sql: &str, arg: i64) -> Result<Vec<AskRow>> {
        let mut st = c.prepare(sql)?;
        let rows = st.query_map([arg], |r| {
            Ok(AskRow {
                id: r.get(0)?,
                question: r.get(1)?,
                answer: r.get(2)?,
                citations: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
                provider: r.get(4)?,
                scope: r.get(5)?,
                created: r.get(6)?,
                thread: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The latest questions, newest first.
    pub fn asks(&self, limit: usize) -> Result<Vec<AskRow>> {
        let c = self.conn.lock().unwrap();
        Self::ask_rows(&c, &format!("SELECT {} FROM asks ORDER BY id DESC LIMIT ?1", Self::ASK_COLS), limit as i64)
    }

    /// One conversation, oldest first.
    pub fn thread(&self, thread: i64) -> Result<Vec<AskRow>> {
        let c = self.conn.lock().unwrap();
        Self::ask_rows(&c, &format!("SELECT {} FROM asks WHERE COALESCE(thread, id) = ?1 ORDER BY id", Self::ASK_COLS), thread)
    }

    /// Delete a whole conversation.
    pub fn delete_thread(&self, thread: i64) -> Result<()> {
        let c = self.conn.lock().unwrap();
        c.execute("DELETE FROM asks WHERE COALESCE(thread, id) = ?1", [thread])?;
        Ok(())
    }
}

fn remove_rows(c: &Connection, doc_id: &str) -> Result<()> {
    c.execute("DELETE FROM passages_fts WHERE passage_id IN (SELECT id FROM passages WHERE meeting_id = ?1)", [doc_id])?;
    c.execute("DELETE FROM passages WHERE meeting_id = ?1", [doc_id])?;
    c.execute("DELETE FROM mentions WHERE meeting_id = ?1", [doc_id])?;
    c.execute("DELETE FROM items WHERE meeting_id = ?1", [doc_id])?;
    Ok(())
}

fn load_entities(c: &Connection) -> Result<Vec<Entity>> {
    let mut st = c.prepare("SELECT key, type, name, aliases, role FROM entities")?;
    let rows = st.query_map([], |r| {
        Ok(Entity {
            key: r.get(0)?,
            kind: r.get(1)?,
            name: r.get(2)?,
            aliases: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
            role: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn entity_row(r: &rusqlite::Row) -> rusqlite::Result<EntityRow> {
    Ok(EntityRow {
        key: r.get(0)?,
        kind: r.get(1)?,
        name: r.get(2)?,
        aliases: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
        role: r.get(4)?,
        meetings: r.get::<_, i64>(5)? as usize,
        last: r.get(6)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tags::{TagEntity, TagItem};

    fn doc(id: &str, date: &str) -> Doc {
        Doc { id: id.into(), title: format!("Meeting {id}"), created: format!("{date}T10:00:00+03:00"), hash: "h".into() }
    }
    fn p(t: f64, s: &str) -> Passage {
        Passage { start_s: t, end_s: t + 30.0, speaker: Some("Speaker 1".into()), text: s.into() }
    }

    #[test]
    fn arabic_keyword_search_and_reindex() {
        let b = Brain::open_in_memory().unwrap();
        let ps = vec![p(0.0, "Speaker 1: الأسعار الجديدة للمشروع"), p(30.0, "Speaker 2: the report goes out Thursday"), p(60.0, "Speaker 1: والفريق المركزي وافق")];
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
        b.index(&doc("a", "2026-09-01"), &ps, &embs).unwrap();
        // Diacritics, hamza and the article don't block a match.
        let hits = b.search("الاسعار", None, &Filter::default(), 5).unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].start_s, 0.0);
        let hits = b.search("فريق مركزي", None, &Filter::default(), 5).unwrap();
        assert_eq!(hits[0].start_s, 60.0);
        // Meaning ranking alone
        let hits = b.search("zzz", Some(&[0.0, 1.0]), &Filter::default(), 1).unwrap();
        assert_eq!(hits[0].start_s, 30.0);
        // Re-indexing replaces, not duplicates
        b.index(&doc("a", "2026-09-01"), &ps[..1], &embs[..1]).unwrap();
        assert_eq!(b.search("thursday", None, &Filter::default(), 5).unwrap().len(), 0);
        assert_eq!(b.indexed().unwrap().len(), 1);
        // Date filter
        let f = Filter { from: Some("2026-10-01".into()), ..Default::default() };
        assert!(b.search("اسعار", None, &f, 5).unwrap().is_empty());
    }

    #[test]
    fn tags_merge_across_meetings_into_a_timeline() {
        let b = Brain::open_in_memory().unwrap();
        b.index(&doc("m1", "2026-09-01"), &[p(0.0, "x")], &[vec![1.0]]).unwrap();
        b.index(&doc("m2", "2026-09-15"), &[p(0.0, "y")], &[vec![1.0]]).unwrap();
        let t1 = DocTags {
            entities: vec![
                TagEntity { name: "Person One".into(), kind: "person".into(), aliases: vec!["الشخص".into()], summary: "Asked for a budget of 5.".into(), timestamps: vec!["01:10".into()], ..Default::default() },
                TagEntity { name: "Org One".into(), kind: "company".into(), summary: "Requested a budget.".into(), ..Default::default() },
            ],
            actions: vec![TagItem { text: "Send report".into(), owner: "Person One".into(), due: "Thursday".into(), entities: vec!["Org One".into()], timestamp: "02:00".into() }],
            ..Default::default()
        };
        b.set_tags("m1", &t1).unwrap();
        let t2 = DocTags {
            entities: vec![TagEntity { name: "Person".into(), kind: "person".into(), aliases: vec!["الشخص".into()], role: "Team lead".into(), summary: "Budget reduced to 3.".into(), ..Default::default() }],
            decisions: vec![TagItem { text: "Budget set at 3".into(), entities: vec!["الشخص".into()], ..Default::default() }],
            ..Default::default()
        };
        b.set_tags("m2", &t2).unwrap();
        let ents = b.entities().unwrap();
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].name, "Person One");
        assert_eq!(ents[0].meetings, 2);
        assert_eq!(ents[0].role, "Team lead");
        let page = b.entity_page("person:person one").unwrap().unwrap();
        assert_eq!(page.mentions.iter().map(|m| m.meeting_id.as_str()).collect::<Vec<_>>(), vec!["m1", "m2"]);
        assert_eq!(page.mentions[0].timestamps, vec![70.0]);
        assert_eq!(page.mentions[0].items[0].text, "Send report");
        assert_eq!(page.mentions[1].items[0].kind, "decision");
        // Entity filter keeps search inside that entity's meetings.
        let f = Filter { entity: Some("org:org one".into()), ..Default::default() };
        let hits = b.search("y x", Some(&[1.0]), &f, 5).unwrap();
        assert!(hits.iter().all(|h| h.doc_id == "m1"));
        // Re-tagging a meeting replaces its mentions; removing it drops orphaned entities.
        b.remove("m1").unwrap();
        assert_eq!(b.entities().unwrap().len(), 1);
    }

    #[test]
    fn asks_are_kept() {
        let b = Brain::open_in_memory().unwrap();
        let c = vec![ask::Citation { n: 1, meeting_id: "m".into(), title: "T".into(), start_s: 5.0, speaker: None }];
        let first = b.save_ask("q?", "a [#1]", &c, "Local", "", None).unwrap();
        let a = b.asks(10).unwrap();
        assert_eq!(a[0].citations, c);
        assert_eq!(a[0].thread, first);
        // A follow-up joins the conversation; a new question starts its own.
        b.save_ask("and then?", "b", &[], "Local", "", Some(first)).unwrap();
        let other = b.save_ask("other?", "c", &[], "Local", "", None).unwrap();
        let t = b.thread(first).unwrap();
        assert_eq!(t.iter().map(|r| r.question.as_str()).collect::<Vec<_>>(), vec!["q?", "and then?"]);
        assert_eq!(b.thread(other).unwrap().len(), 1);
        b.delete_thread(first).unwrap();
        assert_eq!(b.asks(10).unwrap().len(), 1);
    }

    #[test]
    fn old_asks_table_gets_threads() {
        let dir = std::env::temp_dir().join(format!("mahdar-brain-mig-{}", std::process::id()));
        let path = dir.join("brain.db");
        std::fs::create_dir_all(&dir).unwrap();
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE asks (id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, citations TEXT NOT NULL, \
                 provider TEXT NOT NULL, scope TEXT NOT NULL DEFAULT '', created TEXT NOT NULL); \
                 INSERT INTO asks (question, answer, citations, provider, created) VALUES ('q', 'a', '[]', 'p', 'now');",
            )
            .unwrap();
        }
        let b = Brain::open(&path).unwrap();
        let a = b.asks(5).unwrap();
        assert_eq!(a[0].thread, a[0].id);
        drop(b);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
