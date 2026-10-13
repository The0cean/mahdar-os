//! Tags: the people, organisations, initiatives and topics of a meeting, with what was said
//! about each, plus decisions and action items. Produced by the writer model as JSON, then
//! merged with the entities already known from other meetings.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

pub const TYPES: &[&str] = &["person", "org", "initiative", "topic"];

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DocTags {
    #[serde(default)]
    pub entities: Vec<TagEntity>,
    #[serde(default)]
    pub decisions: Vec<TagItem>,
    #[serde(default)]
    pub actions: Vec<TagItem>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TagEntity {
    pub name: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub timestamps: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TagItem {
    pub text: String,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub due: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub timestamp: String,
}

/// An entity as stored: `key` is stable ("kind:normalised name").
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Entity {
    pub key: String,
    pub kind: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub role: String,
}

/// The four kinds; anything else the model invents becomes "topic".
pub fn kind(s: &str) -> &'static str {
    match s.trim().to_lowercase().as_str() {
        "person" | "people" | "participant" => "person",
        "org" | "organization" | "organisation" | "company" | "bank" | "client" | "institution" | "regulator" => "org",
        "initiative" | "project" | "program" | "programme" | "product" => "initiative",
        _ => "topic",
    }
}

const HONORIFICS: &[&str] = &["mr", "mrs", "ms", "dr", "eng", "prof", "sheikh", "استاذ", "دكتور", "مهندس", "شيخ", "ا", "د", "م"];

/// Comparable form of a name: folded, punctuation removed, honorifics dropped.
pub fn norm(name: &str) -> String {
    let folded = crate::text::fold(name);
    let words: Vec<&str> = folded.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let start = words.iter().take_while(|w| HONORIFICS.contains(w) || w.strip_prefix("ال").map(|r| HONORIFICS.contains(&r)).unwrap_or(false)).count();
    let rest = if start < words.len() { &words[start..] } else { &words[..] };
    rest.join(" ")
}

pub fn key_for(kind: &str, name: &str) -> String {
    format!("{kind}:{}", norm(name))
}

/// The existing entity this incoming one is the same as: same kind, and its name or any
/// alias equals the existing name or any of its aliases (after [`norm`]).
pub fn resolve(existing: &[Entity], e: &TagEntity) -> Option<usize> {
    let k = kind(&e.kind);
    let mine: Vec<String> = std::iter::once(&e.name).chain(&e.aliases).map(|s| norm(s)).filter(|s| !s.is_empty()).collect();
    existing.iter().position(|x| {
        x.kind == k && std::iter::once(&x.name).chain(&x.aliases).map(|s| norm(s)).any(|n| mine.contains(&n))
    })
}

/// Merge `e` into `into` (aliases, role): the canonical name stays.
pub fn absorb(into: &mut Entity, e: &TagEntity) {
    for a in std::iter::once(&e.name).chain(&e.aliases) {
        let a = a.trim();
        if a.is_empty() || norm(a) == norm(&into.name) || into.aliases.iter().any(|x| norm(x) == norm(a)) {
            continue;
        }
        into.aliases.push(a.to_string());
    }
    if !e.role.trim().is_empty() {
        into.role = e.role.trim().to_string();
    }
}

/// The model's reply as tags: tolerates ```json fences and text around the object.
pub fn parse(reply: &str) -> Result<DocTags> {
    let start = reply.find('{').ok_or_else(|| anyhow!("the tagger didn't return JSON"))?;
    let end = reply.rfind('}').ok_or_else(|| anyhow!("the tagger's JSON is cut off"))?;
    let mut t: DocTags = serde_json::from_str(&reply[start..=end]).map_err(|e| anyhow!("the tagger's JSON is invalid: {e}"))?;
    t.entities.retain(|e| !e.name.trim().is_empty());
    Ok(t)
}

/// The tagging request for one meeting. `known` are entities from other meetings, so the
/// model reuses their names.
pub fn prompt(title: &str, date: &str, transcript: &str, glossary: &str, known: &[Entity]) -> String {
    let mut known_list = String::new();
    for e in known.iter().take(300) {
        known_list.push_str(&format!("- {} ({})", e.name, e.kind));
        if !e.aliases.is_empty() {
            known_list.push_str(&format!(" also: {}", e.aliases.join(", ")));
        }
        known_list.push('\n');
    }
    let glossary = glossary.trim();
    format!(
        r#"Index this meeting for a searchable knowledge base. Return ONLY one JSON object, no Markdown, no comments:

{{"entities": [{{"name": "...", "type": "person|org|initiative|topic", "aliases": ["..."], "role": "...", "summary": "...", "timestamps": ["mm:ss"]}}],
 "decisions": [{{"text": "...", "entities": ["entity name"], "timestamp": "mm:ss"}}],
 "actions": [{{"text": "...", "owner": "...", "due": "...", "entities": ["entity name"], "timestamp": "mm:ss"}}]}}

Rules:
- entities: people who speak or are discussed (person); companies, regulators, departments, clients (org); named projects, initiatives, products, programmes (initiative); the main subjects discussed (topic, 3-8 of them, specific rather than generic).
- name: one canonical name in English letters (transliterate Arabic names). If the entity is in the known list below, use exactly that name. Put other spellings seen in the transcript, including Arabic script, in aliases.
- role: the person's role or organisation when stated, else "".
- summary: 1-2 sentences on what was said by or about this entity in THIS meeting. Keep figures, limits, dates and names exact. Never invent.
- timestamps: up to 5 [mm:ss] times from the transcript where it comes up.
- decisions and actions: only ones actually made; owner and due exactly as stated, "" if not stated; entities = names from your entities list.
- Write summaries and items in English.

Meeting: {title} ({date})
{glossary_block}{known_block}
<transcript>
{transcript}
</transcript>"#,
        glossary_block = if glossary.is_empty() { String::new() } else { format!("\nSpellings to use (glossary):\n{glossary}\n") },
        known_block = if known_list.is_empty() { String::new() } else { format!("\nKnown entities from other meetings:\n{known_list}") },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(key: &str, kind: &str, name: &str, aliases: &[&str]) -> Entity {
        Entity { key: key.into(), kind: kind.into(), name: name.into(), aliases: aliases.iter().map(|s| s.to_string()).collect(), role: String::new() }
    }

    #[test]
    fn merges_by_name_or_alias() {
        let known = vec![ent("person:person one", "person", "Person One", &["الشخص"]), ent("org:org two", "org", "Org Two", &[])];
        let same_alias = TagEntity { name: "Person".into(), kind: "person".into(), aliases: vec!["الشخص".into()], ..Default::default() };
        assert_eq!(resolve(&known, &same_alias), Some(0));
        let honorific = TagEntity { name: "Mr. Person One".into(), kind: "Person".into(), ..Default::default() };
        assert_eq!(resolve(&known, &honorific), Some(0));
        let org = TagEntity { name: "org  TWO".into(), kind: "company".into(), ..Default::default() };
        assert_eq!(resolve(&known, &org), Some(1));
        let other = TagEntity { name: "Person".into(), kind: "person".into(), ..Default::default() };
        assert_eq!(resolve(&known, &other), None, "a bare first name alone is not enough");
        let wrong_kind = TagEntity { name: "Person One".into(), kind: "topic".into(), ..Default::default() };
        assert_eq!(resolve(&known, &wrong_kind), None);

        let mut e = known[0].clone();
        absorb(&mut e, &TagEntity { name: "Person".into(), aliases: vec!["الشخص".into(), "Mr Person".into()], role: "Team lead".into(), ..Default::default() });
        assert_eq!(e.aliases, vec!["الشخص", "Person"]);
        assert_eq!(e.role, "Team lead");
    }

    #[test]
    fn parses_fenced_json() {
        let t = parse("Here:\n```json\n{\"entities\":[{\"name\":\"Speaker 2\",\"type\":\"person\",\"timestamps\":[\"00:12\"]},{\"name\":\" \"}],\"actions\":[{\"text\":\"send report\",\"owner\":\"Speaker 2\",\"due\":\"Thursday\"}]}\n```").unwrap();
        assert_eq!(t.entities.len(), 1);
        assert_eq!(t.actions[0].due, "Thursday");
        assert!(parse("no json here").is_err());
    }

    #[test]
    fn kinds_and_keys() {
        assert_eq!(kind("Organisation"), "org");
        assert_eq!(kind("Project"), "initiative");
        assert_eq!(kind("weird"), "topic");
        assert_eq!(key_for("person", "Dr. Person  One"), "person:person one");
        assert_eq!(norm("الدكتور الشخص"), "الشخص");
    }
}
