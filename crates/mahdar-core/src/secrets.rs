//! The Claude API key, pasted in Settings.
//!
//! Kept in the Windows Credential Manager (service "mahdar") and nowhere else: not in
//! settings.json, meeting files or the log. Lookup order: Mahdar's own entry, then the key
//! saved in Mizan (service "mizan", same name), then `ANTHROPIC_API_KEY`, so one key can
//! serve both apps. Anything shown to the user goes through [`mask`].

use anyhow::{Context, Result};

pub const SERVICE: &str = "mahdar";
const MIZAN_SERVICE: &str = "mizan";
const NAME: &str = "anthropic_api_key";

fn get(service: &str) -> Option<String> {
    keyring::Entry::new(service, NAME).ok()?.get_password().ok().filter(|k| !k.trim().is_empty())
}

/// The key and where it came from ("Mahdar", "Mizan" or "environment").
pub fn anthropic_key() -> Option<(String, &'static str)> {
    if let Some(k) = get(SERVICE) {
        return Some((k, "Mahdar"));
    }
    if let Some(k) = get(MIZAN_SERVICE) {
        return Some((k, "Mizan"));
    }
    std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty()).map(|k| (k, "environment"))
}

pub fn set_anthropic_key(key: &str) -> Result<()> {
    keyring::Entry::new(SERVICE, NAME)?.set_password(key.trim()).context("could not write to the Windows Credential Manager")
}

/// Remove Mahdar's own entry (a key shared from Mizan stays in Mizan).
pub fn delete_anthropic_key() -> Result<()> {
    match keyring::Entry::new(SERVICE, NAME)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e).context("could not remove the key"),
    }
}

/// "sk-ant-…WXYZ": enough to recognise a key, never enough to use it.
pub fn mask(key: &str) -> String {
    let k = key.trim();
    let chars: Vec<char> = k.chars().collect();
    if chars.len() < 12 {
        return "•".repeat(chars.len().min(8));
    }
    let head: String = chars[..7].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_hides_the_middle() {
        assert_eq!(mask("sk-ant-api03-abcdefghijklmnop-WXYZ"), "sk-ant-…WXYZ");
        assert_eq!(mask("short"), "•••••");
    }
}
