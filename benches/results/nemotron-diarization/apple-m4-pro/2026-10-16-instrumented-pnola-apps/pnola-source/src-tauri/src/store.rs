use std::{fs, path::Path, sync::Mutex};
#[derive(Default)]
pub struct Store {
    write: Mutex<()>,
}
impl Store {
    pub fn read(&self, path: &Path) -> Result<serde_json::Value, String> {
        if !path.exists() {
            return Ok(serde_json::json!([]));
        }
        let data = fs::read(path).map_err(|e| e.to_string())?;
        if data.len() > 16 * 1024 * 1024 {
            return Err("History file is too large.".into());
        }
        let entries: serde_json::Value =
            serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        if !entries.is_array() {
            return Err("History is not a transcript list.".into());
        }
        Ok(entries)
    }
    pub fn save(&self, path: &Path, entries: serde_json::Value) -> Result<(), String> {
        if !entries.as_array().is_some_and(|v| v.len() <= 20) {
            return Err("History supports the latest 20 transcripts.".into());
        }
        let data = serde_json::to_vec(&entries).map_err(|e| e.to_string())?;
        if data.len() > 16 * 1024 * 1024 {
            return Err("History is too large. Export some transcripts first.".into());
        }
        let _guard = self.write.lock().unwrap();
        fs::create_dir_all(path.parent().ok_or("Missing history directory")?)
            .map_err(|e| e.to_string())?;
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, data).map_err(|e| e.to_string())?;
        fs::rename(temporary, path).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_save_preserves_existing_history() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcripts.json");
        let store = Store::default();
        let entries = serde_json::json!([{"id":"saved","text":"Keep this transcript"}]);
        store.save(&path, entries.clone()).unwrap();
        assert!(
            store
                .save(&path, serde_json::json!({"wrong":"shape"}))
                .is_err()
        );
        assert_eq!(store.read(&path).unwrap(), entries);
    }
}
