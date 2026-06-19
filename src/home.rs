//! Single home for all higgs identity/state: `~/.higgs` (override with `HIGGS_HOME`).

use std::path::PathBuf;

/// The higgs home directory: `$HIGGS_HOME` if set, else `~/.higgs`.
pub fn higgs_home() -> PathBuf {
    if let Some(over) = std::env::var_os("HIGGS_HOME") {
        return PathBuf::from(over);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".higgs")
}

/// Create the home directory if absent; returns its path.
pub fn ensure_home() -> std::io::Result<PathBuf> {
    let home = higgs_home();
    std::fs::create_dir_all(&home)?;
    Ok(home)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honors_higgs_home_override() {
        // SAFETY: single-threaded test; the var is removed before returning.
        let tmp = std::env::temp_dir().join("higgs-home-override-test");
        unsafe { std::env::set_var("HIGGS_HOME", &tmp) };
        assert_eq!(higgs_home(), tmp);
        unsafe { std::env::remove_var("HIGGS_HOME") };
    }
}
