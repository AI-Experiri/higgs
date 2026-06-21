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
        // Serialize with other env-mutating tests and restore the prior value (cargo runs lib
        // tests in parallel threads of one process).
        let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = std::env::var_os("HIGGS_HOME");
        let tmp = std::env::temp_dir().join("higgs-home-override-test");
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("HIGGS_HOME", &tmp) };
        assert_eq!(higgs_home(), tmp);
        // SAFETY: still under the lock.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HIGGS_HOME", v),
                None => std::env::remove_var("HIGGS_HOME"),
            }
        }
    }
}
