use super::*;

#[test]
fn split_repo_requires_two_segments() {
    assert_eq!(split_repo("org/model"), Some(("org", "model")));
    assert_eq!(
        split_repo("bartowski/Qwen2.5-GGUF"),
        Some(("bartowski", "Qwen2.5-GGUF"))
    );
    assert_eq!(split_repo("gpt2"), None, "single segment");
    assert_eq!(split_repo("a/b/c"), None, "three segments");
    assert_eq!(split_repo("/model"), None, "empty owner");
    assert_eq!(split_repo("org/"), None, "empty name");
}

#[test]
fn classify_maps_each_hf_error_to_its_code() {
    let c = |e: &HFError| classify_hf("org/m", "m.gguf", e).to_string();
    assert!(c(&HFError::AuthRequired).starts_with("[HG029]"));
    assert!(c(&HFError::Forbidden).starts_with("[HG029]"));
    assert!(c(&HFError::RepoNotFound {
        repo_id: "org/m".into()
    })
    .starts_with("[HG030]"));
    assert!(c(&HFError::RevisionNotFound {
        repo_id: "org/m".into(),
        revision: "main".into()
    })
    .starts_with("[HG030]"));
    assert!(c(&HFError::EntryNotFound {
        path: "m.gguf".into(),
        repo_id: "org/m".into()
    })
    .starts_with("[HG030]"));
    assert!(c(&HFError::RateLimited).starts_with("[HG031]"));
    assert!(c(&HFError::Other("boom".into())).starts_with("[HG035]"));
    assert!(c(&HFError::Io(std::io::Error::other("disk"))).starts_with("[HG034]"));
}

#[test]
fn http_status_routes_to_distinct_codes() {
    // 401/403→auth(029), 404→not-found(030), 429→rate-limit(031), else→032.
    let h = |s: u16| http_status_to_error("org/m", "m.gguf", s, format!("HTTP {s}")).to_string();
    assert!(h(401).starts_with("[HG029]"));
    assert!(h(403).starts_with("[HG029]"));
    assert!(h(404).starts_with("[HG030]"));
    assert!(h(429).starts_with("[HG031]"));
    assert!(h(500).starts_with("[HG032]"));
    assert!(h(503).starts_with("[HG032]"));
}
