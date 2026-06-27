use super::*;

#[test]
fn alloc_is_monotonic_and_insert_get_remove() {
    let mut reg: WorkerRegistry<u8> = WorkerRegistry::new();
    let a = reg.insert(10);
    let b = reg.insert(20);
    assert_ne!(a, b);
    assert!(b.0 > a.0, "ids are monotonic");
    assert_eq!(reg.get(a), Some(&10));
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.ids(), vec![a, b]);
    assert_eq!(reg.remove(a), Some(10));
    assert_eq!(reg.get(a), None);
    // ids are never reused even after removal
    let c = reg.insert(30);
    assert!(c.0 > b.0, "freed id is not reused");
}

#[test]
fn worker_id_renders_as_w_prefix() {
    assert_eq!(WorkerId(1).to_string(), "w-1");
}

#[test]
fn empty_registry() {
    let reg: WorkerRegistry<u8> = WorkerRegistry::default();
    assert!(reg.is_empty());
    assert!(reg.ids().is_empty());
}
