// tests/integration.rs
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_read_existing_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("README.md");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "Hello world").unwrap();

    let mut cmd = Command::cargo_bin("kataglyphis-rustprojecttemplate").unwrap();
    cmd.args(["read", "--path", file_path.to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Hello world"));
}

#[test]
fn test_read_nonexistent_file() {
    let mut cmd = Command::cargo_bin("kataglyphis-rustprojecttemplate").unwrap();
    cmd.args(["read", "--path", "nonexistent.txt"]);
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("Failed to read file"));
}

#[test]
fn test_stats_output() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("stats.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "First line\nSecond line with words\n").unwrap();

    let mut cmd = Command::cargo_bin("kataglyphis-rustprojecttemplate").unwrap();
    cmd.args(["stats", "--path", file_path.to_str().unwrap()]);
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Lines: 3"))
        .stdout(predicates::str::contains("Words: 6"))
        .stdout(predicates::str::contains("Bytes: "));
}
