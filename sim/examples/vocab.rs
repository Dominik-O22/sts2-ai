//! Print the policy's vocabularies in index order. `cargo run --release --example vocab > vocab.txt`.
fn main() {
    print!("{}", sim::encode::vocab_text());
}
