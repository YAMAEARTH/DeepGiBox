fn main() {
    let devices = decklink_rust::devicelist();
    for (i, name) in devices.iter().enumerate() {
        println!("{}: {}", i, name);
    }
}
