/// Hamming Distance measures the similarity between two strings of the same length.
/// The Hamming Distance between two strings of the same length is the number of
/// positions at which the corresponding characters are different. Since the length
/// of these strings is equal, we can calculate the Hamming Distance.
pub fn hamming_distance(bytes1: &[u8], bytes2: &[u8]) -> usize {
    let a: u32 = bytes1
        .iter()
        .zip(bytes2)
        .map(|(byte1, byte2)| (byte1 ^ byte2).count_ones())
        .sum();
    a as usize
}

#[test]
fn test_hamming_distance() {
    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 0);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000001];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 1);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000010];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 1);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000011];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 2);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000100];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 1);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b00000101];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 2);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b00000000, 0b00000000, 0b11111111];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 8);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b11111111, 0b00000000, 0b11111111];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 16);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b00000000, 0b11111111, 0b11111111, 0b11111111];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 24);

    let bytes1 = [0b00000000, 0b00000000, 0b00000000, 0b00000000];
    let bytes2 = [0b11111111, 0b11111111, 0b11111111, 0b11111111];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 32);

    let bytes1 = [0b00000000, 0b11111111, 0b00000000, 0b11111111];
    let bytes2 = [0b11111111, 0b00000000, 0b11111111, 0b00000000];
    assert_eq!(hamming_distance(&bytes1, &bytes2), 32);
}
