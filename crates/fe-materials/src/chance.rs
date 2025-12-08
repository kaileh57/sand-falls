/// A probability of 1, in units of 1/65,536.
pub const CHANCE_ONE: u32 = 65_536;

/// Parses a probability written as a percentage (`"2.5%"`) or a fraction (`"1/40"`).
///
/// Returns the probability in units of 1/65,536, rounded down. Parsing uses
/// integer math only, so the result is identical on every platform.
pub fn parse_chance(text: &str) -> Option<u32> {
    let t = text.trim();
    if let Some(pct) = t.strip_suffix('%') {
        let (num, den) = parse_decimal(pct.trim())?;
        // value = num / den percent = num / (den * 100)
        let v = u128::from(num) * u128::from(CHANCE_ONE) / (u128::from(den) * 100);
        return (v <= u128::from(CHANCE_ONE)).then_some(v as u32);
    }
    if let Some((a, b)) = t.split_once('/') {
        let a: u64 = a.trim().parse().ok()?;
        let b: u64 = b.trim().parse().ok()?;
        if b == 0 || a > b {
            return None;
        }
        return Some((u128::from(a) * u128::from(CHANCE_ONE) / u128::from(b)) as u32);
    }
    None
}

/// Parses `"12.345"` into `(12345, 1000)`.
fn parse_decimal(s: &str) -> Option<(u64, u64)> {
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    if !int.chars().all(|c| c.is_ascii_digit()) || !frac.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if frac.len() > 9 {
        return None;
    }
    let den = 10u64.pow(frac.len() as u32);
    let i: u64 = if int.is_empty() { 0 } else { int.parse().ok()? };
    let f: u64 = if frac.is_empty() { 0 } else { frac.parse().ok()? };
    Some((i.checked_mul(den)?.checked_add(f)?, den))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentages() {
        assert_eq!(parse_chance("100%"), Some(CHANCE_ONE));
        assert_eq!(parse_chance("50%"), Some(32_768));
        assert_eq!(parse_chance("0.5%"), Some(327));
        assert_eq!(parse_chance(" 25 %"), Some(16_384));
        assert_eq!(parse_chance("101%"), None);
        assert_eq!(parse_chance("abc%"), None);
        assert_eq!(parse_chance("%"), None);
    }

    #[test]
    fn fractions() {
        assert_eq!(parse_chance("1/2"), Some(32_768));
        assert_eq!(parse_chance("1/65536"), Some(1));
        assert_eq!(parse_chance("2/1"), None);
        assert_eq!(parse_chance("1/0"), None);
    }
}
