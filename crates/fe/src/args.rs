//! A small argument parser: positionals, `--flag`, and `--key value...`.

use std::collections::BTreeMap;

/// Options that take this many values.
const ARITY: &[(&str, usize)] = &[
    ("threads", 1),
    ("ticks", 1),
    ("warmup", 1),
    ("tick", 1),
    ("out", 1),
    ("every", 1),
    ("from", 1),
    ("rect", 4),
    ("until", 1),
    ("tcp", 1),
    ("frames-dir", 1),
];

pub struct Args {
    pub positional: Vec<String>,
    opts: BTreeMap<String, Vec<String>>,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut opts = BTreeMap::new();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            if let Some(name) = a.strip_prefix("--") {
                let n = ARITY.iter().find(|(k, _)| *k == name).map_or(0, |(_, n)| *n);
                let vals: Vec<String> = argv.iter().skip(i + 1).take(n).cloned().collect();
                if vals.len() != n {
                    return Err(format!("--{name} needs {n} value(s)"));
                }
                opts.insert(name.to_string(), vals);
                i += 1 + n;
            } else {
                positional.push(a.clone());
                i += 1;
            }
        }
        Ok(Self { positional, opts })
    }

    pub fn flag(&self, name: &str) -> bool {
        self.opts.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.opts.get(name).and_then(|v| v.first()).map(String::as_str)
    }

    pub fn values(&self, name: &str) -> Option<&[String]> {
        self.opts.get(name).map(Vec::as_slice)
    }

    pub fn num<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.get(name)
            .map(|v| v.parse().map_err(|_| format!("--{name} '{v}' is not a number")))
            .transpose()
    }

    pub fn threads(&self) -> usize {
        self.num::<usize>("threads").ok().flatten().unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
        })
    }

    pub fn pos(&self, i: usize, what: &str) -> Result<&str, String> {
        self.positional.get(i).map(String::as_str).ok_or_else(|| format!("missing {what}"))
    }
}
