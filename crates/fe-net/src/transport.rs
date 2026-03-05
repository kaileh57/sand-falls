//! Packet transports: real UDP and a simulated network.

use fe_core::SeqRng;
use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};

/// A remote endpoint. Its meaning depends on the transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Peer(pub u64);

/// Sends and receives datagrams.
pub trait Transport {
    /// Sends one datagram. Delivery is not guaranteed.
    fn send(&mut self, to: Peer, data: &[u8]);
    /// Returns the next received datagram, if any.
    fn recv(&mut self) -> Option<(Peer, Vec<u8>)>;
}

/// UDP transport. Peers are numbered as they first appear.
pub struct UdpTransport {
    socket: UdpSocket,
    by_addr: BTreeMap<SocketAddr, Peer>,
    by_peer: BTreeMap<Peer, SocketAddr>,
    buf: Vec<u8>,
}

impl UdpTransport {
    /// Binds a non-blocking socket.
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            by_addr: BTreeMap::new(),
            by_peer: BTreeMap::new(),
            buf: vec![0; 65_536],
        })
    }

    /// Registers a remote address and returns its peer id.
    pub fn peer_for(&mut self, addr: SocketAddr) -> Peer {
        if let Some(p) = self.by_addr.get(&addr) {
            return *p;
        }
        let p = Peer(self.by_addr.len() as u64 + 1);
        self.by_addr.insert(addr, p);
        self.by_peer.insert(p, addr);
        p
    }

    /// The local address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }
}

impl Transport for UdpTransport {
    fn send(&mut self, to: Peer, data: &[u8]) {
        if let Some(addr) = self.by_peer.get(&to) {
            let _ = self.socket.send_to(data, addr);
        }
    }

    fn recv(&mut self) -> Option<(Peer, Vec<u8>)> {
        match self.socket.recv_from(&mut self.buf) {
            Ok((n, addr)) => {
                let p = self.peer_for(addr);
                Some((p, self.buf[..n].to_vec()))
            }
            Err(_) => None,
        }
    }
}

/// Settings for the simulated network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimNetConfig {
    /// Round-trip time in milliseconds (half each way).
    pub rtt_ms: u64,
    /// Random extra delay per packet, up to this many milliseconds.
    pub jitter_ms: u64,
    /// Percentage of packets dropped.
    pub loss_pct: u32,
    /// Seed for loss and jitter.
    pub seed: u64,
}

impl Default for SimNetConfig {
    fn default() -> Self {
        Self { rtt_ms: 40, jitter_ms: 5, loss_pct: 0, seed: 1 }
    }
}

struct InFlight {
    deliver_at: u64,
    seq: u64,
    from: Peer,
    to: Peer,
    data: Vec<u8>,
}

/// A simulated network shared by several endpoints, driven by a virtual clock.
///
/// Delivery order, delay and loss come from a seeded generator, so a soak run
/// over it is reproducible.
pub struct SimNet {
    cfg: SimNetConfig,
    rng: SeqRng,
    now_us: u64,
    seq: u64,
    queue: Vec<InFlight>,
    inbox: BTreeMap<Peer, Vec<(Peer, Vec<u8>)>>,
    /// Packets dropped so far.
    pub dropped: u64,
    /// Packets delivered so far.
    pub delivered: u64,
}

impl SimNet {
    /// Creates a network.
    pub fn new(cfg: SimNetConfig) -> Self {
        Self {
            cfg,
            rng: SeqRng::new(cfg.seed),
            now_us: 0,
            seq: 0,
            queue: Vec::new(),
            inbox: BTreeMap::new(),
            dropped: 0,
            delivered: 0,
        }
    }

    /// Current virtual time in microseconds.
    pub const fn now(&self) -> u64 {
        self.now_us
    }

    /// Advances the clock and delivers everything due.
    pub fn advance_to(&mut self, now_us: u64) {
        self.now_us = now_us;
        let mut due: Vec<InFlight> = Vec::new();
        let mut rest = Vec::with_capacity(self.queue.len());
        for p in self.queue.drain(..) {
            if p.deliver_at <= now_us { due.push(p) } else { rest.push(p) }
        }
        self.queue = rest;
        due.sort_by_key(|p| (p.deliver_at, p.seq));
        for p in due {
            self.inbox.entry(p.to).or_default().push((p.from, p.data));
            self.delivered += 1;
        }
    }

    /// Sends from one endpoint to another.
    pub fn send(&mut self, from: Peer, to: Peer, data: &[u8]) {
        if self.rng.below(100) < self.cfg.loss_pct {
            self.dropped += 1;
            return;
        }
        let jitter = if self.cfg.jitter_ms > 0 {
            u64::from(self.rng.below(self.cfg.jitter_ms as u32 * 1000))
        } else {
            0
        };
        let deliver_at = self.now_us + self.cfg.rtt_ms * 500 + jitter;
        self.seq += 1;
        self.queue.push(InFlight { deliver_at, seq: self.seq, from, to, data: data.to_vec() });
    }

    /// Takes the next delivered packet for an endpoint.
    pub fn recv(&mut self, at: Peer) -> Option<(Peer, Vec<u8>)> {
        let inbox = self.inbox.get_mut(&at)?;
        if inbox.is_empty() { None } else { Some(inbox.remove(0)) }
    }

    /// A [`Transport`] view of this network for one endpoint.
    pub fn endpoint(&mut self, me: Peer) -> SimEndpoint<'_> {
        SimEndpoint { net: self, me }
    }
}

/// One endpoint's view of a [`SimNet`].
pub struct SimEndpoint<'a> {
    net: &'a mut SimNet,
    me: Peer,
}

impl Transport for SimEndpoint<'_> {
    fn send(&mut self, to: Peer, data: &[u8]) {
        self.net.send(self.me, to, data);
    }

    fn recv(&mut self) -> Option<(Peer, Vec<u8>)> {
        self.net.recv(self.me)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simnet_delays_and_drops() {
        let mut n = SimNet::new(SimNetConfig { rtt_ms: 100, jitter_ms: 0, loss_pct: 0, seed: 3 });
        n.send(Peer(1), Peer(2), b"hi");
        n.advance_to(49_000);
        assert!(n.recv(Peer(2)).is_none());
        n.advance_to(50_000);
        assert_eq!(n.recv(Peer(2)), Some((Peer(1), b"hi".to_vec())));
        let mut lossy =
            SimNet::new(SimNetConfig { rtt_ms: 0, jitter_ms: 0, loss_pct: 50, seed: 9 });
        for _ in 0..1000 {
            lossy.send(Peer(1), Peer(2), b"x");
        }
        assert!((400..600).contains(&lossy.dropped), "{}", lossy.dropped);
    }
}
