use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::{fs, process};

use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::{Message, MessageType};
use tf_demo_parser::demo::parser::{DemoParser, MessageHandler, ParserState};
use tf_demo_parser::demo::packet::datatable::{ParseSendTable, ServerClass};
use tf_demo_parser::demo::sendprop::{SendPropIdentifier, SendPropValue};
use tf_demo_parser::Demo;

fn ident(t: &str, n: &str) -> SendPropIdentifier {
    SendPropIdentifier::new(t, n)
}

fn f32_of(props: &HashMap<SendPropIdentifier, SendPropValue>, t: &str, n: &str) -> f32 {
    match props.get(&ident(t, n)) {
        Some(SendPropValue::Float(v)) => *v,
        Some(SendPropValue::Integer(v)) => *v as f32,
        _ => f32::NAN,
    }
}

fn comp_of(
    props: &HashMap<SendPropIdentifier, SendPropValue>,
    tables: &[&str],
    name: &str,
) -> f32 {
    for t in tables {
        match props.get(&ident(t, name)) {
            Some(SendPropValue::Float(v)) => return *v,
            Some(SendPropValue::Integer(v)) => return *v as f32,
            Some(SendPropValue::Vector(v)) => return v.x,
            Some(SendPropValue::VectorXY(v)) => return v.x,
            _ => {}
        }
    }
    f32::NAN
}

fn i64_of(props: &HashMap<SendPropIdentifier, SendPropValue>, t: &str, n: &str) -> i64 {
    match props.get(&ident(t, n)) {
        Some(SendPropValue::Integer(v)) => *v,
        Some(SendPropValue::Float(v)) => *v as i64,
        _ => -1,
    }
}

struct Ent {
    class: String,
    props: HashMap<SendPropIdentifier, SendPropValue>,
}

struct Dump {
    ents: HashMap<usize, Ent>,
    out: BufWriter<File>,
    pout: BufWriter<File>,
    rows: u64,
    // Dump-time liveness ledger: per player entity, counters for
    // [rows, pos_updates, eye_updates, mspd_updates]. The pipeline once trained
    // for months on a FROZEN eye-yaw column (SourceTV never populates
    // m_angEyeAngles for remote players) and nothing noticed. This ledger is
    // printed at the end of every run so a dead column is visible immediately.
    live: HashMap<usize, [u64; 4]>,
    last: HashMap<usize, (i64, i64, i64, u32)>, // quantized (ox, oy, ey), mspd bits
}

fn xy(
    props: &HashMap<SendPropIdentifier, SendPropValue>,
    tables: &[&str],
    name: &str,
) -> (f32, f32) {
    for t in tables {
        if let Some(v) = props.get(&ident(t, name)) {
            match v {
                SendPropValue::Vector(v) => return (v.x, v.y),
                SendPropValue::VectorXY(v) => return (v.x, v.y),
                _ => {}
            }
        }
    }
    (f32::NAN, f32::NAN)
}

fn is_projectile(class: &str) -> bool {
    class.starts_with("CTFProjectile")
        || class.starts_with("CTFGrenade")
        || class.starts_with("CTFBaseRocket")
        || class.starts_with("CTFWeaponBaseGrenadeProj")
        || class.starts_with("CTFFlameRocket")
        || class.starts_with("CTFStunBall")
        || class.starts_with("CTFBall_Ornament")
}

impl Dump {
    fn porigin(&self, e: &Ent) -> (f32, f32, f32) {
        const T: &[&str] = &[
            "DT_TFBaseRocket",
            "DT_TFWeaponBaseGrenadeProj",
            "DT_BaseEntity",
            "DT_TFNonLocalPlayerExclusive",
        ];
        let (x, y) = xy(&e.props, T, "m_vecOrigin");
        let mut z = f32::NAN;
        for t in T {
            let v = f32_of(&e.props, t, "m_vecOrigin[2]");
            if !v.is_nan() {
                z = v;
                break;
            }
        }
        (x, y, z)
    }

    fn pvel(&self, e: &Ent) -> (f32, f32, f32) {
        const T: &[&str] = &[
            "DT_TFBaseRocket",
            "DT_TFWeaponBaseGrenadeProj",
            "DT_BaseGrenade",
        ];
        for t in T {
            if let Some(v) = e.props.get(&ident(t, "m_vInitialVelocity")) {
                match v {
                    SendPropValue::Vector(v) => return (v.x, v.y, v.z),
                    SendPropValue::VectorXY(v) => return (v.x, v.y, f32::NAN),
                    _ => {}
                }
            }
        }
        (f32::NAN, f32::NAN, f32::NAN)
    }
    fn origin(&self, e: &Ent) -> (f32, f32, f32) {
        // SourceTV: all players are non-local. A stale Local baseline shadows
        // fresh NonLocal deltas if preferred, so NonLocal goes FIRST.
        const T: &[&str] = &[
            "DT_TFNonLocalPlayerExclusive",
            "DT_TFLocalPlayerExclusive",
            "DT_TFPlayer",
            "DT_BaseEntity",
        ];
        let (x, y) = xy(&e.props, T, "m_vecOrigin");
        let mut z = f32::NAN;
        for t in T {
            let v = f32_of(&e.props, t, "m_vecOrigin[2]");
            if !v.is_nan() {
                z = v;
                break;
            }
        }
        (x, y, z)
    }

    fn vel(&self, e: &Ent) -> (f32, f32, f32) {
        const T: &[&str] = &[
            "DT_TFPlayer",
            "DT_BasePlayer",
            "DT_BaseEntity",
            "DT_LocalPlayerExclusive",
        ];
        // [0] may be VectorXY (x,y) with [1]/[2] floats, or three floats
        let x = comp_of(&e.props, T, "m_vecVelocity[0]");
        let mut y = comp_of(&e.props, T, "m_vecVelocity[1]");
        let mut z = comp_of(&e.props, T, "m_vecVelocity[2]");
        if y.is_nan() || z.is_nan() {
            for t in T {
                if let Some(SendPropValue::VectorXY(v)) =
                    e.props.get(&ident(t, "m_vecVelocity[0]"))
                {
                    if y.is_nan() {
                        y = v.y;
                    }
                    if z.is_nan() {
                        z = comp_of(&e.props, T, "m_vecVelocity[1]");
                    }
                    break;
                }
            }
        }
        (x, y, z)
    }

    // Per-tick max speed. In TF2 the run-speed cap is a NETWORKED prop
    // (CTFPlayer::SetMaxSpeed -> m_flMaxspeed), so the wishspeed clamp in a
    // ProcessMovement rollout can use the real value instead of a per-class
    // constant. Table list mirrors origin(): NonLocal before Local.
    fn maxspeed(&self, e: &Ent, idx: usize) -> f32 {
        const T: &[&str] = &[
            "DT_TFPlayer",
            "DT_BasePlayer",
            "DT_TFNonLocalPlayerExclusive",
            "DT_TFLocalPlayerExclusive",
        ];
        for t in T {
            let v = f32_of(&e.props, t, "m_flMaxspeed");
            if !v.is_nan() {
                return v;
            }
        }
        // player-resource flattened array, e.g. m_iMaxSpeed003
        for t in ["DT_TFPlayerResource", "DT_PlayerResource"] {
            for n in ["m_iMaxSpeed", "m_flMaxSpeed", "m_iMaxSpeedForPlayer"] {
                let key = format!("{:03}{}", idx, n);
                let v = f32_of(&e.props, t, &key);
                if !v.is_nan() {
                    return v;
                }
                let key2 = format!("{}{:03}", n, idx);
                let v = f32_of(&e.props, t, &key2);
                if !v.is_nan() {
                    return v;
                }
            }
        }
        f32::NAN
    }

    fn eye(&self, e: &Ent) -> (f32, f32) {
        // NonLocal first (see origin()): the Local table holds a stale baseline
        // that must NEVER shadow the live NonLocal value. NOTE: this was once
        // written as last-wins, which froze eyey at the spawn angle for months.
        // First non-NaN wins, per component.
        const T: &[&str] = &["DT_TFNonLocalPlayerExclusive", "DT_TFLocalPlayerExclusive"];
        let mut p = f32::NAN;
        let mut y = f32::NAN;
        for t in T {
            if p.is_nan() {
                let a = f32_of(&e.props, t, "m_angEyeAngles[0]");
                if !a.is_nan() {
                    p = a;
                }
            }
            if y.is_nan() {
                let b = f32_of(&e.props, t, "m_angEyeAngles[1]");
                if !b.is_nan() {
                    y = b;
                }
            }
        }
        (p, y)
    }
}

impl MessageHandler for Dump {
    type Output = ();

    fn does_handle(message_type: MessageType) -> bool {
        matches!(message_type, MessageType::PacketEntities)
    }

    fn handle_message(&mut self, message: &Message, tick: DemoTick, ps: &ParserState) {
        if let Message::PacketEntities(pe) = message {
            if std::env::var("DUMP_BIG").is_ok() {
                for ent in pe.entities.iter() {
                    let cls = ps.server_classes[usize::from(ent.server_class)]
                        .name
                        .to_string();
                    if cls != "CTFPlayer" {
                        continue;
                    }
                    // count candidate origin/velocity identifiers present in deltas
                    for (t, n) in [
                        ("DT_TFPlayer", "m_vecOrigin"),
                        ("DT_TFPlayer", "m_vecOrigin[2]"),
                        ("DT_BaseEntity", "m_vecOrigin"),
                        ("DT_TFLocalPlayerExclusive", "m_vecOrigin"),
                        ("DT_TFNonLocalPlayerExclusive", "m_vecOrigin"),
                        ("DT_LocalPlayerExclusive", "m_vecVelocity[0]"),
                        ("DT_LocalPlayerExclusive", "m_vecVelocity[1]"),
                        ("DT_TFPlayer", "m_vecVelocity[0]"),
                    ] {
                        let id = SendPropIdentifier::new(t, n);
                        if ent.props.iter().any(|p| p.identifier == id) {
                            println!("HIT {t}.{n}");
                            self.rows += 1;
                        }
                    }
                    if (ent.props.len() as u64) > self.rows && self.rows < 1000000 {
                        self.rows = ent.props.len() as u64;
                        println!(
                            "BIG idx={} nprops={} tick={:?}",
                            ent.entity_index,
                            ent.props.len(),
                            tick
                        );
                        for p in ent.props.iter() {
                            println!("  {:?} = {}", p.identifier, p.value);
                        }
                    }
                }
            }
            if std::env::var("DUMP_RAW").is_ok() {
                // Full delta contents for one entity: every identifier hash +
                // value, so nothing can hide behind a wrong name guess.
                let want_ent: usize = std::env::var("DUMP_ENT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2);
                let max_tick: u32 = std::env::var("DUMP_MAXTICK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1300);
                if u32::from(tick) <= max_tick {
                    for ent in pe.entities.iter() {
                        if usize::from(ent.entity_index) != want_ent {
                            continue;
                        }
                        let mut parts: Vec<String> = Vec::new();
                        for p in ent.props.iter() {
                            parts.push(format!("{:?}={:?}", p.identifier, p.value));
                        }
                        println!(
                            "RAW tick={} ent={} [{}]",
                            u32::from(tick),
                            want_ent,
                            parts.join(", ")
                        );
                    }
                }
            }
            if std::env::var("DUMP_STUN").is_ok() && (u32::from(tick) % 200) < 2 {
                // Airblast-victim detection probe: TF2 may flag airblast via the
                // shared stun props (m_hStunner = the pyro's handle -> direction).
                // Prints only entities carrying nonzero stun state.
                for (idx2, e2) in self.ents.iter() {
                    if e2.class != "CTFPlayer" {
                        continue;
                    }
                    let mut parts: Vec<String> = Vec::new();
                    for (t, n) in [
                        ("DT_TFPlayerShared", "m_iStunFlags"),
                        ("DT_TFPlayerShared", "m_iStunIndex"),
                        ("DT_TFPlayerShared", "m_hStunner"),
                        ("DT_TFPlayerShared", "m_flMovementStunTime"),
                        ("DT_TFPlayerShared", "m_iMovementStunAmount"),
                        ("DT_TFPlayerShared", "m_nPlayerCond"),
                    ] {
                        let v = i64_of(&e2.props, t, n);
                        if v != -1 && v != 0 {
                            parts.push(format!("{t}.{n}={v}"));
                        }
                    }
                    if !parts.is_empty() {
                        println!(
                            "STUN tick={} ent={} [{}]",
                            u32::from(tick),
                            idx2,
                            parts.join(", ")
                        );
                    }
                }
            }
            if std::env::var("DUMP_EYE").is_ok() {
                // Packet-level truth: which eye-angle identifiers actually ARRIVE
                // in per-packet deltas for CTFPlayer, and with what values.
                // (table, prop) candidates cover vector vs flattened namings.
                let cands: Vec<(String, String, SendPropIdentifier)> = [
                    "DT_TFNonLocalPlayerExclusive",
                    "DT_TFLocalPlayerExclusive",
                    "DT_TFPlayer",
                    "DT_BasePlayer",
                ]
                .iter()
                .flat_map(|t| {
                    [
                        "m_angEyeAngles",
                        "m_angEyeAngles[0]",
                        "m_angEyeAngles[1]",
                        "m_angEyeAngles[2]",
                    ]
                    .iter()
                    .map(move |n| {
                        (
                            t.to_string(),
                            n.to_string(),
                            SendPropIdentifier::new(t, n),
                        )
                    })
                })
                .collect();
                let want_ent: usize = std::env::var("DUMP_ENT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2);
                for ent in pe.entities.iter() {
                    if usize::from(ent.entity_index) != want_ent {
                        continue;
                    }
                    let mut hits: Vec<String> = Vec::new();
                    for p in ent.props.iter() {
                        for (t, n, id) in cands.iter() {
                            if p.identifier == *id {
                                hits.push(format!("{t}.{n}={:?}", p.value));
                            }
                        }
                    }
                    // always print one line per packet so absence is visible too
                    println!(
                        "EYE tick={} ent={} nprops={} hits=[{}]",
                        u32::from(tick),
                        want_ent,
                        ent.props.len(),
                        hits.join(", ")
                    );
                }
            }
            if std::env::var("DUMP_PROPS").is_ok() {
                let skip: usize = std::env::var("DUMP_SKIP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                for ent in pe.entities.iter() {
                    let cls = ps.server_classes[usize::from(ent.server_class)]
                        .name
                        .to_string();
                    if cls != "CTFPlayer" {
                        continue;
                    }
                    if self.rows < skip as u64 {
                        self.rows += 1;
                        continue;
                    }
                    println!(
                        "PLAYER idx={} nprops={} tick={}",
                        ent.entity_index,
                        ent.props.len(),
                        u32::from(tick),
                    );
                    for p in ent.props.iter() {
                        println!("  {:?} = {}", p.identifier, p.value);
                    }
                    self.rows += 1;
                    if self.rows > 6 {
                        std::process::exit(0);
                    }
                    continue;
                }
            }
            for r in pe.removed_entities.iter() {
                self.ents.remove(&usize::from(*r));
            }
            // entities actually present in THIS message's delta = fresh data
            use std::collections::HashSet;
            let fresh: HashSet<usize> =
                pe.entities.iter().map(|e| usize::from(e.entity_index)).collect();
            for ent in pe.entities.iter() {
                let idx = usize::from(ent.entity_index);
                let class =
                    ps.server_classes[usize::from(ent.server_class)].name.to_string();
                let e = self.ents.entry(idx).or_insert_with(|| Ent {
                    class: class.clone(),
                    props: HashMap::new(),
                });
                e.class = class;
                // seed baseline on first sight so deltas resolve to full state
                if e.props.is_empty() {
                    for p in ent.get_baseline_props(ps).iter() {
                        e.props.insert(p.identifier, p.value.clone());
                    }
                }
                for p in ent.props.iter() {
                    e.props.insert(p.identifier, p.value.clone());
                }
            }
            // snapshot players after applying this message
            // class labels via CTFPlayerResource m_iPlayerClass[idx] (flattened array elements)
            let mut res_props: Option<&HashMap<SendPropIdentifier, SendPropValue>> = None;
            for e in self.ents.values() {
                if e.class == "CTFPlayerResource" {
                    res_props = Some(&e.props);
                    break;
                }
            }
            for (idx, e) in self.ents.iter() {
                if e.class != "CTFPlayer" && !is_projectile(&e.class) {
                    continue;
                }
                if is_projectile(&e.class) {
                    let (ox, oy, oz) = self.porigin(e);
                    if ox.is_nan() {
                        continue;
                    }
                    let (vx, vy, vz) = self.pvel(e);
                    let line = format!(
                        "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}\n",
                        u32::from(tick),
                        idx,
                        e.class,
                        ox,
                        oy,
                        oz,
                        vx,
                        vy,
                        vz,
                        i64_of(&e.props, "DT_BaseEntity", "m_iTeamNum"),
                        if fresh.contains(idx) { 1 } else { 0 },
                    );
                    self.pout.write_all(line.as_bytes()).unwrap();
                    continue;
                }
                let fr = if fresh.contains(idx) { 1 } else { 0 };
                let (ox, oy, oz) = self.origin(e);
                if ox.is_nan() {
                    continue;
                }
                let (vx, vy, vz) = self.vel(e);
                let (ep, ey) = self.eye(e);
                let mspd = self.maxspeed(e, *idx);
                if std::env::var("DUMP_SPEED").is_ok() && (u32::from(tick) % 500) < 3 {
                    let mut cands: Vec<String> = Vec::new();
                    for t in [
                        "DT_TFPlayer",
                        "DT_BasePlayer",
                        "DT_TFNonLocalPlayerExclusive",
                        "DT_TFLocalPlayerExclusive",
                    ] {
                        for n in [
                            "m_flMaxspeed",
                            "m_flMaxSpeed",
                            "m_flMovementSpeed",
                            "m_flSpeed",
                        ] {
                            let v = f32_of(&e.props, t, n);
                            if !v.is_nan() {
                                cands.push(format!("{t}.{n}={v}"));
                            }
                        }
                    }
                    if let Some(rp) = res_props {
                        for n in ["m_iMaxSpeed", "m_flMaxSpeed", "m_iPlayerSpeed"] {
                            for f in [format!("{n}{:03}", idx), format!("{:03}{n}", idx)] {
                                let v = f32_of(rp, "DT_TFPlayerResource", &f);
                                if !v.is_nan() {
                                    cands.push(format!("RES.{f}={v}"));
                                }
                            }
                        }
                    }
                    println!(
                        "SPEED tick={} ent={} mspd={} cands=[{}]",
                        u32::from(tick),
                        idx,
                        mspd,
                        cands.join(", ")
                    );
                }
                let mut cls = i64_of(&e.props, "DT_TFPlayerClassShared", "m_iClass");
                if cls < 0 {
                    if let Some(rp) = res_props {
                        cls = i64_of(
                            rp,
                            "m_iPlayerClass",
                            &format!("{:03}", idx),
                        );
                    }
                }
                let line = format!(
                    "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{}\n",
                    u32::from(tick),
                    idx,
                    cls,
                    ox,
                    oy,
                    oz,
                    vx,
                    vy,
                    vz,
                    ep,
                    ey,
                    mspd,
                    i64_of(&e.props, "DT_BasePlayer", "m_fFlags"),
                    i64_of(&e.props, "DT_BasePlayer", "m_iHealth"),
                    i64_of(&e.props, "DT_BasePlayer", "m_lifeState"),
                    i64_of(&e.props, "DT_BaseEntity", "m_iTeamNum"),
                    fr,
                );
                self.out.write_all(line.as_bytes()).unwrap();
                self.rows += 1;
                // liveness ledger at CSV resolution ({:.2}): a column that never
                // updates for a moving player is a BUG (cf. frozen eyey).
                {
                    let q = |x: f32| (x * 100.0).round() as i64;
                    let cur = (q(ox), q(oy), q(ey), mspd.to_bits());
                    let e2 = self.live.entry(*idx).or_insert([0, 0, 0, 0]);
                    e2[0] += 1;
                    if let Some(prev) = self.last.get(idx) {
                        if prev.0 != cur.0 || prev.1 != cur.1 {
                            e2[1] += 1;
                        }
                        if prev.2 != cur.2 {
                            e2[2] += 1;
                        }
                        if prev.3 != cur.3 {
                            e2[3] += 1;
                        }
                    }
                    self.last.insert(*idx, cur);
                }
            }
        }
    }

    fn handle_header(&mut self, _header: &Header) {}
    fn into_output(self, _state: &ParserState) -> Self::Output {
        let mut ids: Vec<usize> = self.live.keys().cloned().collect();
        ids.sort_unstable();
        for idx in ids {
            let c = self.live.get(&idx).cloned().unwrap_or([0, 0, 0, 0]);
            let mut flags = String::new();
            if c[0] > 500 && c[1] == 0 {
                flags.push_str(" STALE");
            }
            if c[1] > 0 && c[2] == 0 {
                // remote SourceTV players never get eye updates; every consumer
                // must treat eyey as a spawn constant, never a time series
                flags.push_str(" EYE-FROZEN");
            }
            if c[1] > 0 && c[3] == 0 {
                flags.push_str(" MSPD-FROZEN");
            }
            eprintln!(
                "LIVE ent={} rows={} pos#={} eye#={} mspd#={}{}",
                idx, c[0], c[1], c[2], c[3], flags
            );
        }
    }

    fn handle_data_tables(
        &mut self,
        tables: &[ParseSendTable],
        _server_classes: &[ServerClass],
        _parser_state: &ParserState,
    ) {
        if std::env::var("DUMP_PTABLE").is_ok() {
            // Full prop inventory for every *Player* table: pick networked
            // quantities by name+type instead of guessing hashes.
            for t in tables {
                let tn = t.name.as_str();
                if !tn.contains("Player") {
                    continue;
                }
                for p in t.props.iter() {
                    println!("PROP {tn}.{} : {:?} {:?}", p.name.as_str(), p.prop_type, p.flags);
                }
            }
            std::process::exit(0);
        }
        if std::env::var("DUMP_TABLES").is_ok() {
            for t in tables {
                let tn = t.name.as_str();
                for p in t.props.iter() {
                    let n = p.name.as_str();
                    let nl = n.to_lowercase();
                    if tn.contains("Exclusive")
                        || nl.contains("veloc")
                        || nl.contains("eyeangles")
                        || nl.contains("eyeposition")
                    {
                        println!("TABLE {tn}.{n}");
                    }
                }
            }
            std::process::exit(0);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: demodump <in.dem> <out.csv>");
        process::exit(2);
    }
    let data = fs::read(&args[1]).expect("read demo");
    let out = BufWriter::new(File::create(&args[2]).expect("create csv"));
    let ppath = args[2].clone() + ".proj.csv";
    let pout = BufWriter::new(File::create(&ppath).expect("create proj csv"));
    let demo = Demo::new(&data);
    let stream = demo.get_stream();
    let mut d = Dump { ents: HashMap::new(), out, pout, rows: 0, live: HashMap::new(), last: HashMap::new() };
    let header_out = format!("tick,ent,class,ox,oy,oz,vx,vy,vz,eyep,eyey,mspd,flags,health,life,team,fresh\n");
    d.out.write_all(header_out.as_bytes()).unwrap();
    d.pout.write_all(b"tick,ent,class,ox,oy,oz,vx,vy,vz,team,fresh\n").unwrap();
    let parser = DemoParser::new_all_with_analyser(stream, d);
    let (_header, _out) = parser.parse().expect("parse");
    eprintln!("done");
}
