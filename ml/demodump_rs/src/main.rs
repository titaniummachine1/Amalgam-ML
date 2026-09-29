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

    fn eye(&self, e: &Ent) -> (f32, f32) {
        // NonLocal first (see origin()): stale Local baseline shadows otherwise.
        const T: &[&str] = &["DT_TFNonLocalPlayerExclusive", "DT_TFLocalPlayerExclusive"];
        let mut p = f32::NAN;
        let mut y = f32::NAN;
        for t in T {
            let a = f32_of(&e.props, t, "m_angEyeAngles[0]");
            let b = f32_of(&e.props, t, "m_angEyeAngles[1]");
            if !a.is_nan() {
                p = a;
            }
            if !b.is_nan() {
                y = b;
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
                    "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{}\n",
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
                    i64_of(&e.props, "DT_BasePlayer", "m_fFlags"),
                    i64_of(&e.props, "DT_BasePlayer", "m_iHealth"),
                    i64_of(&e.props, "DT_BasePlayer", "m_lifeState"),
                    i64_of(&e.props, "DT_BaseEntity", "m_iTeamNum"),
                    fr,
                );
                self.out.write_all(line.as_bytes()).unwrap();
                self.rows += 1;
            }
        }
    }

    fn handle_header(&mut self, _header: &Header) {}
    fn into_output(self, _state: &ParserState) -> Self::Output {}

    fn handle_data_tables(
        &mut self,
        tables: &[ParseSendTable],
        _server_classes: &[ServerClass],
        _parser_state: &ParserState,
    ) {
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
    let mut d = Dump { ents: HashMap::new(), out, pout, rows: 0 };
    let header_out = format!("tick,ent,class,ox,oy,oz,vx,vy,vz,eyep,eyey,flags,health,life,team,fresh\n");
    d.out.write_all(header_out.as_bytes()).unwrap();
    d.pout.write_all(b"tick,ent,class,ox,oy,oz,vx,vy,vz,team,fresh\n").unwrap();
    let parser = DemoParser::new_all_with_analyser(stream, d);
    let (_header, _out) = parser.parse().expect("parse");
    eprintln!("done");
}
