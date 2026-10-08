use tf_demo_parser::demo::sendprop::SendPropIdentifier;
fn main() {
    let tables = [
        "DT_TFPlayer",
        "DT_BaseEntity",
        "DT_BasePlayer",
        "DT_TFLocalPlayerExclusive",
        "DT_TFNonLocalPlayerExclusive",
        "DT_LocalPlayerExclusive",
        "DT_TFPlayerClassShared",
        "DT_TFPlayerResource",
        "m_iPlayerClass",
    ];
    let names = [
        "m_vecOrigin",
        "m_vecOrigin[2]",
        "m_vecVelocity[0]",
        "m_vecVelocity[1]",
        "m_vecVelocity[2]",
        "m_angEyeAngles[0]",
        "m_angEyeAngles[1]",
        "m_iClass",
        "m_fFlags",
        "m_iHealth",
    ];
    for t in tables {
        for n in names {
            println!("{} . {} = {:?}", t, n, SendPropIdentifier::new(t, n));
        }
    }
    for i in 0..34 {
        println!(
            "m_iPlayerClass . {:03} = {:?}",
            i,
            SendPropIdentifier::new("m_iPlayerClass", &format!("{:03}", i))
        );
    }
}
