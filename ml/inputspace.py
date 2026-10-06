"""INPUT-SPACE policy: predict the usercmd (forwardmove, sidemove) instead of raw accel,
and roll it out through the real engine Accelerate math.

This module is the SINGLE SOURCE OF TRUTH for the rollout: the trainer and the
hit-rate evaluator both import it, because "eval dynamics must match training
dynamics" is the one lesson that cost this project a fake collapse (STATUS.md).

PHYSICS (all in dataset units, i.e. the same units the labels were built in):
  friction-first order, per tick, ground only:
      drop  = max(|v|, STOPSPEED) * sv_friction * dt      sv_friction = 4
      |v|  -= drop
  engine CBasePlayer::Accelerate( v, wishdir, wishspeed, accel ):
      flAddSpeed   = wishspeed - |v|                       (no accel above wishspeed)
      flAccelSpeed = min( flAddSpeed, accel * dt * wishspeed )
      v           += wishdir * flAccelSpeed                 sv_accelerate = 10 (air too)
  wishdir  = (fm, sm) / |(fm, sm)|                          (engine normalizes wishvel)
  wishspeed= maxspeed * min(|(fm, sm)|, 1)                 (engine clamps to MaxSpeed)
  then     p += v * dt                                     (v is the backward-difference
                                                            velocity, so this is exact)

SCALE — measured, not assumed (see STATUS.md 2026-09-30):
  * DT_BasePlayer.m_flMaxspeed (networked, per tick) is 240.0 for 100% of the
    soldier (cls=3) rows and 240*(1-4/66)=225 is exactly the grounded speed
    plateau observed in the data (mode 230). So maxspeed=240 hu/s, and the
    friction/stopspeed/accel constants above are in these units. maxspeed is
    supplied by the class one-hot via MAXV1, same as every other trainer.

REPARAM (why head/10 is an exact champion re-encoding):
  with (fm,sm) = accel_head_out / 10, wishspeed = maxspeed*|accel|/10, so
  accel*dt*wishspeed = 10*dt*maxspeed*|accel|/10 = maxspeed*|accel|*dt, which is
  exactly the velocity delta the accel-space champion applied. The only
  differences are the two clamps the champion ignored: wishspeed <= maxspeed
  (|accel| <= 10, the physical input-accel ceiling) and flAddSpeed <= 0
  (cannot accelerate when already at/above the wishspeed).

FRAME CAVEAT: there is no view angle in these demos (m_angEyeAngles lives in
DT_TFLocalPlayerExclusive / DT_TFNonLocalPlayerExclusive, which SourceTV never
populates for remote players - the DB column eyey is a single frozen spawn value
per entity). So the "ego@T frame" is really the VELOCITY-heading frame from the
build_ds_all fallback rule, and (fm,sm) live in that frame. No yaw rotation and
no dyaw head: there is no ground truth to supervise one with.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

DT = 1.0 / 66.0
HB = 24.0
ACCEL = 10.0            # sv_accelerate; sv_airaccelerate assumed equal (documented)
FRICTION = 4.0          # sv_friction
STOPSPEED = 100.0       # sv_stopspeed
REPARAM = 10.0          # accel_head_out = (fm, sm) * REPARAM
MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
MAXV1T = torch.from_numpy(MAXV1)
# difficulty weights w(t) = (t/66)^1.3718 (measured coast-error fit, R2=0.998)
W_T = torch.from_numpy(np.maximum((np.arange(1, 67) / 66.0) ** 1.3718, 0.02).astype(np.float32))

# Version-proof sqrt: bare .sqrt() at exactly 0 gives NaN gradients on some
# torch builds (measured 2.4.1; the B champion trained under 2.12). Standing
# players (v=0) and missing threats (vth=0, rel=0) hit exact zero constantly,
# and 0*NaN=NaN poisons through clamps and present-gates alike. Every length
# in a backward path goes through slen(); forward values change by <1e-6.
_LEN_EPS = 1e-12


def slen(v, dim=-1):
    """sqrt(sum(v^2) + eps): backward-safe vector length."""
    return torch.sqrt((v ** 2).sum(dim=dim, keepdim=True) + _LEN_EPS)


class Net(nn.Module):
    """GRU(20, 96) trunk (14 base + 6 threat feats), head -> 2 = (fm, sm) via tanh.

    Input dim stays 20 so gru_threatsimB.pt warm-starts directly; the head is the
    champion's head rescaled by 1/REPARAM (see module docstring).
    """

    def __init__(self, h=96, n_in=20, n_out=2):
        super().__init__()
        self.gru = nn.GRU(n_in, h, batch_first=True)
        self.head = nn.Linear(h, n_out)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])

    def load_reparam(self, ckpt, reparam=REPARAM):
        """Warm start from an accel-space checkpoint: trunk copied, head/REPARAM.

        A head copied verbatim would ask the engine for wishspeed = 10x too small
        (a near-inert model whose Accelerate branch is dead => no gradient), so the
        rescale is what makes epoch 0 the champion's own rollout.
        """
        sd = torch.load(ckpt, map_location='cpu')
        out = {}
        for k, v in sd.items():
            out[k] = v / reparam if k.startswith('head.') else v
        self.load_state_dict(out)
        return self


def friction(v, gr):
    """sv_friction, ground only. Identical to the champion's helper; accepts
    (B,2) step state or (B,T,2) sequences (gr broadcasts on the leading dims)."""
    g = gr
    while g.dim() < v.dim():
        g = g.unsqueeze(1)
    sp = slen(v)
    drop = torch.clamp(sp, min=STOPSPEED) * FRICTION * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (g * scale + (1.0 - g))


def accelerate(v, wh, w, gr, addspeed=True, rule='dot'):
    """Friction first, then CGameMovement::Accelerate. wh unit, w = wishspeed.

    rule='dot' (CORRECT Source, CGameMovement::Accelerate): the add-speed check
        projects the velocity ONTO THE WISHDIR, it does NOT use |v|:
            currentspeed = v . wh
            addspeed = wishspeed - currentspeed;  if <= 0: no accel
            dv = min(addspeed, accel*dt*wishspeed)
        Consequence: a player at the speed cap strafing sideways gets FULL
        lateral accel (v.wh ~= 0), and an angled push delivers a large RADIAL
        component. This is exactly the +500..1500 hu/s^2 "excess" radial accel
        the labels show at the plateau - it is legal engine behavior, and the
        earlier magnitude-based rule ('mag') wrongly forbade it (-16pp @300ms).
    rule='mag': the WRONG version (|v| instead of v.wh). Kept for ablation only.
    addspeed=False drops the early-out entirely (data-calibrated fallback).
    """
    v = friction(v, gr)
    if not addspeed:
        return v + wh * (ACCEL * DT * w)
    if rule == 'dot':
        cur = (v * wh).sum(-1, keepdim=True)
    else:
        cur = slen(v)
    add = torch.clamp(w - cur, min=0.0)
    dv = torch.minimum(add, ACCEL * DT * w)
    return v + wh * dv


def policy_out(head_out, mode, cap=REPARAM):
    """Map the raw head output to the (fm, sm) usercmd pair.

    mode='radial' (default, used everywhere): direction preserved, magnitude
        min(|out|, cap). This is the identity on the ball |out|<=cap, so a head
        warm-started from the accel-space champion (rescaled by 1/cap) reproduces
        the champion rollout exactly, and it is a hard bound on the output.
    mode='tanh': per-component tanh, i.e. squashing. MEASURED to be a bad
        parameterization for this head: tanh(x/10) is 2% low at the median
        accel but 19% low at p90 and 30% low past the cap, and that magnitude
        error alone cost 16pp of hit-rate at 300ms in the reparam check.
    mode='linear': no bound at all (diagnostic only).
    """
    if mode == 'tanh':
        return torch.tanh(head_out)
    if mode == 'linear':
        return head_out
    mag = slen(head_out, dim=1)
    return head_out * torch.clamp(cap / mag.clamp_min(1e-6), max=1.0)


def rollout(net, xb, s0, feedback=True, threat_sim=True, mode='radial', cap=REPARAM,
            addspeed=True, wscale=1.0, wcap=None, rule='dot'):
    """66-step differentiable rollout. Returns (B, 66, 2) positions in ego@T frame.

    xb   (B,66,20) warmup window (14 base + 6 threat)
    s0   (B,5)     threat sim state at T: relpos ego(2), vel ego(2), present(1)
    feedback  feed the previous step's throttle |(fm,sm)| into the dyn slot that
              used to hold the (never-existing) dyaw feedback.
    exact   diagnostic: force the add-speed branch to hand the engine exactly the
            accel-space champion's dv (W = |v| + dv_champ), i.e. run the champion
            through the engine code path. Must reproduce the champion's hit-rate;
            if it does not, the physics module is wrong, not the parameterization.
    """
    mv = (xb[:, -1:, 5:14] * MAXV1T).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    thr = thr_frozen
    mvs = mv.squeeze(1)
    # wishspeed ceiling: the engine's own per-tick MaxSpeed where available.
    # 5.5% of the cls==3 rows are actually heavies (mspd 400/384) and they are
    # exactly the high-speed population whose accel a 240 cap forbids.
    wcapv = mvs if wcap is None else wcap.reshape(-1, 1).to(mvs.dtype)
    traj = []
    for _ in range(66):
        raw = net.head(hh[-1])
        if mode == 'accel':
            # the champion's own line, through this harness: verifies friction,
            # step order, threat sim and GRU feedback against the ledger numbers
            v = friction(v, gr) + raw * mvs * DT
            mag = slen(raw, dim=1)
            fb = torch.zeros_like(mag)
        else:
            o = policy_out(raw, mode, cap)
            mag = slen(o, dim=1)
            wh = o / mag.clamp_min(1e-6)
            tau = mag if mode == 'linear' else torch.clamp(mag, max=1.0)
            v = accelerate(v, wh, wcapv * wscale * tau, gr, addspeed=addspeed, rule=rule)
            fb = tau
        p = p + v * DT
        if threat_sim:
            q = q + vth * DT
            rel = q - p
            dist = slen(rel, dim=1).clamp_min(1e-6)
            ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
            ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
            infov = (ang.abs() < (np.pi / 4)).float()
            appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
            # NOTE: appr is clamped BEFORE dividing. torch.where does NOT protect
            # division-by-zero in backward: dist/0 in the dead branch still
            # yields 0/0=NaN grads (measured). appr<=50 takes the ones branch,
            # so clamping the denominator changes no forward value.
            tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                              torch.ones_like(dist))
            live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
            thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
        sp = slen(v) / mvs
        if not feedback:
            fb = torch.zeros_like(sp)
        fb = fb.unsqueeze(1)
        dyn = torch.cat([(v / mvs).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), fb, thr], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def vel_loss(traj, Hb, Vb, lam=0.1, norm=240.0):
    """Velocity (inertia) supervision: per-tick local gradients.

    Position loss backprops through 66 integrations (ill-conditioned: early
    velocity errors explode into late position errors). Velocity targets are
    FREE (backward difference of the dense displacements) and give each step a
    direct, well-conditioned signal: the policy's accel IS a velocity delta.
    Small weight: hit-rate (position) stays the metric; this is the stabilizer.
    v_pred[h] recovered exactly from traj (p += v*DT): v = diff(p)/DT.
    Needs both endpoints valid (h=0 needs only tgt[0]; origin is exact).
    """
    tgt = torch.nan_to_num(torch.from_numpy(np.asarray(Hb)).float(), nan=0.0)
    m = torch.from_numpy(np.asarray(Vb)).float()
    mv = torch.cat([m[:, 0:1], m[:, 1:] * m[:, :-1]], dim=1).unsqueeze(-1)
    vp = torch.cat([traj[:, 0:1], traj[:, 1:] - traj[:, :-1]], dim=1) / DT
    vt = torch.cat([tgt[:, 0:1], tgt[:, 1:] - tgt[:, :-1]], dim=1) / DT
    per = ((((vp - vt) / norm) ** 2).sum(-1, keepdim=True)
           * W_T.view(1, -1, 1) * mv)
    return lam * (per.sum(dim=1) / mv.sum(dim=1).clamp_min(1)).mean()


def dense_loss(net, xb, s0, Hb, Vb, **kw):
    """Champion's hitbox loss: log-inside the 24hu AABB, quadratic outside, w(t).

    tgt is nan_to_num'd: invalid ticks carry NaN targets with mask 0, and
    NaN*0=NaN would poison the whole batch otherwise (short-series data made
    invalid ticks common; this latent bug killed the first forces run).
    """
    traj = rollout(net, xb, s0, **kw)
    tgt = torch.nan_to_num(torch.from_numpy(np.asarray(Hb)).float(), nan=0.0)
    m = torch.from_numpy(np.asarray(Vb)).unsqueeze(-1).float()
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * W_T.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean()


# ---------------------------------------------------------------------------
# KNOWN-FORCES ROLLOUT (accel policy): physics settles the tick, input lands last.
# Per step, in order:
#   1. vf = friction(v)                       (known)
#   2. vm = vf + ext_impulse                   (known external force, fixed scale)
#   3. p += vm * DT                            (tick resolved - input not involved)
#   4. v = vm + a_policy * DT                  (network's input accel, bitter end)
# The policy NEVER interacts with friction/impulses inside a step, so credit
# assignment is clean: position error after step 4 decomposes into (physics
# error, fixed) + (policy error, learned). Knockback samples become trainable;
# the impulse-cap backstop stays only for UNKNOWN forces (has=0).
# net: GRU(23,96) + head(96,2); warmup X is 23-dim (KBW fills the ext slots).
# ---------------------------------------------------------------------------

def load_forces(netX, ckpt20):
    """Warm start a wider accel net (23/29-dim) from a 20-dim checkpoint.

    New input columns zero-initialized (epoch 0 == 20-dim behavior with kb=0);
    everything else copied. Generic over target width.
    """
    sd = torch.load(ckpt20, map_location='cpu')
    out = {}
    for k, v in sd.items():
        if k == 'gru.weight_ih_l0':
            w = torch.zeros_like(netX.gru.weight_ih_l0)
            w[:, 0:v.shape[1]] = v
            out[k] = w
        else:
            out[k] = v
    netX.load_state_dict(out)
    return netX


# Extra input layout (after the 20 base dims): 3 knock [kx,ky,dmg] + 6 flags
# [fired_near, fired_mine, hurt_me, hurt_near, hp_self, hp_thr] = 9 dims (29 total).
NEX = 9
NKB = 3


def rollout_forces(net, xb, s0, kb, wcap=None, k_imp=0.0, threat_sim=True,
                   collide=None):
    """66-step known-forces rollout. Returns (B,66,2) positions in ego@T frame.

    xb  (B,66,20+NEX): 14 base + 6 threat + 3 warmup-force slots (KBW)
        + 6 flag/health slots (TSF). NEX=9.
    s0  (B,5): threat sim state at T
    kb  (B,66,4): [kx, ky, dmg, has] future forces per step (build_knock.py)
    wcap (B,1) or None: wishspeed ceiling (unused by the accel head, kept for
        signature symmetry; the cap backstop lives in the loss weights)
    k_imp: damage -> Δv scale (build_knock K_fit; 0 = feature-only)
    collide(p_new, p_old, v, aux) or None: SECOND physics pass (walls). Takes
        the integrated position + pre-integration velocity, returns
        (p_resolved, v_slid). None = identity (no map yet). The wall grid
        plugs in here without touching the rollout; aux carries per-sample
        map id so one function serves all arenas.
    """
    mv = (xb[:, -1:, 5:14] * MAXV1T).sum(-1, keepdim=True)
    mvs = mv.squeeze(1)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mvs
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    thr = thr_frozen
    traj = []
    for t in range(66):
        raw = net.head(hh[-1])
        a = raw * mvs
        vf = friction(v, gr)
        vm = vf + kb[:, t, 0:2] * (kb[:, t, 2:3] * k_imp)
        pn = p + vm * DT
        if collide is not None:
            pn, vm = collide(pn, p, vm, None)
        p = pn
        v = vm + a * DT
        if threat_sim:
            q = q + vth * DT
            rel = q - p
            dist = slen(rel, dim=1).clamp_min(1e-6)
            ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
            ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
            infov = (ang.abs() < (np.pi / 4)).float()
            appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
            # NOTE: appr is clamped BEFORE dividing. torch.where does NOT protect
            # division-by-zero in backward: dist/0 in the dead branch still
            # yields 0/0=NaN grads (measured). appr<=50 takes the ones branch,
            # so clamping the denominator changes no forward value.
            tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                              torch.ones_like(dist))
            live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
            thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
        sp = slen(v) / mvs
        # per-step extra: current knock force + frozen flag/health state.
        # Future flags are unknowable (no future info); freezing last-known is
        # the honest runtime match. Knockback IS teacher-forced (every-tick
        # re-predict regime, same as kb physics above).
        kbh = kb[:, t, 0:3]
        feat_kb = torch.cat([kbh[:, 0:2], (kbh[:, 2:3] / 100.0)], dim=-1)
        dyn = torch.cat([(v / mvs).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                         thr, feat_kb.unsqueeze(1),
                         xb[:, -1:, 20 + NKB:20 + NEX]], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def forces_loss(net, xb, s0, kb, Hb, Vb, w2=None, k_imp=0.0, threat_sim=True,
                collide=None, lam_vel=0.1):
    """Dense hitbox loss through the known-forces rollout.

    w2 (B,66,1) or None: per-tick trainability weights (1 = trainable, 0 =
    trash). Trash = teleports/class/loadout gaps (state breaks), NaNs, and
    impulse-cap violations with UNKNOWN force. Knockback WITH known force
    stays trainable - that is the whole point. Multiplies the validity mask.
    lam_vel: velocity (inertia) stabilizer weight (see vel_loss).
    """
    traj = rollout_forces(net, xb, s0, kb, k_imp=k_imp, threat_sim=threat_sim,
                          collide=collide)
    tgt = torch.nan_to_num(torch.from_numpy(np.asarray(Hb)).float(), nan=0.0)
    m = torch.from_numpy(np.asarray(Vb)).unsqueeze(-1).float()
    if w2 is not None:
        m = m * torch.from_numpy(np.asarray(w2)).float()
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * W_T.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean() + vel_loss(traj, Hb, Vb, lam=lam_vel)


# ---------------------------------------------------------------------------
# FULL PACKET DESIGN (generation C): head -> 3 = (fm, sm, dyaw) with yaw rotation.
# Unblocked 2026-10-01 when the extractor eye() bug was fixed (m_angEyeAngles IS
# sent for remote players; the reader was shadowing it with a stale baseline).
#
# Per step: (fm,sm) lives in the CURRENT predicted view frame; psi accumulates
# the predicted yaw deltas; the wish vector is rotated by psi into the ego@T
# frame; engine Accelerate (accel ceiling, addspeed OFF - measured) applies it.
# Pitch is skipped: TF2 wishdir uses yaw only.
#
# INPUT SUPERVISION: position loss alone collapses the bounded policy into
# coasting (measured 59.3 -> 44.0 @300ms with loss improving). So every
# predictable input quantity gets its own per-tick term, all under the same
# difficulty weights w(t):
#   throttle magnitude -> QUADRATIC penalty on (tau - tau_hat), gated on
#     observability (inputs are unobservable at steady state: near-zero accel
#     means the magnitude could be anything, so those ticks teach nothing)
#   wish direction     -> LOG reward on alignment, -log((1+cos)/2): 0 when the
#     decoded wishdir matches the observed accel direction, steep when opposed.
#     Direction is observable whenever the magnitude is.
#   yaw delta          -> QUADRATIC penalty vs the true eye delta, gated on eye
#     freshness.
# The throttle MAGNITUDE is the only thing never directly supervised at steady
# state - it belongs to the engine (wishspeed = maxspeed * throttle).
# ---------------------------------------------------------------------------

def load_packet(net22, ckpt20, reparam=REPARAM):
    """Warm start a 22-dim packet net from a 20-dim accel checkpoint.

    GRU input columns 20:22 zero-initialized (epoch 0 == 20-dim behavior);
    head rows 0:2 = champion/REPARAM, BUT row 1 NEGATED: the packet sm is
    D-positive (right, usercmd sidemove) while the ego frame's +y is LEFT, so
    the champion's ay (left-positive) must flip sign to become a D-positive sm.
    Without the flip the init rollout steers mirrored (measured 5x position
    error). Head rows 2: (dyaw, pitch) zero-init (no turning at init).
    """
    sd = torch.load(ckpt20, map_location='cpu')
    out = {}
    for k, v in sd.items():
        if k == 'gru.weight_ih_l0':
            w = torch.zeros_like(net22.gru.weight_ih_l0)
            w[:, 0:v.shape[1]] = v
            out[k] = w
        elif k.startswith('head.weight'):
            w = torch.zeros_like(net22.head.weight)
            w[0] = v[0] / reparam
            w[1] = -v[1] / reparam
            out[k] = w
        elif k.startswith('head.bias'):
            b = torch.zeros_like(net22.head.bias)
            b[0] = v[0] / reparam
            b[1] = -v[1] / reparam
            out[k] = b
        else:
            out[k] = v
    net22.load_state_dict(out)
    return net22


def rollout_packet(net, xb, s0, wcap, yaw_scale_deg, addspeed=False,
                   threat_sim=True, rule='dot', pitch_scale_deg=8.0,
                   psi_force=None):
    """66-step packet rollout with yaw rotation.

    xb  (B,66,22): 20 base feats + 2 command slots (zeros in the warmup window;
        filled with the predicted (fm,sm) during rollout)
    s0  (B,5): threat sim state at T
    wcap (B,1): per-sample wishspeed ceiling (m_flMaxspeed at T, runtime-known)
    Head outputs (fm, sm, dyaw[, dpitch]); pitch does not affect the 2D rollout
    (TF2 wishdir uses yaw only) but is predicted + lightly supervised as part of
    the packet. Returns traj, tau, what, dyaw_deg, dpitch_deg.
    psi_force (B,66) degrees or None: override the accumulated yaw offset
        (diagnostic: teacher-forced true psi = upper bound on yaw modeling;
        if it beats predicted psi clearly, yaw prediction is the bottleneck).
    """
    mv = (xb[:, -1:, 5:14] * MAXV1T).sum(-1, keepdim=True)
    mvs = mv.squeeze(1)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mvs
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    thr = thr_frozen
    psi = torch.zeros_like(gr)
    traj, taus, whats, dyaws, pits = [], [], [], [], []
    for _t in range(66):
        raw = net.head(hh[-1])
        o = policy_out(raw[:, 0:2], 'radial', REPARAM)
        fm, sm = o[:, 0:1], o[:, 1:2]
        mag = slen(o, dim=1)
        tau = torch.clamp(mag, max=1.0)
        dy = torch.tanh(raw[:, 2:3]) * yaw_scale_deg
        dp = (torch.tanh(raw[:, 3:4]) * pitch_scale_deg
              if raw.shape[1] > 3 else torch.zeros_like(dy))
        pr = psi * (np.pi / 180.0)
        c, s = torch.cos(pr), torch.sin(pr)
        # ego@T wishdir = fm*forward(psi) + sm*right(psi),
        # forward=(c,s), right=(s,-c) for D-positive sm (Source: +yaw = left,
        # ego +y = left, so the D key drives -y).
        wx = fm * c + sm * s
        wy = fm * s - sm * c
        what = torch.cat([wx, wy], dim=1) / mag.clamp_min(1e-6)
        v = accelerate(v, what, wcap * tau, gr, addspeed=addspeed, rule=rule)
        p = p + v * DT
        if psi_force is not None:
            # psi_force[:, t] = accumulated TRUE yaw through tick T+t-1, i.e.
            # cumsum(true_dyaw) shifted right by one with 0 prepended: the exact
            # rotation the engine would use at this step.
            psi = psi_force[:, _t:_t + 1]
        else:
            psi = psi + dy
        if threat_sim:
            q = q + vth * DT
            rel = q - p
            dist = slen(rel, dim=1).clamp_min(1e-6)
            ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
            ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
            infov = (ang.abs() < (np.pi / 4)).float()
            appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
            # NOTE: appr is clamped BEFORE dividing. torch.where does NOT protect
            # division-by-zero in backward: dist/0 in the dead branch still
            # yields 0/0=NaN grads (measured). appr<=50 takes the ones branch,
            # so clamping the denominator changes no forward value.
            tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                              torch.ones_like(dist))
            live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
            thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
        sp = slen(v) / mvs
        dy_rad = dy * (np.pi / 180.0)
        # LAYOUT (warm-start critical): first 20 dims EXACTLY match the 20-dim
        # net ([v,sp,gr,slot3,thr] ++ static one-hot); the 2 command slots go
        # LAST so old weight columns stay aligned. slot3 carries dy_rad (it was
        # always-zero during accel training, and dy=0 at init, so consistent).
        dyn = torch.cat([(v / mvs).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), dy_rad.unsqueeze(1), thr], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static, o.unsqueeze(1)], dim=-1), hh)
        traj.append(p)
        taus.append(tau)
        whats.append(what)
        dyaws.append(dy)
        pits.append(dp)
    return (torch.stack(traj, dim=1), torch.stack(taus, dim=1),
            torch.stack(whats, dim=1), torch.stack(dyaws, dim=1),
            torch.stack(pits, dim=1))


def packet_labels(vel, gr, maxspeed, accel_floor=0.5, accel_full=2.0):
    """Inverse-dynamics input labels from TRUE velocities (eye-frame C data).

    vel (B,67,2): v_0 from the warmup window, v_1..v_66 = Hd/DT. gr (B,1).
    maxspeed (B,1): per-sample cap. Returns tau_hat (B,66,1), what_hat (B,66,2),
    obs (B,66,1) observability weight ramping 0->1 between accel_floor and
    accel_full (label units, i.e. |A|/MAXV with MAXV=240: p50 is 2.37, so 0.5 is
    weak/noisy and 2.0 is a solid observable input).

    The throttle label uses the DOT-PRODUCT add-speed rule (CGameMovement):
    W must satisfy min(W - v.what, 0.1515*W) = |dv|, i.e.
    W = max(|dv|/0.1515, v.what + |dv|). A magnitude-only rule (|v| instead of
    v.what) would starve the throttle wherever the player is fast; the dot rule
    is what makes angled pushes legal and matches the labels.
    """
    vf = friction(vel[:, :-1].float(), gr).double()
    dv = vel.double()[:, 1:] - vf
    n = slen(dv, dim=-1)
    what = dv / n.clamp_min(1e-9)
    v0 = vel.double()[:, :-1]
    along = (v0 * what).sum(-1, keepdim=True)
    ms = maxspeed.double()
    while ms.dim() < n.dim():
        ms = ms.unsqueeze(1)
    w_need = torch.maximum(n / (ACCEL * DT), along + n)
    tau_hat = (w_need / ms).clamp(0.0, 1.5)
    a_mag = n / DT / ms
    obs = ((a_mag - accel_floor) / (accel_full - accel_floor)).clamp(0.0, 1.0)
    return tau_hat.float(), what.float(), obs.float()


def input_loss(tau, what, dyaw_deg, tau_hat, what_hat, dyaw_hat,
               obs, fresh, valid, yaw_scale_deg,
               lam_thr=0.3, lam_dir=0.3, lam_yaw=0.1,
               dpitch_deg=None, dpitch_hat=None, pitch_scale_deg=8.0,
               lam_pit=0.02):
    """Per-tick input supervision, all under the same difficulty weights w(t).

    tau/what/dyaw_deg: predicted (B,66,1)/(B,66,2)/(B,66,1,deg).
    tau_hat/what_hat: inverse-dynamics labels. dyaw_hat: true eye deltas (deg).
    obs: observability (0 at steady state). fresh: eye-fresh & tick-valid.
    valid: tick validity. Pitch (optional, low weight - it does not affect the
    2D rollout) is supervised the same way when dpitch_deg/dpitch_hat given.
    Returns scalar.
    """
    l_thr = obs * (tau.nan_to_num(0.0) - tau_hat.nan_to_num(0.0)) ** 2
    cos = (what.nan_to_num(0.0) * what_hat.nan_to_num(0.0)).sum(-1, keepdim=True)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    l_dir = obs * (-torch.log((1.0 + cos) / 2.0 + 1e-6))
    l_yaw = fresh * ((dyaw_deg - dyaw_hat.nan_to_num(0.0)) / yaw_scale_deg) ** 2
    per = (l_thr * lam_thr + l_dir * lam_dir + l_yaw * lam_yaw)
    if dpitch_deg is not None and dpitch_hat is not None:
        l_pit = fresh * ((dpitch_deg - dpitch_hat.nan_to_num(0.0)) / pitch_scale_deg) ** 2
        per = per + l_pit * lam_pit
    per = per * W_T.view(1, -1, 1) * valid
    return (per.sum(dim=1) / valid.sum(dim=1).clamp_min(1)).mean()
