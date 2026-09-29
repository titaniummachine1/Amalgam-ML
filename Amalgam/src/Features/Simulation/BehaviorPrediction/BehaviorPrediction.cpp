#include "BehaviorPrediction.h"
#include "weights_soldier_dense66.h"

#include <cmath>
#include <cstring>
#include <unordered_map>

// Self-contained GRU + Linear forward, int8 weights dequantized once to fp32.
// MUST match training exactly (ml/train_dense66.py, ml/build_ds_all.py):
// same feature order, same ego-frame math, same backward-diff velocity,
// same eye-yaw fallback rule, same per-class MAXV table. Any drift here
// silently invalidates the trained weights.
//
// VERIFICATION STATUS (2026-09-29, see ml/verify_gru.py):
// - gate math exact vs torch: 2.4e-07 single-step.
// - int8 weights: hit-rate preserved (97.0/82.3/58.6/28.1/12.2 vs fp32
//   97.0/82.5/58.2/27.2/12.4 — within noise, 500ms even +0.9).
// - recurrent fp32 rounding divergence (~0.3 hidden / ~2 output units worst case
//   after 66 steps) is inherent to ANY reimplementation (torch CUDA vs CPU
//   diverges the same way); magnitude matches the proven-safe int8 perturbation.
//   Final proof requires in-game trajectory logging — flag any systematic miss.

namespace
{
	constexpr float kTickDt = 1.f / 66.f;
	constexpr int kWarmup = 66;   // 1s warmup tail consumed from the 2s ring
	constexpr int kRing = 132;    // 2s ring per player
	constexpr int kMaxRoll = 132;
	constexpr int kFeat = 14;     // 5 base + 9 class one-hot
	constexpr int kHid = 96;

	static_assert(BehaviorPolicy::kNin == kFeat, "weights/header arch mismatch");
	static_assert(BehaviorPolicy::kH == kHid, "weights/header arch mismatch");

	// MUST equal ml MAXV_CLS (training normalization — do not "improve" to SDK values)
	float ClassMaxSpeed(int iClass)
	{
		switch (iClass)
		{
		case 1: return 400.f;
		case 2: return 300.f;
		case 3: return 240.f;
		case 4: return 280.f;
		case 5: return 320.f;
		case 6: return 230.f;
		case 7: return 300.f;
		case 8: return 300.f;
		case 9: return 300.f;
		default: return 300.f;
		}
	}

	inline float Sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
	inline float WrapRad(float a)
	{
		while (a > 3.14159265f) a -= 6.2831853f;
		while (a < -3.14159265f) a += 6.2831853f;
		return a;
	}
}

struct CBehaviorPrediction::TargetState
{
	float m_flPos[kRing][3] = {};
	float m_flEyeYaw[kRing] = {}; // degrees, as training saw them
	float m_flGrounded[kRing] = {};
	int m_iClass = 0;
	int m_iHead = 0;
	int m_iCount = 0;
	float m_flHidden[kHid] = {}; // reserved: incremental hidden update (phase 3 perf)
};

void CBehaviorPrediction::Reset(int iEntIndex)
{
	if (iEntIndex < 0)
	{
		m_mTargets.clear();
		return;
	}
	m_mTargets.erase(iEntIndex);
}

void CBehaviorPrediction::EnsureWeights()
{
	if (m_bWeightsReady)
		return;
	const auto deq = [](const int8_t* q, float* f, size_t n, float s)
	{
		for (size_t i = 0; i < n; i++)
			f[i] = float(q[i]) * s;
	};
	deq(BehaviorPolicy::kW_gru_weight_ih_l0, m_Wih, sizeof(m_Wih) / sizeof(float), BehaviorPolicy::kScale_gru_weight_ih_l0);
	deq(BehaviorPolicy::kW_gru_weight_hh_l0, m_Whh, sizeof(m_Whh) / sizeof(float), BehaviorPolicy::kScale_gru_weight_hh_l0);
	deq(BehaviorPolicy::kW_gru_bias_ih_l0, m_Bih, sizeof(m_Bih) / sizeof(float), BehaviorPolicy::kScale_gru_bias_ih_l0);
	deq(BehaviorPolicy::kW_gru_bias_hh_l0, m_Bhh, sizeof(m_Bhh) / sizeof(float), BehaviorPolicy::kScale_gru_bias_hh_l0);
	deq(BehaviorPolicy::kW_head_weight, m_HeadW, sizeof(m_HeadW) / sizeof(float), BehaviorPolicy::kScale_head_weight);
	deq(BehaviorPolicy::kW_head_bias, m_HeadB, sizeof(m_HeadB) / sizeof(float), BehaviorPolicy::kScale_head_bias);
	m_bWeightsReady = true;
}

void CBehaviorPrediction::PushState(int iEntIndex, const Vec3& vOrigin, float flEyeYawDeg, bool bGrounded, int iClass)
{
	auto& st = m_mTargets[iEntIndex]; // default-constructs on first sight
	if (st.m_iCount > 0 && (st.m_iClass != iClass))
	{
		// class change invalidates history — hard reset, never hold stale state
		st = TargetState();
	}
	st.m_iClass = iClass;
	const int w = st.m_iHead % kRing;
	st.m_flPos[w][0] = vOrigin.x;
	st.m_flPos[w][1] = vOrigin.y;
	st.m_flPos[w][2] = vOrigin.z;
	st.m_flEyeYaw[w] = flEyeYawDeg;
	st.m_flGrounded[w] = bGrounded ? 1.f : 0.f;
	st.m_iHead++;
	if (st.m_iCount < kRing)
		st.m_iCount++;
}

// Build the 66x14 warmup feature block from the ring tail, exactly as training.
static int BuildWarmup(const CBehaviorPrediction::TargetState& st, float flMaxV, float ouFeat[kWarmup][kFeat], float* pYawRad)
{
	if (st.m_iCount < 22) // ablation: 0.33s minimum viable warmup
		return 0;
	const int n = st.m_iCount < kWarmup ? st.m_iCount : kWarmup;
	// gather tail in order (oldest..newest), index 0 = oldest
	float ox[kWarmup], oy[kWarmup], eye[kWarmup], gr[kWarmup];
	for (int i = 0; i < n; i++)
	{
		const int idx = (st.m_iHead - n + i) % kRing;
		const int ii = idx < 0 ? idx + kRing : idx;
		ox[i] = st.m_flPos[ii][0];
		oy[i] = st.m_flPos[ii][1];
		eye[i] = st.m_flEyeYaw[ii];
		gr[i] = st.m_flGrounded[ii];
	}
	// backward-diff velocity (hu/s), mirroring training (demo networked vel was stale)
	float vx[kWarmup], vy[kWarmup];
	vx[0] = 0.f;
	vy[0] = 0.f;
	for (int i = 1; i < n; i++)
	{
		vx[i] = (ox[i] - ox[i - 1]) / (1.f / 66.f);
		vy[i] = (oy[i] - oy[i - 1]) / (1.f / 66.f);
	}
	// ego frame at newest tick, with velocity-heading fallback (training rule)
	float flYaw = eye[n - 1];
	if (!(flYaw == flYaw))
		flYaw = 0.f;
	{
		float m = 0.f;
		for (int i = 0; i < n; i++)
			m += eye[i];
		m /= n;
		float va = 0.f;
		for (int i = 0; i < n; i++)
			va += (eye[i] - m) * (eye[i] - m);
		va = sqrtf(va / n);
		const float sp = sqrtf(vx[n - 1] * vx[n - 1] + vy[n - 1] * vy[n - 1]);
		if (va < 0.5f && sp > 50.f)
			flYaw = atan2f(vy[n - 1], vx[n - 1]) * 57.29578f;
	}
	const float a = flYaw * 0.01745329f;
	const float c = cosf(-a), s = sinf(-a);
	*pYawRad = a;
	for (int i = 0; i < n; i++)
	{
		const float evx = (c * vx[i] - s * vy[i]) / flMaxV;
		const float evy = (s * vx[i] + c * vy[i]) / flMaxV;
		const float esp = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]) / flMaxV;
		float dy = 0.f;
		if (i > 0)
		{
			dy = (eye[i] - eye[i - 1]) * 0.01745329f;
			while (dy > 3.14159265f) dy -= 6.2831853f;
			while (dy < -3.14159265f) dy += 6.2831853f;
		}
		ouFeat[i][0] = evx;
		ouFeat[i][1] = evy;
		ouFeat[i][2] = esp;
		ouFeat[i][3] = gr[i];
		ouFeat[i][4] = dy;
		for (int k = 0; k < 9; k++)
			ouFeat[i][5 + k] = (k == st.m_iClass - 1) ? 1.f : 0.f;
	}
	return n;
}

int CBehaviorPrediction::Predict(int iEntIndex, int iTicks, BehaviorPrediction_t* pOut, int iMaxOut)
{
	EnsureWeights();
	auto it = m_mTargets.find(iEntIndex);
	if (it == m_mTargets.end() || !pOut || iMaxOut <= 0)
		return 0;
	auto& st = it->second;
	const float flMaxV = ClassMaxSpeed(st.m_iClass);
	float feat[kWarmup][kFeat] = {};
	float flYawRad = 0.f;
	const int n = BuildWarmup(st, flMaxV, feat, &flYawRad);
	if (!n)
		return 0;

	// GRU warmup (PyTorch layout: gates order r, z, n; h' = (1-z)*n + z*h)
	float h[kHid] = {};
	for (int t = 0; t < n; t++)
	{
		const float* x = feat[t];
		float hn[kHid];
		for (int j = 0; j < kHid; j++)
		{
			float ir = m_Bih[j], iz = m_Bih[kHid + j], in_ = m_Bih[2 * kHid + j];
			float hr = m_Bhh[j], hz = m_Bhh[kHid + j], hn_ = m_Bhh[2 * kHid + j];
			for (int k = 0; k < kFeat; k++)
			{
				ir += x[k] * m_Wih[j * kFeat + k];
				iz += x[k] * m_Wih[(kHid + j) * kFeat + k];
				in_ += x[k] * m_Wih[(2 * kHid + j) * kFeat + k];
			}
			for (int k = 0; k < kHid; k++)
			{
				hr += h[k] * m_Whh[j * kHid + k];
				hz += h[k] * m_Whh[(kHid + j) * kHid + k];
				hn_ += h[k] * m_Whh[(2 * kHid + j) * kHid + k];
			}
			const float r = Sigmoid(ir + hr);
			const float z = Sigmoid(iz + hz);
			const float nn = tanhf(in_ + r * hn_);
			hn[j] = (1.f - z) * nn + z * h[j];
		}
		memcpy(h, hn, sizeof(h));
	}
	memcpy(st.m_flHidden, h, sizeof(h)); // seed for future incremental update

	// rollout: accel head -> integrate (ego hu), feedback mirrors training unroll
	const int nSteps = iTicks < iMaxOut ? iTicks : iMaxOut;
	const int nStepsC = nSteps > kMaxRoll ? kMaxRoll : nSteps;
	float v[2] = { feat[n - 1][0] * flMaxV, feat[n - 1][1] * flMaxV };
	float grLast = feat[n - 1][3];
	int written = 0;
	for (int s = 0; s < nStepsC; s++)
	{
		float acc[2] = { m_HeadB[0], m_HeadB[1] };
		for (int j = 0; j < kHid; j++)
		{
			acc[0] += h[j] * m_HeadW[0 * kHid + j];
			acc[1] += h[j] * m_HeadW[1 * kHid + j];
		}
		acc[0] *= flMaxV;
		acc[1] *= flMaxV; // ego hu/s^2
		v[0] += acc[0] * kTickDt;
		v[1] += acc[1] * kTickDt;
		pOut[written].m_vAccel.x = acc[0];
		pOut[written].m_vAccel.y = acc[1];
		pOut[written].m_vAccel.z = 0.f;
		pOut[written].m_flConfidence = 1.f; // value head not yet landed (phase 2)
		written++;

		const float sp = sqrtf(v[0] * v[0] + v[1] * v[1]) / flMaxV;
		float xin[kFeat] = { v[0] / flMaxV, v[1] / flMaxV, sp, grLast, 0.f,
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		for (int k = 0; k < 9; k++)
			xin[5 + k] = (k == st.m_iClass - 1) ? 1.f : 0.f;
		float hn2[kHid];
		for (int j = 0; j < kHid; j++)
		{
			float ir = m_Bih[j], iz = m_Bih[kHid + j], in_ = m_Bih[2 * kHid + j];
			float hr = m_Bhh[j], hz = m_Bhh[kHid + j], hn_ = m_Bhh[2 * kHid + j];
			for (int k = 0; k < kFeat; k++)
			{
				ir += xin[k] * m_Wih[j * kFeat + k];
				iz += xin[k] * m_Wih[(kHid + j) * kFeat + k];
				in_ += xin[k] * m_Wih[(2 * kHid + j) * kFeat + k];
			}
			for (int k = 0; k < kHid; k++)
			{
				hr += h[k] * m_Whh[j * kHid + k];
				hz += h[k] * m_Whh[(kHid + j) * kHid + k];
				hn_ += h[k] * m_Whh[(2 * kHid + j) * kHid + k];
			}
			const float r = Sigmoid(ir + hr);
			const float z = Sigmoid(iz + hz);
			const float nn = tanhf(in_ + r * hn_);
			hn2[j] = (1.f - z) * nn + z * h[j];
		}
		memcpy(h, hn2, sizeof(h));
	}
	return written;
}
