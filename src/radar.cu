#include <map>
#include <math_constants.h>

#include "../../include/GPUCodingInterface.h"
#include "../common/export_symbols.h"
#include "../common/utils_cuda.h"

// ======= std/C++ helpers we use =============================================
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <direct.h>
#include <errno.h>
#include <ctime>
#include <iostream>

// ======= ADDED (for 3 fixes) ================================================
#include <cstdint>
#include <limits>
// ============================================================================

// Debug flag: 1 = run small Eq.3.5 kernel, 0 = keep it off
#define EQ35_DEBUG 0

#define SYNTH_FROM_DETECTIONS 1

// === Ansys export switches (0 = off [default], 1 = on) =====================
#define EXPORT_ANSYS_AVX 0
#define EXPORT_ANSYS_ADC 0

// Max number of raw interaction points we want to put into the Synth debug buffer.
#define SYNTH_IP_DEBUG_MAX  20000

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static void PrintWorkingDir()
{
    char buf[1024];
    if (_getcwd(buf, sizeof(buf)) != nullptr) {
        std::printf("[GPU RSI] Working dir: %s\n", buf);
    } else {
        std::printf("[GPU RSI] Working dir: <_getcwd failed>\n");
    }
}

// ======= Smart pointer for CUDA device memory ===============================
template <typename T>
struct CudaDevicePtr {
    T* p = nullptr;
    CudaDevicePtr() = default;
    ~CudaDevicePtr() { if (p) cudaFree(p); }
    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;
    CudaDevicePtr(CudaDevicePtr&& o) noexcept : p(o.p) { o.p = nullptr; }
    CudaDevicePtr& operator=(CudaDevicePtr&& o) noexcept { std::swap(p, o.p); return *this; }
    T** get_addr() { return &p; }
    operator T*() const { return p; }
};

// ======= Beam LUT constants & storage =======================================
static constexpr int kBeamAzi  = 1801;
static constexpr int kBeamEle  = 1801;
static constexpr int kBeamTxRx = 16;

static float* d_beamLUT = nullptr;
static int    g_azSize  = kBeamAzi;
static int    g_elSize  = kBeamEle;

static const char* kBeamLUTBin =
    "C:/Project/Data/BeamLUT/beam_steering_dummy.bin";

// ======= Sensor constants for Eq. 3.5 synthesizer ===========================
struct SensorParams {
  float f0;
  float fs;
  float Tc;
  float dF;
  int   Ns;
  int   Nc;
  int   Ntx;
  int   Nrx;
  float c;
};
static SensorParams g_params = {
  76.5e9f, 25.0e6f, 40.96e-6f, 600.0e6f, 1024, 320, 4, 4, 299792458.0f
};

// MIMO phase table [tx][seq(8)]
static const float kMimoHost[4][8] = {
  {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
  {0.f, (float)M_PI, 0.f, (float)M_PI, 0.f, (float)M_PI, 0.f, (float)M_PI},
  {0.f, (float)M_PI*0.5f, (float)M_PI, (float)M_PI*1.5f, 0.f, (float)M_PI*0.5f, (float)M_PI, (float)M_PI*1.5f},
  {0.f, (float)M_PI*0.25f, (float)M_PI*0.5f, (float)M_PI*0.75f, (float)M_PI, (float)M_PI*1.25f, (float)M_PI*1.5f, (float)M_PI*1.75f}
};
__constant__ float d_mimo[4][8];

// ======= File outputs ========================================================
static const char* kRadarOutDir = "SimOutput\\DLL Cuda Outputs";

static void EnsureSimOutputDir() {
  if (_mkdir("SimOutput") == 0) {
    std::printf("[GPU RSI] Created SimOutput folder\n");
  } else if (errno != EEXIST) {
    std::printf("[GPU RSI] Warning: cannot create SimOutput (errno=%d)\n", errno);
  }
  if (_mkdir(kRadarOutDir) == 0) {
    std::printf("[GPU RSI] Created %s folder\n", kRadarOutDir);
  } else if (errno != EEXIST) {
    std::printf("[GPU RSI] Warning: cannot create %s (errno=%d)\n", kRadarOutDir, errno);
  }
}

static void MakeTimestamp(char* buf, int bufSize)
{
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    if (lt) {
        std::strftime(buf, bufSize, "%Y-%m-%d_%H-%M-%S", lt);
    } else {
        std::snprintf(buf, bufSize, "no_time");
    }
}

// --- ANSYS AVX streaming export ---------------------------------------------
static bool g_avxStreamInit   = false;
static char g_avxStreamName[256];
static long g_avxStreamNextId = 1;

static void ExportDetectionsToANSYS_AVX_GPU(const std::vector<tOutputPointC_R>& dets)
{
    if (dets.empty()) return;

    EnsureSimOutputDir();

    if (!g_avxStreamInit) {
        char ts[64];
        MakeTimestamp(ts, sizeof(ts));
        std::snprintf(g_avxStreamName, sizeof(g_avxStreamName),
              "SimOutput\\DLL Cuda Outputs\\Ansys_AVX_GPU_%s.avx", ts);
        FILE* f = std::fopen(g_avxStreamName, "w");
        if (!f) {
            std::printf("[GPU] ERROR: cannot open %s for write\n", g_avxStreamName);
            return;
        }
        std::fprintf(f, "/prep7\n");
        std::fprintf(f, "! Generated from GPU RSI detections (all frames). Units: meters.\n");
        std::fprintf(f, "! K, id, x, y, z\n");
        std::fclose(f);
        g_avxStreamInit = true;
        std::printf("[GPU] Streaming AVX file opened: %s\n", g_avxStreamName);
    }

    FILE* f = std::fopen(g_avxStreamName, "a");
    if (!f) {
        std::printf("[GPU] ERROR: cannot open %s for append\n", g_avxStreamName);
        return;
    }
    for (const auto& d : dets) {
        const double x = (double)d.tCoord_R.CoordCartOut_R.x;
        const double y = (double)d.tCoord_R.CoordCartOut_R.y;
        const double z = (double)d.tCoord_R.CoordCartOut_R.z;
        std::fprintf(f, "K,%ld,%.12f,%.12f,%.12f\n", g_avxStreamNextId++, x, y, z);
    }
    std::fclose(f);
}

struct SynthDetection {
  float r0_m;   float v0_ms;  float az_deg;  float el_deg;  float amp_lin;
};

// ======= stock data ==========================================================
struct RadarInitData {
  int nRangeBins;
  int nVelocityBins;
  int nPhiBins;
  int nDetectionsMax;
  int nEq35Max;
  unsigned int* nOutputPointCloudDevice;
  float* d_frame;
  struct SynthDetection* d_dets;
  float rangeMin, rangeMax;
  float velMin, velMax;
};

static std::map<int, RadarInitData> radarInitData;
static std::vector<tOutputPointC_R> g_allDetectionsGPU;
static bool g_avxWrittenOnce = false;

static bool g_ifBinInit  = false;
static char g_ifBinTS[64];
static char g_ifBinName[256];
static unsigned long long g_ifFrameCount = 0ULL;

static bool g_adcInit      = false;
static char g_adcTS[64];
static unsigned long long g_adcFrameId = 0ULL;

static constexpr int LOG_SENSOR_INDEX = 0;

static bool g_runInit = false;
static char g_runTS[64] = {0};

static const char* g_rsiDetFile  = "SimOutput\\DLL Cuda Outputs\\RSI_Detections.txt";
static const char* g_synthInFile = "SimOutput\\DLL Cuda Outputs\\RSI_SynthInput.txt";

static char g_metaFile[256]       = {0};
static char g_frameIndexFile[256] = {0};

static unsigned long long g_frameSeq      = 0ULL;
static unsigned long long g_binFrameIndex = 0ULL;

template <typename RC>
static auto TryGetTime_s_impl(const RC& rc, int) -> decltype(rc.time_s, double()) {
    return (double)rc.time_s;
}
template <typename RC>
static auto TryGetTime_s_impl(const RC& rc, long) -> decltype(rc.time, double()) {
    return (double)rc.time;
}
static double TryGetTime_s_impl(...) {
    return std::numeric_limits<double>::quiet_NaN();
}

static double GetBestTime_s(const user_RadarCube& radarCube)
{
    double t = TryGetTime_s_impl(radarCube, 0);
    if (!(t == t)) {
        const double frame_dt = (double)g_params.Nc * (double)g_params.Tc;
        t = (double)g_frameSeq * frame_dt;
    }
    return t;
}

static void InitRunFilesOnce()
{
    if (g_runInit) return;

    EnsureSimOutputDir();
    MakeTimestamp(g_runTS, sizeof(g_runTS));

    std::snprintf(g_ifBinName, sizeof(g_ifBinName),
              "SimOutput\\DLL Cuda Outputs\\RawSignalSynth_GPU_%s.bin", g_runTS);
    std::snprintf(g_metaFile, sizeof(g_metaFile),
              "SimOutput\\DLL Cuda Outputs\\RawSignalSynth_GPU_%s.meta.txt", g_runTS);
    std::snprintf(g_frameIndexFile, sizeof(g_frameIndexFile),
              "SimOutput\\DLL Cuda Outputs\\RawSignalSynth_GPU_%s_frameIndex.csv", g_runTS);

    {
        FILE* f = std::fopen(g_rsiDetFile, "wb");
        if (f) {
            std::fprintf(f,
              "TimeFired_s,FrameID,SensorIdx,DetIdx,Range_m,Azimuth_rad,Elevation_rad,Vel_mps,Power_dB\n");
            std::fclose(f);
        }
    }
    {
        FILE* f = std::fopen(g_synthInFile, "wb");
        if (f) {
            std::fprintf(f,
              "TimeFired_s,FrameID,SensorIdx,Ndet,DetIdx,Range_m,Azimuth_deg,Vel_mps,Amp_lin,Power_dB\n");
            std::fclose(f);
        }
    }
    {
        FILE* f = std::fopen(g_metaFile, "wb");
        if (f) {
            const double frame_dt = (double)g_params.Nc * (double)g_params.Tc;
            std::fprintf(f, "RawSignalSynth metadata\n");
            std::fprintf(f, "timestamp=%s\n", g_runTS);
            std::fprintf(f, "model=range_doppler_coupled\n");
            std::fprintf(f, "format=float32\n");
            std::fprintf(f, "layout=idx=((s*Nc)+c)*Nrx+rx\n");
            std::fprintf(f, "Ns=%d\nNc=%d\nNrx=%d\nNtx=%d\n", g_params.Ns, g_params.Nc, g_params.Nrx, g_params.Ntx);
            std::fprintf(f, "fs=%.9f\nTc=%.12e\ndF=%.9f\nf0=%.9f\n", g_params.fs, g_params.Tc, g_params.dF, g_params.f0);
            std::fprintf(f, "frame_dt=%.12e\n", frame_dt);
            std::fclose(f);
        }
    }
    {
        FILE* f = std::fopen(g_frameIndexFile, "wb");
        if (f) {
            std::fprintf(f, "BinFrameIndex,FrameID,TimeFired_s,SensorIdx,NdetUse\n");
            std::fclose(f);
        }
    }

    g_frameSeq = 0ULL;
    g_binFrameIndex = 0ULL;
    g_ifBinInit = false;
    g_ifFrameCount = 0ULL;
    g_runInit = true;

    std::printf("[GPU] Run init TS=%s\n", g_runTS);
    std::printf("[GPU] RSI detections: %s\n", g_rsiDetFile);
    std::printf("[GPU] Synth input  : %s\n", g_synthInFile);
    std::printf("[GPU] BIN          : %s\n", g_ifBinName);
    std::printf("[GPU] META         : %s\n", g_metaFile);
    std::printf("[GPU] FrameIndex   : %s\n", g_frameIndexFile);
}

static void AppendFrameIndexRow(double time_s, int sensorIdx, size_t ndetUse)
{
    FILE* f = std::fopen(g_frameIndexFile, "ab");
    if (!f) return;
    std::fprintf(f, "%llu,%llu,%.9f,%d,%zu\n",
                 (unsigned long long)g_binFrameIndex,
                 (unsigned long long)g_frameSeq,
                 time_s, sensorIdx, ndetUse);
    std::fclose(f);
}

static void AppendRSIDetections(double time_s, int sensorIdx, const std::vector<tOutputPointC_R>& detUse)
{
    FILE* f = std::fopen(g_rsiDetFile, "ab");
    if (!f) return;
    unsigned int detIdx = 0;
    for (const auto& d : detUse) {
        const double x = (double)d.tCoord_R.CoordCartOut_R.x;
        const double y = (double)d.tCoord_R.CoordCartOut_R.y;
        const double z = (double)d.tCoord_R.CoordCartOut_R.z;
        const double range = std::sqrt(x*x + y*y + z*z);
        const double az    = std::atan2(y, x);
        const double el    = std::atan2(z, std::sqrt(x*x + y*y));
        std::fprintf(f,
          "%.9f,%llu,%d,%u,%.6f,%.9f,%.9f,%.6f,%.3f\n",
          time_s, (unsigned long long)g_frameSeq,
          sensorIdx, detIdx++, range, az, el,
          (double)d.vel, (double)d.PowerdB);
    }
    std::fclose(f);
}

static void AppendSynthInput(double time_s, int sensorIdx, const std::vector<SynthDetection>& hSynth)
{
    FILE* f = std::fopen(g_synthInFile, "ab");
    if (!f) return;
    const int Ndet = (int)hSynth.size();
    for (int i = 0; i < Ndet; ++i) {
        const auto& s = hSynth[i];
        const double p_lin = (double)s.amp_lin * (double)s.amp_lin + 1e-20;
        const double p_db  = 10.0 * std::log10(p_lin);
        std::fprintf(f,
          "%.9f,%llu,%d,%d,%d,%.6f,%.6f,%.6f,%.9e,%.3f\n",
          time_s, (unsigned long long)g_frameSeq,
          sensorIdx, Ndet, i,
          (double)s.r0_m, (double)s.az_deg,
          (double)s.v0_ms, (double)s.amp_lin, p_db);
    }
    std::fclose(f);
}

static void ExportIFFrameToAnsysADC(const std::vector<float>& h_frame,
                                    int Ns, int Nc, int Nrx, int Ntx)
{
    EnsureSimOutputDir();

    if (!g_adcInit) {
        MakeTimestamp(g_adcTS, sizeof(g_adcTS));
        g_adcInit    = true;
        g_adcFrameId = 0ULL;
    }

    char fname[256];
    std::snprintf(fname, sizeof(fname),
              "SimOutput\\DLL Cuda Outputs\\ADC_Synth_GPU_%s_frame_%06llu_adcsamples.txt",
              g_adcTS, (unsigned long long)g_adcFrameId);

    FILE* f = std::fopen(fname, "w");
    if (!f) {
        std::printf("[GPU] ERROR: cannot open %s for write (ADC)\n", fname);
        return;
    }

    std::fprintf(f, "%d %d %d %d", Ntx, Nrx, Nc, Ns);
    for (int tx = 0; tx < Ntx; ++tx) std::fprintf(f, " %d", tx);
    for (int rx = 0; rx < Nrx; ++rx) std::fprintf(f, " %d", rx);
    std::fprintf(f, " %.6f %.6f %.6f %.6f\n",
                 0.0, (double)(Nc - 1), 0.0, (double)(Ns - 1));

    for (int rx = 0; rx < Nrx; ++rx) {
        for (int c = 0; c < Nc; ++c) {
            for (int s = 0; s < Ns; ++s) {
                const size_t idx = ((size_t)s * (size_t)Nc + (size_t)c) * (size_t)Nrx + (size_t)rx;
                std::fprintf(f, "%.12e\n", (double)h_frame[idx]);
            }
        }
    }

    std::fclose(f);
    std::printf("[GPU] Wrote ADC frame #%llu to %s\n",
                (unsigned long long)g_adcFrameId, fname);
    ++g_adcFrameId;
}

__device__ __forceinline__
float calcAbsoluteSquare(float2 z) { return z.x*z.x + z.y*z.y; }

// ============================================================================
//   EQ.3.5 RAW-SIGNAL SYNTHESIZER — WITH RANGE-DOPPLER COUPLING
// ============================================================================
__device__ __forceinline__
float beam_lut_fetch(const float* lut, int azIdx, int tx, int rx, int elIdx,
                     int azSize, int elSize) {
  int row = tx*4 + rx;
  size_t idx = ((size_t)azIdx * 16 + (size_t)row) * (size_t)elSize + (size_t)elIdx;
  return lut[idx];
}

__device__ __forceinline__
int ang_deg_to_idx(float deg) {
  float x = (deg + 90.0f) * 10.0f;
  int i = (int)lrintf(x);
  if (i < 0) i = 0; if (i > 1800) i = 1800;
  return i;
}

__device__ __forceinline__
float randn01(uint32_t& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  float u1 = (s * 2.3283064365386963e-10f); if (u1 < 1e-7f) u1 = 1e-7f;
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  float u2 = (s * 2.3283064365386963e-10f);
  return sqrtf(-2.f * logf(u1)) * __cosf(2.f * (float)M_PI * u2);
}

__global__ void buildSynthDetectionsFromIPs(
    const user_RadarInteractionPoint** interactionPoints,
    int nInteractionPoints,
    SynthDetection* out,
    int maxOut)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nInteractionPoints || idx >= maxOut) return;
  const user_RadarInteractionPoint* ip = interactionPoints[idx];
  SynthDetection d;
  d.r0_m    = ip->range;
  d.v0_ms   = ip->vel;
  d.az_deg  = ip->direction.angles.phi * (180.0f / (float)M_PI);
  d.el_deg  = 0.0f;
  d.amp_lin = hypotf(ip->electric_field.x, ip->electric_field.y);
  out[idx] = d;
}

__global__ void synth_eq35_kernel_lut(
    const SynthDetection* dets, int Ndet,
    const float* __restrict__ lut, int azSize, int elSize,
    SensorParams P, float noise_sigma,
    float* __restrict__ outFrame)
{
  int s  = blockIdx.x * blockDim.x + threadIdx.x;
  int c  = blockIdx.y * blockDim.y + threadIdx.y;
  int rx = blockIdx.z;
  if (s >= P.Ns || c >= P.Nc || rx >= P.Nrx) return;

  float dt = 1.0f / P.fs;
  float t  = s * dt;
  float t2 = t * t;
  int seq  = c & 7;

  uint32_t seed = 2166136261u ^ (rx*131u + c*313u + s*911u);
  float acc = 0.f;

  #pragma unroll
  for (int k = 0; k < Ndet; ++k) {
    SynthDetection d = dets[k];
    int iaz = ang_deg_to_idx(d.az_deg);
    int iel = ang_deg_to_idx(d.el_deg);

    float A   = -(2.f * (float)M_PI / P.c) * (2.f * d.v0_ms * P.f0 + 2.f * d.r0_m * P.dF / P.Tc);
    float mod = fmodf(4.f * (float)M_PI * P.f0 * d.v0_ms * c * P.Tc / P.c, 2.f * (float)M_PI);
    float phi_coupling = -(4.f * (float)M_PI * P.dF / (P.c * P.Tc)) * d.v0_ms * t2;
    float phi_carrier_const = -(4.f * (float)M_PI * P.f0 / P.c) * d.r0_m;

    for (int tx = 0; tx < 4; ++tx) {
      float phi_mimo = d_mimo[tx][seq];
      float phi_beam = beam_lut_fetch(lut, iaz, tx, rx, iel, azSize, elSize);

      float phi = A * t
                + phi_coupling
                + phi_carrier_const
                + phi_beam
                + phi_mimo
                - mod;

      acc += d.amp_lin * __cosf(phi);
    }
  }

  acc += noise_sigma * randn01(seed);
  size_t idx = ((size_t)s * (size_t)P.Nc + (size_t)c) * (size_t)P.Nrx + (size_t)rx;
  outFrame[idx] = acc;
}

// ============================================================================
//                 stock kernels
// ============================================================================
__global__
static void writeInteractionPointToRadarCube(
  float rMin, float rMax, float vMin, float vMax,
  int nInteractionPoints,
  const user_RadarInteractionPoint** interactionPoints,
  user_RadarCube radarCube)
{
  unsigned int indexInteractionPoint = getThreadIndex1d();
  if (indexInteractionPoint >= nInteractionPoints) return;

  float range    = interactionPoints[indexInteractionPoint]->range;
  float velocity = interactionPoints[indexInteractionPoint]->vel;
  float angle    = interactionPoints[indexInteractionPoint]->direction.angles.phi;

  float rangeClamped    = (range    >= rMax) ? rMax : range;
  rangeClamped          = (rangeClamped <= rMin) ? rMin : rangeClamped;
  float velocityClamped = (velocity >= vMax) ? vMax : velocity;
  velocityClamped       = (velocityClamped <= vMin) ? vMin : velocityClamped;
  float angleClamped    = (angle >= +CUDART_PIO2_F) ? +CUDART_PIO2_F : angle;
  angleClamped          = (angleClamped <= -CUDART_PIO2_F) ? -CUDART_PIO2_F : angleClamped;

  int indexRangeBin    = (int)(((rangeClamped    - rMin) / (rMax - rMin)) * (radarCube.n_r   - 1));
  int indexVelocityBin = (int)(((velocityClamped - vMin) / (vMax - vMin)) * (radarCube.n_v   - 1));
  int indexAngleBin    = (int)(((angleClamped + CUDART_PIO2_F) / CUDART_PI_F) * (radarCube.n_vrx - 1));

  int indexCell = indexRangeBin
                + indexVelocityBin * radarCube.n_r
                + indexAngleBin    * radarCube.n_r * radarCube.n_v;

  float2 e = interactionPoints[indexInteractionPoint]->electric_field;
  atomicAdd(&radarCube.radar_cube[indexCell].x, e.x);
  atomicAdd(&radarCube.radar_cube[indexCell].y, e.y);
}

__global__
static void computePeakFinder(
  float rMin, float rMax, float vMin, float vMax,
  int nBinsRange, int nBinsVelocity, int nBinsAngle,
  float2* radarCube, int nMaxDetections,
  unsigned int* const nOutputPointCloudDevice,
  tOutputPointC_R* const outputPointCloudDevice)
{
  int nPeakFinderLayers = 5;

  uint3 tid = getThreadIndex3d();
  int indexRange    = (int)tid.x;
  int indexVelocity = (int)tid.y;
  int indexAngle    = (int)tid.z;
  if (indexRange >= nBinsRange || indexVelocity >= nBinsVelocity || indexAngle >= nBinsAngle) return;

  unsigned int idx = indexRange
                   + indexVelocity * nBinsRange
                   + indexAngle    * nBinsRange * nBinsVelocity;
  float powerUT      = calcAbsoluteSquare(radarCube[idx]);
  float powerDecibel = 10.f * log10f(powerUT + 1e-20f);
  if (powerDecibel < -90.f) return;

  float peak = 0.f;
  for (int dr = -nPeakFinderLayers; dr <= +nPeakFinderLayers; ++dr) {
    int ir = indexRange + dr; if (ir < 0 || ir >= nBinsRange) continue;
    for (int dv = -nPeakFinderLayers; dv <= +nPeakFinderLayers; ++dv) {
      int iv = indexVelocity + dv; if (iv < 0 || iv >= nBinsVelocity) continue;
      unsigned int baseAngle = (unsigned int)indexAngle * (unsigned int)nBinsRange * (unsigned int)nBinsVelocity;
      int ii = ir + iv * nBinsRange + (int)baseAngle;
      peak = fmaxf(peak, calcAbsoluteSquare(radarCube[ii]));
    }
  }
  if (fabs(powerUT) <= 0.999f * peak) return;

  peak = 0.f;
  for (int da = -nPeakFinderLayers; da <= +nPeakFinderLayers; ++da) {
    int ia = indexAngle + da; if (ia < 0 || ia >= nBinsAngle) continue;
    int ii = indexRange + indexVelocity * nBinsRange + ia * nBinsRange * nBinsVelocity;
    peak = fmaxf(peak, calcAbsoluteSquare(radarCube[ii]));
  }
  if (fabs(powerUT) <= 0.999f * peak) return;

  float range    = (float)indexRange    / (float)(nBinsRange    - 1) * (rMax - rMin) + rMin;
  float velocity = (float)indexVelocity / (float)(nBinsVelocity - 1) * (vMax - vMin) + vMin;
  float angle    = (float)indexAngle    / (float)(nBinsAngle    - 1) * CUDART_PI_F - CUDART_PIO2_F;

  unsigned int w = atomicInc(nOutputPointCloudDevice, UINT_MAX);
  if (w >= (unsigned int)nMaxDetections) return;

  outputPointCloudDevice[w].vel                       = velocity;
  outputPointCloudDevice[w].PowerdB                   = powerDecibel;
  outputPointCloudDevice[w].tCoord_R.CoordCartOut_R.x = range * cosf(angle);
  outputPointCloudDevice[w].tCoord_R.CoordCartOut_R.y = range * sinf(angle);
  outputPointCloudDevice[w].tCoord_R.CoordCartOut_R.z = 0.f;
}

// ============================================================================
//                               USER INIT
// ============================================================================
LIBAPI
int radarInit(
  int sensorIndex,
  int nRangeBins, int nVelocityBins, int nVRxBins,
  float rangeMin, float rangeMax, float velocityMin, float velocityMax,
  int nDetectionsMax,
  float2* positionVRx, float* amplitudeVRx, float* phaseVRx,
  user_RadarDataType dataMode, user_RadarDataType outputType,
  const char* userParameters)
{
  std::printf("[GPU] radarInit() enter, sensorIndex=%d\n", sensorIndex);
  std::fflush(stdout);

  if (RADAR_OUTPUT_VRX == outputType) {
    std::cerr << __FUNCTION__ << "(): Wrong output type.\n"; return -1;
  }
  if (RADAR_OUTPUT_VRX == dataMode) {
    std::cerr << __FUNCTION__ << "(): Wrong data input type.\n"; return -1;
  }

  RadarInitData initData;
  initData.nRangeBins    = nRangeBins;
  initData.nVelocityBins = nVelocityBins;
  initData.nPhiBins      = nVRxBins;
  initData.nDetectionsMax = nDetectionsMax;
  initData.nOutputPointCloudDevice = nullptr;
  initData.d_frame = nullptr;
  initData.d_dets  = nullptr;
  initData.rangeMin = rangeMin;  initData.rangeMax = rangeMax;
  initData.velMin   = velocityMin; initData.velMax = velocityMax;

  CHECK_CUDA(cudaMalloc(&initData.nOutputPointCloudDevice, sizeof(unsigned int)));

  static bool globalInitDone = false;
  if (!globalInitDone) {
    std::printf("[GPU] radarInit(): entering globalInit block\n");
    PrintWorkingDir();
    EnsureSimOutputDir();
    CHECK_CUDA(cudaMemcpyToSymbol(d_mimo, kMimoHost, sizeof(kMimoHost)));

    const size_t nBytes = (size_t)kBeamAzi * kBeamTxRx * (size_t)kBeamEle * sizeof(float);
    std::vector<float> host(nBytes / sizeof(float));
    FILE* fb = std::fopen(kBeamLUTBin, "rb");
    if (fb) {
      size_t bytesRead = std::fread(host.data(), 1, nBytes, fb);
      std::fclose(fb);
      if (bytesRead == nBytes) {
        CHECK_CUDA(cudaMalloc(&d_beamLUT, nBytes));
        CHECK_CUDA(cudaMemcpy(d_beamLUT, host.data(), nBytes, cudaMemcpyHostToDevice));
        std::printf("[GPU] Beam LUT loaded, %.1f MB\n", nBytes/1048576.0);
      } else {
        std::printf("[GPU] ERROR: LUT size mismatch\n");
      }
    } else {
      std::printf("[GPU] ERROR: LUT not found at %s\n", kBeamLUTBin);
    }
    globalInitDone = true;
  }

  if (d_beamLUT != nullptr) {
    const size_t frameElems = (size_t)g_params.Ns * g_params.Nc * g_params.Nrx;
    CHECK_CUDA(cudaMalloc(&initData.d_frame, frameElems * sizeof(float)));
    initData.nEq35Max = std::max(nDetectionsMax, SYNTH_IP_DEBUG_MAX);
    CHECK_CUDA(cudaMalloc(&initData.d_dets, (size_t)initData.nEq35Max * sizeof(SynthDetection)));
  }

  radarInitData.emplace(sensorIndex, std::move(initData));
  std::printf("[GPU] radarInit() exit OK, sensorIndex=%d\n", sensorIndex);
  std::fflush(stdout);
  return 0;
}

// ============================================================================
//                               USER GENERATION
// ============================================================================
LIBAPI
int radarSignalGeneration(
  int sensorIndex,
  const user_RadarInteractionPoint** interactionPoints,
  int nInteractionPoints,
  user_RadarCube radarCube)
{
  std::printf("[GPU] radarSignalGeneration() enter, sensorIndex=%d, nIPs=%d\n",
              sensorIndex, nInteractionPoints);
  std::fflush(stdout);

  auto itInit = radarInitData.find(sensorIndex);
  if (itInit == radarInitData.end()) return 0;
  auto& initData = itInit->second;

  unsigned int nBinsCube = radarCube.n_r * radarCube.n_v * radarCube.n_vrx;
  CHECK_CUDA(cudaMemset(radarCube.radar_cube, 0, nBinsCube * sizeof(float2)));
  if (0 == nInteractionPoints) return 0;

  unsigned int blockSize = DefaultBlocksize;
  unsigned int gridSize  = (nInteractionPoints + DefaultBlocksize - 1) / DefaultBlocksize;
  writeInteractionPointToRadarCube<<<gridSize, blockSize>>>(
    initData.rangeMin, initData.rangeMax,
    initData.velMin,   initData.velMax,
    nInteractionPoints, interactionPoints, radarCube);
  CHECK_CUDA(cudaGetLastError());

  std::printf("[GPU] radarSignalGeneration() exit\n");
  std::fflush(stdout);
  return 0;
}

// ============================================================================
//                               USER PROCESSING
// ============================================================================
LIBAPI
int radarSignalProcessing(
  int sensorIndex,
  user_RadarCube radarCube,
  tOutputPointC_R* outputPointCloudDevice,
  int* nOutputPointCloudHost,
  tOutputVRx_R* outputVrxDevice,
  int* nOutputVrxHost)
{
  std::printf("[GPU] radarSignalProcessing() enter, sensorIndex=%d\n", sensorIndex);
  std::fflush(stdout);

  auto& initData = radarInitData.at(sensorIndex);

  CHECK_CUDA(cudaMemset(initData.nOutputPointCloudDevice, 0, sizeof(unsigned int)));

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(
    (radarCube.n_r   + blockSize.x - 1) / blockSize.x,
    (radarCube.n_v   + blockSize.y - 1) / blockSize.y,
    (radarCube.n_vrx + blockSize.z - 1) / blockSize.z);

  computePeakFinder<<<gridSize, blockSize>>>(
    initData.rangeMin, initData.rangeMax,
    initData.velMin,   initData.velMax,
    radarCube.n_r, radarCube.n_v, radarCube.n_vrx,
    radarCube.radar_cube,
    initData.nDetectionsMax,
    initData.nOutputPointCloudDevice,
    outputPointCloudDevice);
  CHECK_CUDA(cudaGetLastError());

  unsigned int nDetectionsHost = 0;
  CHECK_CUDA(cudaMemcpy(&nDetectionsHost, initData.nOutputPointCloudDevice,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));
  if (nDetectionsHost > (unsigned)initData.nDetectionsMax)
    nDetectionsHost = (unsigned)initData.nDetectionsMax;
  *nOutputPointCloudHost = (int)nDetectionsHost;

  std::vector<tOutputPointC_R> detHost;
  std::vector<tOutputPointC_R> detUse;

  if (nDetectionsHost > 0) {
    detHost.resize(nDetectionsHost);
    CHECK_CUDA(cudaMemcpy(detHost.data(), outputPointCloudDevice,
                          nDetectionsHost * sizeof(tOutputPointC_R),
                          cudaMemcpyDeviceToHost));

    const float PWR_DB_MIN = -95.0f;
    const float R_MIN      =   7.0f;
    const float R_MAX      = 120.0f;
    const int   TOPK       = 100;

    for (const auto& d : detHost) {
      if (d.PowerdB < PWR_DB_MIN) continue;
      const double x = d.tCoord_R.CoordCartOut_R.x;
      const double y = d.tCoord_R.CoordCartOut_R.y;
      const double z = d.tCoord_R.CoordCartOut_R.z;
      const double r = std::sqrt(x*x + y*y + z*z);
      if (r < R_MIN || r > R_MAX) continue;
      detUse.push_back(d);
    }
    std::sort(detUse.begin(), detUse.end(),
              [](const tOutputPointC_R& a, const tOutputPointC_R& b){
                  return a.PowerdB > b.PowerdB; });
    if ((int)detUse.size() > TOPK) detUse.resize(TOPK);

    g_allDetectionsGPU.insert(g_allDetectionsGPU.end(), detUse.begin(), detUse.end());

#if EXPORT_ANSYS_AVX
    ExportDetectionsToANSYS_AVX_GPU(detUse);
#endif
  }

#if SYNTH_FROM_DETECTIONS
  const bool isLogSensor = (sensorIndex == LOG_SENSOR_INDEX);

  if (isLogSensor && d_beamLUT != nullptr) {

    InitRunFilesOnce();
    const double time_s = GetBestTime_s(radarCube);

    int nDetsToUse = (int)std::min((size_t)initData.nDetectionsMax, detUse.size());
    std::vector<SynthDetection> hSynth;
    hSynth.reserve((size_t)nDetsToUse);

    for (int i = 0; i < nDetsToUse; ++i) {
      const auto& d = detUse[i];
      const float x = d.tCoord_R.CoordCartOut_R.x;
      const float y = d.tCoord_R.CoordCartOut_R.y;
      SynthDetection sd;
      sd.r0_m    = std::sqrt(x*x + y*y);
      sd.v0_ms   = d.vel;
      sd.az_deg  = std::atan2(y, x) * (180.f / (float)M_PI);
      sd.el_deg  = 0.f;
      sd.amp_lin = std::sqrt(std::max(std::pow(10.f, d.PowerdB/10.f), 1e-20f));
      hSynth.push_back(sd);
    }

    AppendSynthInput(time_s, sensorIndex, hSynth);

    if (!hSynth.empty()) {
      CHECK_CUDA(cudaMemcpy(initData.d_dets, hSynth.data(),
                            hSynth.size() * sizeof(SynthDetection),
                            cudaMemcpyHostToDevice));
    }

    const int NsDbg  = g_params.Ns;
    const int NcDbg  = g_params.Nc;
    const int NrxDbg = g_params.Nrx;
    SensorParams Pdbg = g_params;
    const float noiseSigma = 0.f;

    dim3 blockSynth(128, 4, 1);
    dim3 gridSynth(
        (NsDbg + blockSynth.x - 1) / blockSynth.x,
        (NcDbg + blockSynth.y - 1) / blockSynth.y,
        NrxDbg);

    std::printf("[GPU] Launching synth kernel: Ns=%d Nc=%d Ndet=%d FrameID=%llu\n",
                NsDbg, NcDbg, (int)hSynth.size(), (unsigned long long)g_frameSeq);
    std::fflush(stdout);

    synth_eq35_kernel_lut<<<gridSynth, blockSynth>>>(
        initData.d_dets, (int)hSynth.size(),
        d_beamLUT, kBeamAzi, kBeamEle,
        Pdbg, noiseSigma,
        initData.d_frame);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const size_t frameElems = (size_t)NsDbg * (size_t)NcDbg * (size_t)NrxDbg;
    std::vector<float> h_frame(frameElems);
    CHECK_CUDA(cudaMemcpy(h_frame.data(), initData.d_frame,
                          frameElems * sizeof(float), cudaMemcpyDeviceToHost));

#if EXPORT_ANSYS_ADC
    ExportIFFrameToAnsysADC(h_frame, NsDbg, NcDbg, NrxDbg, g_params.Ntx);
#endif

    if (!g_ifBinInit) {
      FILE* fb0 = std::fopen(g_ifBinName, "wb");
      if (fb0) std::fclose(fb0);
      g_ifBinInit    = true;
      g_ifFrameCount = 0ULL;
      std::printf("[GPU] Raw-signal file: %s\n", g_ifBinName);
    }
    FILE* fb = std::fopen(g_ifBinName, "ab");
    if (fb) {
      std::fwrite(h_frame.data(), sizeof(float), frameElems, fb);
      std::fclose(fb);
      ++g_ifFrameCount;
    }

    AppendRSIDetections(time_s, sensorIndex, detUse);
    AppendFrameIndexRow(time_s, sensorIndex, detUse.size());

    ++g_binFrameIndex;
    ++g_frameSeq;
  }
#endif

  return 0;
}

// ============================================================================
//                               USER CLEANUP
// ============================================================================
LIBAPI
int radarCleanup(int sensorIndex)
{
  std::printf("[GPU] radarCleanup() enter, sensorIndex=%d\n", sensorIndex);
  std::fflush(stdout);

  auto it = radarInitData.find(sensorIndex);
  if (it != radarInitData.end()) {
    if (it->second.nOutputPointCloudDevice) cudaFree(it->second.nOutputPointCloudDevice);
    if (it->second.d_frame)                 cudaFree(it->second.d_frame);
    if (it->second.d_dets)                  cudaFree(it->second.d_dets);
    radarInitData.erase(it);
  }

  if (radarInitData.empty()) {
    if (!g_allDetectionsGPU.empty()) g_allDetectionsGPU.clear();
    if (d_beamLUT) { cudaFree(d_beamLUT); d_beamLUT = nullptr; }
    g_runInit      = false;
    g_ifBinInit    = false;
    g_ifFrameCount = 0ULL;
    g_frameSeq     = 0ULL;
    g_binFrameIndex = 0ULL;
  }

  return 0;
}

// ============================================================================
//                     SIGNATURE STATIC ASSERTS
// ============================================================================
#include <type_traits>
static_assert(std::is_same<decltype(&radarInit), user_RadarInit_Fct>::value,
              "Signature of radarInit differs from expected");
static_assert(std::is_same<decltype(&radarSignalGeneration), user_RadarSignalGeneration_Fct>::value,
              "Signature of radarSignalGeneration differs from expected");
static_assert(std::is_same<decltype(&radarSignalProcessing), user_RadarSignalProcessing_Fct>::value,
              "Signature of radarSignalProcessing differs from expected");
static_assert(std::is_same<decltype(&radarCleanup), user_RadarCleanup_Fct>::value,
              "Signature of radarCleanup differs from expected");
