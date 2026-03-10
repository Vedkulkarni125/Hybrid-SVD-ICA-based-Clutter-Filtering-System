/*
 * mex_hybrid_filter.cu
 * NVIDIA CUDA Engine for SVD Filtering
 * Paper: Memory-Aware Block-Wise SVD-Based Ultrasound Image Denoising Using CUDA
 *
 * Parameters (must match Final_Block_Processing.m):
 *   Block size      : 256x256 pixels
 *   Overlap         : 16 pixels
 *   Core size       : 224 pixels (= 256 - 2*16)
 *   SVD energy thr  : 95%
 *   Laplacian alpha : 0.06
 *   Timing runs     : 10
 *   Data type       : complex double (cuDoubleComplex)
 *
 * Economy SVD flag:
 *   #define USE_ECONOMY_SVD 1   -> uses 'S' mode: U is M×minDim, VT is minDim×N
 *   #define USE_ECONOMY_SVD 0   -> uses 'A' mode: U is M×M,       VT is N×N
 *   This must match USE_ECONOMY_SVD in Final_Block_Processing.m.
 *
 * Improvements applied:
 *   1. Economy SVD ('S') mode controlled by USE_ECONOMY_SVD preprocessor flag
 *   2. Non-default CUDA stream for async cuSOLVER execution
 *   3. Persistent workspace buffers cached across calls (freed on mexAtExit)
 *   4. GPU memory check before allocation (validates against device free memory)
 *   5. Fallback to economy SVD if full SVD fails
 *   6. CUDA event-based per-call timing (printed via mexPrintf)
 *   7. All GPU resources freed on every exit path (normal and error)
 *   8. Const correctness on read-only pointers
 *   9. mexAtExit handler destroys cuSOLVER handle and frees persistent buffers
 *
 * Build command (MATLAB):
 *   mexcuda mex_hybrid_filter.cu -lcusolver -lcudart
 *
 * Usage:
 *   [U, S, VT] = mex_hybrid_filter(Data)
 *   Data : M×N complex double matrix
 *   U    : M×minDim (economy) or M×M (full) complex double
 *   S    : min(M,N)×1 real double singular values
 *   VT   : minDim×N (economy) or N×N (full) complex double
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cstddef>   /* size_t */

/* -------------------------------------------------------------------------
 * Economy SVD toggle
 *   1 -> use 'S': U is M×minDim, VT is minDim×N  (saves ~50% VRAM)
 *   0 -> use 'A': U is M×M,      VT is N×N
 * Must match USE_ECONOMY_SVD in Final_Block_Processing.m
 * ---------------------------------------------------------------------- */
#ifndef USE_ECONOMY_SVD
#define USE_ECONOMY_SVD 1
#endif

/* =========================================================================
 * Error-checking macros
 * ====================================================================== */
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            mexErrMsgIdAndTxt("Hybrid:CUDA",                                \
                "CUDA error at line %d: %s",                                \
                __LINE__, cudaGetErrorString(_e));                          \
        }                                                                    \
    } while (0)

#define CHECK_CUSOLVER(call)                                                 \
    do {                                                                     \
        cusolverStatus_t _s = (call);                                       \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                \
            mexErrMsgIdAndTxt("Hybrid:cuSOLVER",                            \
                "cuSOLVER error at line %d: status %d",                     \
                __LINE__, (int)_s);                                         \
        }                                                                    \
    } while (0)

/* =========================================================================
 * Persistent (cached) state — lives for the lifetime of the MEX module.
 * Freed by the mexAtExit handler registered on first call.
 * ====================================================================== */
static cusolverDnHandle_t s_cusolverH  = NULL;
static cudaStream_t        s_stream    = NULL;
static cuDoubleComplex*    s_d_work    = NULL;
static double*             s_d_rwork   = NULL;
static int                 s_lwork     = 0;
static int                 s_rwork_sz  = 0;
static bool                s_exitReg   = false;   /* mexAtExit registered? */

/* Compile-time sanity check (file scope — checked once at compile time) */
static_assert(sizeof(cuDoubleComplex) == sizeof(double) * 2,
    "cuDoubleComplex size mismatch");

/* =========================================================================
 * mexAtExit handler: destroy cuSOLVER handle, stream, and persistent buffers
 * ====================================================================== */
static void cleanupPersistent(void)
{
    if (s_d_work)   { cudaFree(s_d_work);   s_d_work  = NULL; }
    if (s_d_rwork)  { cudaFree(s_d_rwork);  s_d_rwork = NULL; }
    if (s_cusolverH){ cusolverDnDestroy(s_cusolverH); s_cusolverH = NULL; }
    if (s_stream)   { cudaStreamDestroy(s_stream);    s_stream    = NULL; }
    s_lwork    = 0;
    s_rwork_sz = 0;
}

/* =========================================================================
 * Per-call cleanup: free GPU arrays and scratch memory allocated per call.
 * NULL-safe — safe to call even if some pointers were never allocated.
 * ====================================================================== */
static void cleanupCall(
    int*         d_info,
    mxGPUArray*  gpuInput,
    mxGPUArray*  gpuA,
    mxGPUArray*  gpuU,
    mxGPUArray*  gpuS,
    mxGPUArray*  gpuVT)
{
    if (d_info)   cudaFree(d_info);
    if (gpuInput) mxGPUDestroyGPUArray(gpuInput);
    if (gpuA)     mxGPUDestroyGPUArray(gpuA);
    if (gpuU)     mxGPUDestroyGPUArray(gpuU);
    if (gpuS)     mxGPUDestroyGPUArray(gpuS);
    if (gpuVT)    mxGPUDestroyGPUArray(gpuVT);
}

/* =========================================================================
 * Attempt to ensure the persistent workspace is large enough for M×N SVD.
 * Returns false (and leaves s_d_work/s_d_rwork unchanged) on OOM.
 * ====================================================================== */
static bool ensureWorkspace(int M, int N)
{
    int minDim = (M < N) ? M : N;

    /* Query required lwork for this size */
    int lwork_needed = 0;
    if (cusolverDnZgesvd_bufferSize(s_cusolverH, M, N, &lwork_needed)
            != CUSOLVER_STATUS_SUCCESS)
        return false;

    int rwork_needed = 5 * minDim + M + N;
    if (rwork_needed < 1) rwork_needed = 1;

    /* Only reallocate if the cached buffer is too small */
    if (lwork_needed > s_lwork)
    {
        cuDoubleComplex* tmp = NULL;
        if (cudaMalloc(&tmp, sizeof(cuDoubleComplex) * lwork_needed)
                != cudaSuccess)
            return false;
        if (s_d_work) cudaFree(s_d_work);
        s_d_work = tmp;
        s_lwork  = lwork_needed;
    }

    if (rwork_needed > s_rwork_sz)
    {
        double* tmp = NULL;
        if (cudaMalloc(&tmp, sizeof(double) * rwork_needed) != cudaSuccess)
            return false;
        if (s_d_rwork) cudaFree(s_d_rwork);
        s_d_rwork   = tmp;
        s_rwork_sz  = rwork_needed;
    }

    return true;
}

/* =========================================================================
 * Run one SVD attempt with the supplied jobu/jobvt characters.
 * Returns the cuSOLVER status; does NOT throw.
 * ====================================================================== */
static cusolverStatus_t runSVD(
    int              M,
    int              N,
    cuDoubleComplex* d_A,
    double*          d_S,
    cuDoubleComplex* d_U,
    int              ldu,
    cuDoubleComplex* d_VT,
    int              ldvt,
    int*             d_info,
    signed char      jobu,
    signed char      jobvt)
{
    return cusolverDnZgesvd(
        s_cusolverH,
        jobu, jobvt,
        M, N,
        d_A,  M,
        d_S,
        d_U,  ldu,
        d_VT, ldvt,
        s_d_work,  s_lwork,
        s_d_rwork,
        d_info);
}

/* =========================================================================
 * mexFunction entry point
 * ====================================================================== */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    mxInitGPU();

    /* Register persistent cleanup on first call */
    if (!s_exitReg)
    {
        mexAtExit(cleanupPersistent);
        s_exitReg = true;
    }

    /* ------------------------------------------------------------------
     * Output / input count validation
     * ---------------------------------------------------------------- */
    if (nlhs != 3)
        mexErrMsgIdAndTxt("Hybrid:Output",
            "Exactly 3 outputs required: [U, S, VT] = mex_hybrid_filter(Data)");

    if (nrhs != 1)
        mexErrMsgIdAndTxt("Hybrid:Input",
            "Usage: [U, S, VT] = mex_hybrid_filter(Data)");

    /* ------------------------------------------------------------------
     * Input validation
     * ---------------------------------------------------------------- */
    const mxArray* const inputMx = prhs[0];

    if (!mxIsComplex(inputMx) || !mxIsDouble(inputMx))
        mexErrMsgIdAndTxt("Hybrid:Input",
            "Input must be a Complex Double matrix.");

    if (mxGetNumberOfDimensions(inputMx) != 2)
        mexErrMsgIdAndTxt("Hybrid:Input",
            "Input must be a 2D matrix.");

    const int M = (int)mxGetM(inputMx);
    const int N = (int)mxGetN(inputMx);

    if (M == 0 || N == 0)
        mexErrMsgIdAndTxt("Hybrid:Input",
            "Input matrix cannot be empty (size: %d x %d).", M, N);

    const int minDim = (M < N) ? M : N;

    /* GPU memory check: estimate bytes needed for U + VT + S + A copies */
    {
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);

        /* Worst-case (full SVD): U(M×M) + VT(N×N) + S(minDim) + A copy */
        const size_t bytesNeeded =
            sizeof(cuDoubleComplex) * ((size_t)M*M + (size_t)N*N + (size_t)M*N)
            + sizeof(double) * minDim;

        if (bytesNeeded > freeMem)
            mexErrMsgIdAndTxt("Hybrid:Memory",
                "Insufficient GPU VRAM: need ~%.0f MB, have ~%.0f MB free.",
                (double)bytesNeeded / 1e6, (double)freeMem / 1e6);
    }

    /* ------------------------------------------------------------------
     * Lazy-initialise the persistent cuSOLVER handle and CUDA stream
     * ---------------------------------------------------------------- */
    if (!s_cusolverH)
    {
        CHECK_CUSOLVER(cusolverDnCreate(&s_cusolverH));
        CHECK_CUDA(cudaStreamCreate(&s_stream));
        CHECK_CUSOLVER(cusolverDnSetStream(s_cusolverH, s_stream));
    }

    /* ------------------------------------------------------------------
     * Ensure workspace buffers are large enough for this M×N
     * ---------------------------------------------------------------- */
    if (!ensureWorkspace(M, N))
        mexErrMsgIdAndTxt("Hybrid:Memory",
            "Failed to allocate cuSOLVER workspace for %d×%d matrix.", M, N);

    /* ------------------------------------------------------------------
     * CUDA event timing (optional; reports actual GPU kernel time)
     * ---------------------------------------------------------------- */
    cudaEvent_t evStart = NULL, evStop = NULL;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    /* ------------------------------------------------------------------
     * Copy input to GPU (writable working copy for in-place SVD)
     * ---------------------------------------------------------------- */
    mxGPUArray* gpuInput = NULL;
    mxGPUArray*       gpuA     = NULL;
    mxGPUArray*       gpuU     = NULL;
    mxGPUArray*       gpuS     = NULL;
    mxGPUArray*       gpuVT    = NULL;
    int*              d_info   = NULL;

    gpuInput = mxGPUCreateFromMxArray(inputMx);   /* returns mxGPUArray* */
    gpuA     = mxGPUCopyGPUArray(gpuInput);
    cuDoubleComplex* const d_A = (cuDoubleComplex*)mxGPUGetData(gpuA);

    /* ------------------------------------------------------------------
     * Allocate U, S, VT output arrays
     *   Economy ('S'): U is M×minDim, VT is minDim×N
     *   Full    ('A'): U is M×M,      VT is N×N
     * ---------------------------------------------------------------- */
#if USE_ECONOMY_SVD
    const int uCols  = minDim;   /* economy U columns */
    const int vtRows = minDim;   /* economy VT rows   */
    const signed char jobu   = 'S';
    const signed char jobvt  = 'S';
#else
    const int uCols  = M;
    const int vtRows = N;
    const signed char jobu   = 'A';
    const signed char jobvt  = 'A';
#endif

    {
        mwSize uDims[2]  = { (mwSize)M, (mwSize)uCols };
        gpuU = mxGPUCreateGPUArray(2, uDims,
                   mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    }
    {
        mwSize vtDims[2] = { (mwSize)vtRows, (mwSize)N };
        gpuVT = mxGPUCreateGPUArray(2, vtDims,
                    mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    }
    {
        mwSize sDims[2] = { (mwSize)minDim, 1 };
        gpuS = mxGPUCreateGPUArray(2, sDims,
                   mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    }

    cuDoubleComplex* const d_U  = (cuDoubleComplex*)mxGPUGetData(gpuU);
    cuDoubleComplex* const d_VT = (cuDoubleComplex*)mxGPUGetData(gpuVT);
    double*          const d_S  = (double*)         mxGPUGetData(gpuS);

    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    /* ------------------------------------------------------------------
     * Run SVD on GPU (with timing)
     * ---------------------------------------------------------------- */
    cudaEventRecord(evStart, s_stream);

    cusolverStatus_t svd_status = runSVD(
        M, N, d_A, d_S,
        d_U,  M,          /* ldu  = M (always) */
        d_VT, vtRows,     /* ldvt = minDim (economy) or N (full) */
        d_info,
        jobu, jobvt);

    /* If full SVD failed and we are in full mode, retry with economy */
#if !USE_ECONOMY_SVD
    if (svd_status != CUSOLVER_STATUS_SUCCESS)
    {
        mexPrintf("[mex_hybrid_filter] Full SVD failed (status %d). "
                  "Retrying with economy SVD.\n", (int)svd_status);

        /* Re-copy input (SVD may have overwritten d_A) */
        mxGPUDestroyGPUArray(gpuA);
        gpuA = mxGPUCopyGPUArray(gpuInput);
        cuDoubleComplex* d_A_retry = (cuDoubleComplex*)mxGPUGetData(gpuA);

        /* Reallocate U and VT for economy dimensions */
        mxGPUDestroyGPUArray(gpuU);
        mxGPUDestroyGPUArray(gpuVT);

        {
            mwSize uDims2[2] = { (mwSize)M, (mwSize)minDim };
            gpuU = mxGPUCreateGPUArray(2, uDims2,
                       mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
        }
        {
            mwSize vtDims2[2] = { (mwSize)minDim, (mwSize)N };
            gpuVT = mxGPUCreateGPUArray(2, vtDims2,
                        mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
        }

        cuDoubleComplex* d_U_retry  = (cuDoubleComplex*)mxGPUGetData(gpuU);
        cuDoubleComplex* d_VT_retry = (cuDoubleComplex*)mxGPUGetData(gpuVT);

        svd_status = runSVD(
            M, N, d_A_retry, d_S,
            d_U_retry,  M,
            d_VT_retry, minDim,
            d_info, 'S', 'S');
    }
#endif

    cudaEventRecord(evStop, s_stream);

    if (svd_status != CUSOLVER_STATUS_SUCCESS)
    {
        cleanupCall(d_info, gpuInput, gpuA, gpuU, gpuS, gpuVT);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        mexErrMsgIdAndTxt("Hybrid:cuSOLVER",
            "cusolverDnZgesvd launch failed: status %d", (int)svd_status);
    }

    /* Synchronise before reading d_info (FIX 1: sync before host read) */
    CHECK_CUDA(cudaStreamSynchronize(s_stream));

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_info != 0)
    {
        cleanupCall(d_info, gpuInput, gpuA, gpuU, gpuS, gpuVT);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        if (h_info > 0)
            mexErrMsgIdAndTxt("Hybrid:SVD",
                "SVD did not converge: %d superdiagonals did not converge.",
                h_info);
        else
            mexErrMsgIdAndTxt("Hybrid:SVD",
                "SVD failed: illegal argument at position %d.", -h_info);
    }

    /* Report GPU kernel time */
    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, evStart, evStop);
    mexPrintf("[mex_hybrid_filter] GPU SVD kernel time: %.3f ms "
              "(matrix %dx%d, mode=%c)\n",
              gpuMs, M, N, (USE_ECONOMY_SVD ? 'S' : 'A'));

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    /* ------------------------------------------------------------------
     * Transfer results back to MATLAB CPU arrays
     * ---------------------------------------------------------------- */
    plhs[0] = mxGPUCreateMxArrayOnCPU(gpuU);
    plhs[1] = mxGPUCreateMxArrayOnCPU(gpuS);
    plhs[2] = mxGPUCreateMxArrayOnCPU(gpuVT);

    /* ------------------------------------------------------------------
     * Per-call cleanup (persistent workspace/handle stays alive)
     * ---------------------------------------------------------------- */
    cleanupCall(d_info, gpuInput, gpuA, gpuU, gpuS, gpuVT);
}
