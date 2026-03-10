% Final_Block_Processing.m
% BLOCK-WISE GPU SVD PIPELINE (Stable Memory Architecture)
% Paper: Memory-Aware Block-Wise SVD-Based Ultrasound Image Denoising Using CUDA
%
% Pipeline:
%   1. CPU : Load & normalize DICOM image
%   2. CPU : Divide image into overlapping blocks
%   3. GPU : SVD denoise ONE block at a time (mex_hybrid_filter)
%   4. CPU : Energy-based singular value truncation (95%)
%   5. CPU : Reconstruct via core-region stitching
%   6. CPU : Global post-processing (Log compression + Laplacian)
%   7. CPU : Compute evaluation metrics (PSNR, SSIM, ENL, SNR)
%
% Parameters (Paper Section V-B: Algorithm Configuration)
%   BLOCK_SIZE      = 256    pixels (256x256 processing unit)
%   OVERLAP         = 16     pixels (prevents edge artifacts)
%   CORE_SIZE       = 224    valid pixels per block (= 256 - 2*16)
%   ENERGY_THRESH   = 0.95   SVD energy threshold (95%)
%   LAPLACIAN_ALPHA = 0.06   Laplacian edge enhancement strength
%   N_TIMING_RUNS   = 10     runtime samples for mean/std timing
%   USE_ECONOMY_SVD = true   use economy ('S') SVD on non-square blocks
%
% Consistency with mex_hybrid_filter.cu:
%   - USE_ECONOMY_SVD must match #define USE_ECONOMY_SVD in the .cu file
%   - Complex double input to mex (gpuArray is NOT used here; CPU->MEX)
%   - S from MEX is min(M,N)x1 real; U/VT sizes depend on economy flag

clear; clc; close all;

% =========================================================================
% PARAMETERS  (Paper Section V-B: Algorithm Configuration)
% =========================================================================
BLOCK_SIZE      = 256;      % Block Size : 256x256 pixels
OVERLAP         = 16;       % Overlap Size: 16 pixels
CORE_SIZE       = BLOCK_SIZE - 2*OVERLAP;   % = 224 valid pixels per block
ENERGY_THRESH   = 0.95;     % SVD Energy Threshold: 95%
LAPLACIAN_ALPHA = 0.06;     % Laplacian Edge Enhancement alpha
N_TIMING_RUNS   = 10;       % Runtime averaged over 10 runs (Paper Sec V-C)

% Economy SVD flag — must match #define USE_ECONOMY_SVD in mex_hybrid_filter.cu
% true  -> MEX uses 'S' mode: U is M×minDim, VT is minDim×N  (~50% less VRAM)
% false -> MEX uses 'A' mode: U is M×M,      VT is N×N        (full matrices)
USE_ECONOMY_SVD = true;

% =========================================================================
% PARAMETER VALIDATION
% =========================================================================
assert(OVERLAP > 0, 'OVERLAP must be positive.');
assert(BLOCK_SIZE > 2*OVERLAP, ...
    'BLOCK_SIZE (%d) must be > 2*OVERLAP (%d).', BLOCK_SIZE, 2*OVERLAP);
assert(CORE_SIZE >= 8, ...
    'CORE_SIZE too small (%d). Reduce OVERLAP or increase BLOCK_SIZE.', CORE_SIZE);
assert(ENERGY_THRESH > 0 && ENERGY_THRESH <= 1, ...
    'ENERGY_THRESH must be in (0,1].');
assert(N_TIMING_RUNS >= 1, 'N_TIMING_RUNS must be >= 1.');

% =========================================================================
% MEX FILE PATH SETUP
% =========================================================================
mex_file = 'mex_hybrid_filter';
% --- Configure mex_dir to the folder that contains mex_hybrid_filter.mexw64
%     Use an environment variable when set, then fall back to a default.
%     Override this by setting the HYBRID_MEX_DIR environment variable, e.g.:
%       setenv('HYBRID_MEX_DIR', 'D:\MyProject\Hybrid')
mex_dir = getenv('HYBRID_MEX_DIR');
if isempty(mex_dir)
    mex_dir = 'C:\MATLAB\New folder\Hybrid_Project';   % default fallback path
end

if isempty(which(mex_file))
    if exist(mex_dir, 'dir')
        addpath(mex_dir);
        fprintf('Added to path: %s\n', mex_dir);
    else
        error('Hybrid:Path', ...
            ['mex_hybrid_filter not found.\n' ...
             'Edit mex_dir in this script to point to your project folder.\n' ...
             'Current mex_dir: %s'], mex_dir);
    end
end

if isempty(which(mex_file))
    error('Hybrid:Path', ...
        'mex_hybrid_filter still not found after adding path.\nCheck that the .mexw64 file exists in: %s', mex_dir);
else
    fprintf('MEX found: %s\n', which(mex_file));
end

% =========================================================================
% STEP 1: GPU INITIALIZATION
% =========================================================================
disp('=== Memory-Aware Block-Wise SVD Denoising ===');

try
    gpuDevice([]);           % Reset any stale GPU state
    gpu = gpuDevice();
    fprintf('GPU : %s | Free VRAM: %.1f MB\n', ...
        gpu.Name, gpu.AvailableMemory / 1e6);
    if gpu.AvailableMemory < 256e6
        error('Insufficient VRAM (< 256 MB). Cannot proceed.');
    end
catch ME
    error('GPU initialization failed: %s', ME.message);
end

% =========================================================================
% STEP 2: LOAD DICOM (Paper Section V-A)
% =========================================================================
disp('--- Loading DICOM ---');

[f, p] = uigetfile('*.dcm', 'Select DICOM Ultrasound');
if isequal(f, 0)
    disp('No file selected. Exiting.');
    return;
end
dicom_path = fullfile(p, f);
raw        = dicomread(dicom_path);

if size(raw, 3) == 3
    warning('RGB DICOM detected. Converting to grayscale.');
    raw = rgb2gray(raw);
end

if ndims(raw) == 4
    mid = ceil(size(raw, 4) / 2);
    raw = squeeze(raw(:, :, 1, mid));
    fprintf('Multi-frame DICOM: using frame %d of %d\n', mid, size(raw, 4));
elseif ndims(raw) == 3
    mid = ceil(size(raw, 3) / 2);
    raw = raw(:, :, mid);
    fprintf('3D DICOM: using slice %d of %d\n', mid, size(raw, 3));
end

original_image = double(raw);
img_min        = min(original_image(:));
img_max        = max(original_image(:));
original_image = (original_image - img_min) / (img_max - img_min + eps);

[img_rows, img_cols] = size(original_image);
fprintf('Image size: %d x %d pixels\n', img_rows, img_cols);

% =========================================================================
% STEP 3: COMPUTE RAW BASELINE METRICS
% =========================================================================
disp('--- Computing Raw Baseline Metrics ---');

% Robust homogeneous ROI selection with a 51x51 search window clamped to
% image bounds. Use a half-window of 25 so the ROI is always fully inside.
ROI_HALF    = 25;           % half-size of the ROI extraction window
SEARCH_WIN  = 15;           % window size passed to stdfilt

local_std_raw  = stdfilt(original_image, ones(SEARCH_WIN));

% Mask out the image border so the candidate centre is always valid
border = ROI_HALF + 1;
local_std_raw(1:border, :)          = Inf;
local_std_raw(end-border+1:end, :)  = Inf;
local_std_raw(:, 1:border)          = Inf;
local_std_raw(:, end-border+1:end)  = Inf;

[~, idx_raw]   = min(local_std_raw(:));
[r0, c0]       = ind2sub([img_rows, img_cols], idx_raw);

% Clamp centre (redundant but defensive)
r0 = max(ROI_HALF+1, min(img_rows-ROI_HALF, r0));
c0 = max(ROI_HALF+1, min(img_cols-ROI_HALF, c0));

roi_raw  = original_image(r0-ROI_HALF:r0+ROI_HALF, c0-ROI_HALF:c0+ROI_HALF);
enl_raw  = mean(roi_raw(:))^2 / (var(roi_raw(:)) + eps);

% Noise estimate for SNR baseline (std in raw ROI)
noise_raw = std(roi_raw(:)) + eps;
snr_raw   = 20 * log10(mean(roi_raw(:)) / noise_raw);

fprintf('Raw ENL  : %.4f\n', enl_raw);
fprintf('Raw SNR  : %.2f dB\n', snr_raw);

% =========================================================================
% STEP 4: BLOCK-WISE SVD PROCESSING
% =========================================================================
disp('--- Starting Block-Wise SVD Processing ---');

% --- Edge-case: image smaller than one block (no blocking needed) ---
if img_rows < CORE_SIZE || img_cols < CORE_SIZE
    warning('Image (%dx%d) is smaller than CORE_SIZE (%d). Skipping blocking.', ...
        img_rows, img_cols, CORE_SIZE);
    total_blocks  = 1;           % single block — used in display later
    timing_times  = zeros(1, N_TIMING_RUNS);

    for timing_run = 1:N_TIMING_RUNS
        block_complex = complex(original_image);
        gpu = gpuDevice(); wait(gpu);
        t_run = tic;
        [U_nb, S_nb, VT_nb] = mex_hybrid_filter(block_complex);
        wait(gpuDevice());
        timing_times(timing_run) = toc(t_run);
    end

    S_vals = real(S_nb(:));
    energy = cumsum(S_vals.^2) / (sum(S_vals.^2) + eps);
    rank_k = find(energy >= ENERGY_THRESH, 1);
    if isempty(rank_k), rank_k = length(S_vals); end
    S_filt = zeros(size(S_vals));
    S_filt(1:rank_k) = S_vals(1:rank_k);
    if USE_ECONOMY_SVD
        recon = real(U_nb * diag(S_filt) * VT_nb);
    else
        recon = real(U_nb(:,1:length(S_filt)) * diag(S_filt) * VT_nb(1:length(S_filt),:));
    end
    % Crop to original size (MEX output matches input dimensions; defensive clamp)
    out_r = min(size(recon,1), img_rows);
    out_c = min(size(recon,2), img_cols);
    output_image = zeros(img_rows, img_cols);
    output_image(1:out_r, 1:out_c) = recon(1:out_r, 1:out_c);
else
    % --- Pad image so it tiles exactly by CORE_SIZE ---
    pad_rows = mod(CORE_SIZE - mod(img_rows, CORE_SIZE), CORE_SIZE);
    pad_cols = mod(CORE_SIZE - mod(img_cols, CORE_SIZE), CORE_SIZE);
    padded   = padarray(original_image, [pad_rows, pad_cols], 0, 'post');
    [pad_H, pad_W] = size(padded);

    output_image = zeros(pad_H, pad_W);

    row_starts = 1 : CORE_SIZE : (pad_H - CORE_SIZE + 1);
    col_starts = 1 : CORE_SIZE : (pad_W - CORE_SIZE + 1);
    total_blocks = numel(row_starts) * numel(col_starts);

    % Block count sanity check
    assert(total_blocks >= 1, ...
        'Block count validation failed: no blocks created. Check CORE_SIZE/padding logic.');

    fprintf('Total blocks to process: %d\n', total_blocks);

    timing_times = zeros(1, N_TIMING_RUNS);

    for timing_run = 1:N_TIMING_RUNS

        output_temp = zeros(pad_H, pad_W);
        block_count = 0;

        % Flush pending GPU work; then start timer
        gpu = gpuDevice();
        wait(gpu);
        t_run = tic;

        for r = row_starts
            for c = col_starts

                block_count = block_count + 1;

                % --- GPU memory guard: check VRAM before each block ---
                gpu = gpuDevice();
                if gpu.AvailableMemory < 128e6
                    warning('Low VRAM (%.0f MB). Resetting GPU.', ...
                        gpu.AvailableMemory/1e6);
                    try
                        reset(gpuDevice);
                    catch
                        gpuDevice([]);
                    end
                end

                % --- Extract overlapping block ---
                r_block_start = max(1,     r - OVERLAP);
                r_block_end   = min(pad_H, r + CORE_SIZE - 1 + OVERLAP);
                c_block_start = max(1,     c - OVERLAP);
                c_block_end   = min(pad_W, c + CORE_SIZE - 1 + OVERLAP);

                block = padded(r_block_start:r_block_end, ...
                               c_block_start:c_block_end);

                % --- GPU SVD via MEX (complex double) ---
                block_complex = complex(block);

                try
                    [U, S_nb, VT] = mex_hybrid_filter(block_complex);
                catch svd_err
                    warning('MEX SVD failed for block (%d,%d): %s. Using CPU SVD fallback.', ...
                        r, c, svd_err.message);
                    [U, S_mat, V_cpu] = svd(block, 'econ');
                    S_nb = diag(S_mat);
                    VT   = V_cpu';
                end

                % --- Energy-based rank selection ---
                S_vals  = real(S_nb(:));
                energy  = cumsum(S_vals.^2) / (sum(S_vals.^2) + eps);
                rank_k  = find(energy >= ENERGY_THRESH, 1);
                if isempty(rank_k), rank_k = length(S_vals); end

                S_filtered           = zeros(size(S_vals));
                S_filtered(1:rank_k) = S_vals(1:rank_k);

                % --- Reconstruct block (economy vs full SVD) ---
                if USE_ECONOMY_SVD
                    % U: bH×minDim, VT: minDim×bW
                    denoised_block = real(U * diag(S_filtered) * VT);
                else
                    % U: bH×bH, VT: bW×bW — truncate to minDim columns/rows
                    minDim = length(S_filtered);
                    denoised_block = real(U(:,1:minDim) * diag(S_filtered) * VT(1:minDim,:));
                end

                % --- Map core region back to output grid ---
                local_r1 = r - r_block_start + 1;
                local_r2 = local_r1 + CORE_SIZE - 1;
                local_c1 = c - c_block_start + 1;
                local_c2 = local_c1 + CORE_SIZE - 1;

                local_r2 = min(local_r2, size(denoised_block, 1));
                local_c2 = min(local_c2, size(denoised_block, 2));

                out_r2 = r + (local_r2 - local_r1);
                out_c2 = c + (local_c2 - local_c1);

                out_r2 = min(out_r2, pad_H);
                out_c2 = min(out_c2, pad_W);

                output_temp(r:out_r2, c:out_c2) = ...
                    denoised_block(local_r1:local_r2, local_c1:local_c2);

                % --- Progress display (timing run 1 only to avoid clutter) ---
                if timing_run == 1
                    pct = block_count / total_blocks * 100;
                    fprintf('  Block %d/%d (%.1f%%)\r', block_count, total_blocks, pct);
                end

            end
        end

        % Wait for GPU to finish, then stop timer
        wait(gpuDevice());
        timing_times(timing_run) = toc(t_run);

        if timing_run == 1
            fprintf('\n');           % newline after \r progress
            output_image = output_temp;
        end

    end

    fprintf('Mean runtime : %.4f s (std: %.4f s) over %d runs\n', ...
        mean(timing_times), std(timing_times), N_TIMING_RUNS);

    output_image = output_image(1:img_rows, 1:img_cols);
end

% =========================================================================
% STEP 5: GLOBAL POST-PROCESSING
% =========================================================================
disp('--- Applying Global Post-Processing ---');

output_log = log1p(output_image);
output_log = output_log / (max(output_log(:)) + eps);

laplacian_kernel = [0  1  0;
                    1 -4  1;
                    0  1  0];
edge_map     = imfilter(output_log, laplacian_kernel, 'replicate');
output_sharp = output_log - LAPLACIAN_ALPHA * edge_map;

output_final = max(0, min(1, output_sharp));

% =========================================================================
% STEP 6: EVALUATION METRICS
% =========================================================================
disp('--- Computing Evaluation Metrics ---');

psnr_val = psnr(output_final, original_image);
ssim_val = ssim(output_final, original_image);

% ENL — use the same homogeneous ROI identified in the raw image
roi_den  = output_final(r0-ROI_HALF:r0+ROI_HALF, c0-ROI_HALF:c0+ROI_HALF);
enl_den  = mean(roi_den(:))^2 / (var(roi_den(:)) + eps);
enl_imp  = (enl_den - enl_raw) / (enl_raw + eps) * 100;

% SNR (signal = mean of ROI, noise = std of ROI)
noise_den = std(roi_den(:)) + eps;
snr_den   = 20 * log10(mean(roi_den(:)) / noise_den);
snr_imp   = snr_den - snr_raw;

fprintf('\n========== RESULTS (Paper Table I) ==========\n');
fprintf('Metric | Raw         | Denoised    | Improvement\n');
fprintf('-------|-------------|-------------|-------------\n');
fprintf('PSNR   | --          | %.2f dB     | --\n',                 psnr_val);
fprintf('SSIM   | --          | %.4f        | --\n',                  ssim_val);
fprintf('ENL    | %.2f        | %.2f        | +%.1f%%\n',             enl_raw, enl_den, enl_imp);
fprintf('SNR    | %.2f dB     | %.2f dB     | %+.2f dB\n',           snr_raw, snr_den, snr_imp);
fprintf('=============================================\n');
fprintf('Runtime: %.4f s (mean over %d runs)\n', mean(timing_times), N_TIMING_RUNS);

% =========================================================================
% STEP 7: DISPLAY RESULTS
% =========================================================================
figure('Name', 'Block-Wise SVD Denoising Results', 'NumberTitle', 'off');

subplot(1, 3, 1);
imshow(original_image, []);
title(sprintf('Original\nENL: %.2f | SNR: %.1f dB', enl_raw, snr_raw));

subplot(1, 3, 2);
imshow(output_final, []);
title(sprintf('Denoised (Blocks=%d)\nENL: %.2f (+%.1f%%)', ...
    total_blocks, enl_den, enl_imp));

subplot(1, 3, 3);
diff_img = abs(original_image - output_final);
imshow(diff_img, []);
title(sprintf('Difference Map\nPSNR: %.2f | SSIM: %.4f', psnr_val, ssim_val));

sgtitle('Memory-Aware Block-Wise SVD Ultrasound Denoising');

% =========================================================================
% STEP 8: SAVE OUTPUT
% =========================================================================
[~, fname, ~] = fileparts(dicom_path);
out_dir  = p;   % default: same directory as input DICOM

% Automatically create output directory if it doesn't exist
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created output directory: %s\n', out_dir);
end

out_path = fullfile(out_dir, sprintf('%s_denoised.png', fname));
imwrite(output_final, out_path);
fprintf('\nOutput saved: %s\n', out_path);

disp('--- Processing Complete ---');
