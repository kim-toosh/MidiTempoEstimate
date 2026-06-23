"""Global configuration constants for the MIDI tempo estimation system."""

N_PARTICLES: int = 500

TEMPO_MIN: float = 30.0   # BPM
TEMPO_MAX: float = 250.0  # BPM
SIGMA_TEMPO: float = 0.5  # BPM, transition noise

SIGMA_OBS_RATIO: float = 0.05  # observation noise as fraction of beat period

# Extended beat ratio set covering 16th notes, triplets, and longer values
BEAT_RATIOS: list[float] = [
    0.25,   # 16th note
    0.333,  # 8th-note triplet
    0.5,    # 8th note
    0.667,  # quarter-note triplet
    0.75,   # dotted 8th note
    1.0,    # quarter note (1 beat)
    1.5,    # dotted quarter note
    2.0,    # half note
    3.0,    # dotted half note
    4.0,    # whole note
]
BEAT_RATIO_WEIGHTS: list[float] = [
    0.10,   # 16th
    0.05,   # 8th triplet
    0.15,   # 8th
    0.05,   # quarter triplet
    0.05,   # dotted 8th
    0.30,   # quarter (most important)
    0.10,   # dotted quarter
    0.10,   # half
    0.05,   # dotted half
    0.05,   # whole
]

ESS_THRESHOLD_RATIO: float = 0.5  # resample when ESS < N_PARTICLES * this value

METER_NUMERATOR: int = 4   # beats per measure (4/4 time)

TEMPO_PRECISION: int = 2   # decimal places for BPM output (0.01 BPM resolution)

# ── Drum instrument identification ────────────────────────────────────────────
DRUM_CHANNEL: int = 9       # 0-indexed (MIDI channel 10)
USE_DRUM_WEIGHTS: bool = True  # False → uniform weight for all instruments

# ── Autocorrelation-based initial tempo estimator ─────────────────────────────
AUTOCORR_BUFFER_SIZE: int = 16         # number of recent events to keep
AUTOCORR_MIN_EVENTS: int = 6           # minimum events before estimation starts
AUTOCORR_N_PEAKS: int = 3             # number of top BPM candidates to return
AUTOCORR_TEMPO_MIN: float = 30.0           # BPM lower bound for autocorr search
AUTOCORR_TEMPO_MAX: float = 250.0          # BPM upper bound for autocorr search
AUTOCORR_CONFIDENCE_THRESHOLD: float = 0.5   # reserved for future use
CONVERGENCE_THRESHOLD: float = 0.65   # particle filter confidence above which
                                       # autocorr seeding is disabled

# ── Kalman gate ───────────────────────────────────────────────────────────────
KALMAN_Q: float = 1.0           # process noise (BPM²/step)
KALMAN_R: float = 4.0           # observation noise (BPM²)
KALMAN_INIT_VAR: float = 100.0  # initial state variance (BPM²)
KALMAN_INIT_TEMPO: float = 120.0  # initial Kalman mean (BPM)
KALMAN_GATE_SIGMA: float = 3.0  # Mahalanobis gate threshold = SIGMA²
KALMAN_MIN_CONFIDENCE: float = 0.5  # particle-filter confidence below which candidate is rejected
OCTAVE_RATIOS: list[float] = [0.5, 2.0, 0.25, 4.0]  # octave-jump ratio suspects
OCTAVE_TOLERANCE: float = 0.08  # relative tolerance for octave ratio check
KALMAN_SEED_STREAK: int = 5        # consecutive rejects before Kalman feedback seeding starts
KALMAN_SEED_SPREAD_BPM: float = 3.0  # std-dev of Kalman-guided seeds (BPM)

# ── 楽器カテゴリ別観測モデル ────────────────────────────────────────────────
# Kick
KICK_BEAT_RATIOS:  list[float] = [0.5, 1.0, 2.0, 3.0, 4.0]
KICK_BEAT_WEIGHTS: list[float] = [0.20, 0.35, 0.25, 0.10, 0.10]
KICK_DRUM_WEIGHT:  float = 1.0

# Snare
SNARE_BEAT_RATIOS:  list[float] = [0.5, 1.0, 2.0, 3.0]
SNARE_BEAT_WEIGHTS: list[float] = [0.10, 0.30, 0.50, 0.10]
SNARE_DRUM_WEIGHT:  float = 0.9

# HiHat (ratio >= 1.0 は使わない — テンポオクターブ対策)
# ratio=0.5 weight は低く抑える: 16th-note IOI が 240 BPM の ratio=0.5 に一致してしまうため
# HIHAT_SIGMA_OBS_RATIO を広めに取る: jitter による外れ値 IOI (例: 0.1s) が
# 高テンポ (160-200 BPM) で完全一致を引き起こしても 120 BPM を消さないため
HIHAT_BEAT_RATIOS:      list[float] = [0.25, 0.333, 0.5, 0.667, 0.75]
HIHAT_BEAT_WEIGHTS:     list[float] = [0.25, 0.10,  0.35, 0.15, 0.15]
HIHAT_DRUM_WEIGHT:      float = 0.05
HIHAT_SIGMA_OBS_RATIO:  float = 0.10

# Others (Crash / Tom / その他)
OTHERS_BEAT_RATIOS:  list[float] = [0.5, 1.0, 2.0, 3.0]
OTHERS_BEAT_WEIGHTS: list[float] = [0.20, 0.50, 0.20, 0.10]
OTHERS_DRUM_WEIGHT:  float = 0.4

# Crash はダウンビート指標として特別扱い
CRASH_NOTE_NUMBERS: list[int] = [49, 52, 55, 57]
CRASH_DRUM_WEIGHT:  float = 0.8

# ── 近似GCD推定器 ─────────────────────────────────────────────────────────────
GCD_MIN_EVENTS:           int   = 4      # 推定を開始する最低イベント数（Kick+Snare+HiHat合算）
GCD_BUFFER_SIZE:          int   = 6      # タイムスタンプの保持数
GCD_RESOLUTION:           float = 0.002  # 候補GCDの刻み幅（秒）
GCD_TOLERANCE:            float = 0.15   # 残差の許容割合
GCD_CONFIDENCE_THRESHOLD: float = 0.70   # この値以上で粒子初期化に使用
GCD_REFINE_ITER:          int   = 3      # refinementの反復回数
GCD_REINIT_SIGMA:         float = 3.0    # 粒子再初期化時のガウス分布のσ [BPM]

# gcdは観測IOIに共通する最小リズム単位の周期であり、1拍の周期(beat)とは限らない
# （16th/8th/quarter/half/whole noteのいずれかになり得る）。
# gcd周期 * GCD_OCTAVE_RATIOS で得られる周期候補のうち、KalmanGateの現在の
# テンポ推定に最も近いものをbeatテンポとして採用する（particle_filter.py）。
# 最大倍率(=16)は、approx_gcdの探索下限(g_min)と最終レンジチェックの上限
# (TEMPO_MAX*max(GCD_OCTAVE_RATIOS))の算出にも使用する（approx_gcd.py）。
GCD_OCTAVE_RATIOS: list[int] = [1, 2, 3, 4, 6, 8, 16]

# ── 同時イベントのグルーピング ───────────────────────────────────────────────
SPAN_SAME_TIME_SEC: float = 0.07  # この時間内の異なるノートのイベントを同一タイミングとみなす [秒]

# ── イベントタイムアウト ─────────────────────────────────────────────────────
# 前回イベントからこの時間を超えた場合、GCDタイムスタンプバッファをクリアする
# （長い無音区間を挟むと、空白期間がIOIとして混入しGCD推定が破綻するため）。
# 初期値: BPM50, 4/4拍子で5小節分 = 60/50 * 4 * 5 = 24.0 秒
EVENT_TIMEOUT_SEC: float = 24.0

# ── Phase-Coupled Oscillator ─────────────────────────────────────────────────
PCO_ETA_PHASE        : float = 0.15   # 位相同期の強さ（Others/デフォルト）
PCO_ETA_PHASE_STRONG : float = 0.25   # Kick/Snare（強拍系）
PCO_ETA_PHASE_WEAK   : float = 0.05   # HiHat/Others（弱拍系）
PCO_SYNC_THRESHOLD   : float = 0.10   # 平均位相誤差がこの値以内なら同期確立
PCO_PREDICTION_ENABLE: bool  = True   # 次Beat予測を有効にするか

# ── 近似GCD：ratio候補（3連符系を含む統合リスト）─────────────────────────────
GCD_RATIOS: list[float] = [1/6, 0.25, 1/3, 0.5, 2/3, 1.0, 1.5, 2.0]

# ── GCD 複数候補出力 ─────────────────────────────────────────────────────────
GCD_N_CANDIDATES: int = 4          # 上位何件を候補として保持・出力するか
GCD_CANDIDATE_MIN_GAP: float = 0.05  # 候補間の最小差（秒）
