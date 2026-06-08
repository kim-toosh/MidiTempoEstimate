# MIDI Tempo Estimator

Particle Filter (Sequential Monte Carlo) を使ったリアルタイム MIDI テンポ推定システム。  
MIDI NOTE_ON イベントのタイムスタンプから BPM・ビート位相・拍子位置をリアルタイムに推定する。

## 特徴

- **NumPy ベクトル演算**: パーティクル全体の処理を NumPy で一括実行（Python ループなし）
- **Explorer mixture**: パーティクルの 2% を毎ステップ一様事前分布にリセットし、急激なテンポ変化に追従
- **系統的リサンプリング**: 単一一様乱数を使う分散の小さいリサンプリング
- **PyQt6 GUI**: ダークテーマ、Tempo(BPM) グラフ、MIDI / Mock モード切り替え
- **自動接続**: 起動時に最初の MIDI ポートへ自動接続

## 動作要件

- Python 3.9 以上
- MIDI 入力デバイス（GUI の Mock モードを使う場合は不要）

## セットアップ

```bash
# 1. リポジトリをクローン
git clone git@github.com:kim-toosh/MidiTempoEstimate.git
cd MidiTempoEstimate

# 2. 仮想環境を作成・有効化
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. 依存パッケージをインストール
pip install -r requirements.txt
```

## 実行方法

### GUI（推奨）

```bash
source .venv/bin/activate
python -m midi_tempo_hmm.main_gui
```

- 起動時に最初の MIDI ポートへ自動接続・START
- 左パネル: TEMPO / BEAT / MEASURE / CONFIDENCE の現在値
- 右パネル: Tempo(BPM) の時系列グラフ（Y 軸 25〜300 固定）
- STOP / RESET ボタンで再起動なしにリセット可能

#### Mock モード（MIDI デバイス不要）

```bash
python -m midi_tempo_hmm.main_gui --mode mock --tempo 120
```

指定 BPM の仮想イベントを自動生成してフィルタを動かす。動作確認・デモに使用する。

### CLI リアルタイム

```bash
# MIDI ポートの一覧を確認
python -m midi_tempo_hmm.main_realtime --list-ports

# ポートインデックス 0 に接続して推定開始
python -m midi_tempo_hmm.main_realtime --port-index 0
```

Ctrl+C で停止。

## テスト

```bash
source .venv/bin/activate
pytest
```

| テスト | 内容 |
|---|---|
| `test_steady_tempo_120` | 120 BPM に 0.5 BPM 以内で収束 |
| `test_steady_tempo_various` | 60〜180 BPM の各テンポで 1.0 BPM 以内 |
| `test_tempo_change` | 120→140 BPM の急変に 3.0 BPM 以内で追従 |
| `test_humanized_input` | ±10 ms ジッターで 1.0 BPM 以内 |
| `test_beat_position_accuracy` | `beat_position` が常に [0.0, 1.0) 内 |
| `test_single_event_time` | 1 イベントの処理時間が 10 ms 未満 |
| `test_batch_event_time` | バッチ平均 5 ms 未満、最大 10 ms 未満 |

## 設定パラメータ

設定は `midi_tempo_hmm/config.py` で変更できる。

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `N_PARTICLES` | 500 | パーティクル数。多いほど精度向上、処理時間増加 |
| `TEMPO_MIN` / `TEMPO_MAX` | 30 / 250 BPM | 推定テンポの範囲 |
| `SIGMA_TEMPO` | 0.5 BPM | 遷移ノイズ（テンポのランダムウォーク標準偏差） |
| `SIGMA_OBS_RATIO` | 0.05 | 観測ノイズ（ビート周期に対する比率） |
| `BEAT_RATIOS` | [0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0] | IOI 候補比率（サブ・スーパービート） |
| `BEAT_RATIO_WEIGHTS` | [0.1, 0.1, 0.4, 0.15, 0.15, 0.05, 0.05] | 各比率の事前重み |
| `ESS_THRESHOLD_RATIO` | 0.5 | ESS が `N_PARTICLES × この値` を下回るとリサンプル |
| `METER_NUMERATOR` | 4 | 拍子の分子（4/4 拍子なら 4） |

## プロジェクト構成

```
midi_tempo_hmm/
├── config.py             # 設定定数
├── main_gui.py           # PyQt6 GUI エントリポイント
├── main_realtime.py      # CLI リアルタイムエントリポイント
├── main_poc.py           # Proof-of-concept スクリプト
├── core/
│   ├── particle_filter.py  # Particle Filter メインクラス
│   ├── observation.py      # 尤度計算（IOI 観測モデル）
│   ├── resampling.py       # 系統的リサンプリング・ESS 計算
│   └── state.py            # パーティクル状態定義
├── interface/
│   ├── midi_input.py       # python-rtmidi MIDI 入力ハンドラ
│   └── mock_input.py       # テスト用仮想イベント生成
├── output/
│   └── estimator_result.py # 推定結果データクラス
└── tests/
    ├── test_accuracy.py    # 精度テスト
    └── test_performance.py # 処理時間テスト
```
