# microduck-tidy-up

小型二足ロボット Microduck(Pollen Robotics)が、頭のカメラで床の物を探し、歩いて近づき、くちばしで拾って片付けエリアへ運ぶ。
Physical AI 応用 1 講座の最終課題として作成した。すべて MuJoCo シミュレーション上で動作する。

- デモ動画:(YouTube の URL)
- 説明資料:(提出した PDF と同じ内容)

| 結果 | 値 |
|---|---|
| 片付け(物 3 個 × 配置 3 通り) | 9/9 |
| 拾う動作(立方体・球・寝かせた直方体の小物 6 種、各 10 回) | 59/60 |

上流の [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) は **一切編集せず**、ライブラリとして読み込んで使う。
このリポジトリに含まれるのは、本課題で作成した部分だけである。

| 区分 | 内容 |
|---|---|
| 上流から利用 | ロボットの 3D モデル、サーボ特性モデル(BAM)、歩行・しゃがみの学習タスク、学習基盤(mjlab + PPO)、学習済みの歩行・立ちポリシー |
| 本リポジトリ(学習) | BeakPick:くちばしで物を拾うタスク / SearchWalk:うつむいて歩くための歩行の追加学習 |
| 本リポジトリ(その他) | くちばし関節の追加、前向きカメラと物体の位置推定、探索から片付けまでの状態機械、把持候補の物体 14 種、評価・動画作成スクリプト |

## セットアップ

Linux(WSL2 可)、CUDA GPU(学習時のみ)、[uv](https://docs.astral.sh/uv/) が必要。

```bash
# 1. 上流を隣に置き、その環境を作る(このリポジトリと同じ階層に microduck_rl/ を置く)
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl && git checkout 29e887e
UV_PROJECT_ENVIRONMENT=~/.venvs/microduck_rl uv sync
cd ..

# 2. 公式の歩行・立ちポリシー(alpha_walking.onnx, alpha_stand.onnx)を policies/ に置く
#    入手先: Hugging Face の pollen-robotics/microduck-policies
#    別の場所に置く場合は MICRODUCK_POLICIES=<そのディレクトリ> を指定する

# 3. このリポジトリ
git clone <このリポジトリ> microduck-tidy-up
cd microduck-tidy-up
```

以下のコマンドはすべて `microduck-tidy-up/` 直下で実行する。
`run.sh` は上流の venv(`MICRODUCK_VENV`、既定 `~/.venvs/microduck_rl`)の python をそのまま使う。上流環境へは何もインストールしない。
上流を別の場所に置いた場合は `MICRODUCK_RL=<そのパス>` を指定する。

```bash
# 探して拾って片付けるデモ(拾う ONNX, 出力動画, 配置の乱数, 物体数, 最大秒数, 置き位置の幅, 歩行 ONNX)
bash run.sh python scripts/fetch_demo.py onnx/v2_1000.onnx videos/fetch_demo.mp4 1 3 180 narrow onnx/sw2_final.onnx

# 候補物体ごとの拾う成功率(試行数, 物体キー, 動画出力先)
bash run.sh python scripts/pick_eval.py onnx/v2_1000.onnx 10 all -

# 候補 14 種を 1 回ずつ拾う一覧動画
bash run.sh python scripts/make_montage.py onnx/v2_1000.onnx videos/montage.mp4 all 0 4
```

`onnx/` には学習済みポリシーを同梱している。

| ファイル | 中身 |
|---|---|
| `v2_1000.onnx` | BeakPick 1000 反復(デモで使用) |
| `v2_final.onnx` | BeakPick 4000 反復(比較用) |
| `v3wide_final.onnx` | 置き位置を広げた追加学習(比較用) |
| `sw2_final.onnx` | SearchWalk 最終版(デモで使用) |

## くちばしの関節

実機は 15 サーボで、15 個目が口(`mouth`)。上流の学習モデルは口を頭に固定した 14 サーボ構成。
このリポジトリでは `MjSpec` で読み込み時に下顎を別ボディに切り出し、ヒンジ `passive_mouth` を追加する。

| 項目 | 内容 |
|---|---|
| ヒンジ位置 | 口サーボ出力軸と反対側ベアリングを結ぶ左右軸(CAD メッシュから実測) |
| 可動域 | 0〜0.70 rad(0 = CAD の閉状態) |
| 駆動(CPU 推論) | MuJoCo アクチュエータは追加しない。`BeakServo` が `qfrc_applied` でトルク制限付き PD をかける。上流の推論コードは `model.nu`(14)を関節数として使うため |
| 駆動(GPU 学習) | `add_beak(actuator=True)` で位置アクチュエータを追加。ポリシーのアクションからは除外し、位相フックが目標角を書く |
| 制御則 | kp 0.6 N·m/rad、kd 0.01、トルク上限 0.25 N·m(学習・推論で共通) |
| qpos 上の位置 | `head_roll` の直後(実機の関節順と同じ) |
| 総質量 | 不変(下顎 8 g を頭から移す) |
| 接触 | 下顎メッシュの凸包は口腔を埋めるため、上下の内面に薄い箱パッドを置く |
| 衝突ビット | 物体は bit 4 を使う。頭の下殻の凸包は口腔に張り出すため、物体とだけ当たらないようにする |

| 検証 | 結果 |
|---|---|
| 歩行への影響(alpha_walking, vx=0.3, 8 s, 平地) | 上流 0.702 m / くちばし付き 0.693 m、転倒なし |
| 固定胴体でくわえて首振り・下向き | 15・20 mm 立方体(10〜30 g)は保持。25 mm 以上は落下 |
| くわえたまま歩行(20 mm 立方体) | 10/20/30 g、前進・旋回・部屋シーンで全て保持、転倒なし |
| 上流のしゃがみポリシーで床から拾う | 失敗。口先が床上約 27 mm で止まる |

```bash
bash run.sh python scripts/walk_carry.py carry20 0.3 8 - 0 0 flat    # くわえたまま歩く
bash run.sh python scripts/grasp_fixed.py 0.020 0.010 0.5             # 固定胴体での把持
bash run.sh python scripts/pick_probe.py 0.072 0.010 0.05 0.36        # 上流しゃがみで拾う試行
```

## BeakPick タスク(`Mjlab-BeakPick-Flat-MicroDuck`)

しゃがむ → くちばしを開いて物体を挟む → 閉じる → 持ったまま立ち上がる、を 1 サイクル 5 秒で学習する。
上流の GroundPick タスクを土台に、観測 61 次元・行動 14 次元の共通形式はそのまま。

| 位相 | 内容 |
|---|---|
| 0.00〜0.30 | しゃがむ(0.08 でくちばしを開く) |
| 0.30〜0.45 | 底で待機(0.36 でくちばしを閉じる) |
| 0.45〜0.80 | くわえたまま立ち上がる |
| 0.80〜1.00 | 立って保持 |

| 項目 | 設計 |
|---|---|
| くちばし | ポリシーは動かさない。位相で開閉(実機のデーモンと同じ役割分担) |
| 物体 | 20 mm 立方体。足先前方 70 mm に ±15/20 mm・ヨー ±34° のばらつき。質量 4〜30 g、摩擦 0.5〜1.2 をランダム化 |
| 物体位置の入力 | 上流で未使用(ゼロ埋め)の body_pose コマンド欄に、規定位置からのずれ (dx, dy) を入れる |
| 主な報酬 | 下降中: 把持点と物体中心の距離(幅 15/4/1.2 cm の 3 段)。上昇中: 把持点に物体がある状態での持ち上げ高さ・保持 |
| 継承した報酬 | 立ち上がり姿勢・胴体の直立・両足接地・足の平坦・動作の滑らかさ・口先を床へ(重み 1 に軽減) |
| 批評側だけの観測 | 物体の位置・速度、くちばし開度 |

```bash
# 学習(RTX 2070 で 2048 環境・1 反復約 3.9 秒)
bash run.sh python scripts/mjlab_cli.py train Mjlab-BeakPick-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 4000 --agent.logger tensorboard --agent.run-name v2

# ONNX 書き出し(正規化層込み)
bash run.sh python scripts/mjlab_cli.py export Mjlab-BeakPick-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/beak_pick/<run>/model_N.pt
```

拾う成功率(v2_1000、指令遅延 20 ms、各 10 回)。

| 物 | 成功 |
|---|---|
| 木製キューブ 20 mm、サイコロ 16 mm、丸めた紙、消しゴム(横置き)、LEGO 2×2 を 2 段 | 各 10/10 |
| LEGO 2×4(1 段) | 9/10 |
| ミニカー | 5/10 |
| ワインコルク(立て)、スポンジ片 | 各 1/10 |
| 単 3 電池(立て)、ペンのキャップ(立て)、限界例 3 種 | 0/10 |

## 探して拾って片付ける(Fetch)

`src/microduck_tidy/fetch.py` の状態機械が、学習済みのポリシーを切り替えて使う。

| 層 | 中身 |
|---|---|
| 知覚 | 頭の前向きカメラ(上流の head_camera は頭の内側で後ろ向きのため新設、垂直画角 60°)の領域分割+深度から物体の位置を推定。誤差は 1 m 先でも 9 mm 以内 |
| 歩行 | SearchWalk ポリシー(下記)。速度と首の角度を指令 |
| 拾う | BeakPick ポリシー。物体の推定位置からのずれを入力 |
| 状態機械 | 探索(円を描いて歩く)→ 接近 → 最終接近(舵を取りながら低速)→ 位置確認 → 拾う → 運ぶ → くちばしを開く → 次へ |

### SearchWalk タスク(`Mjlab-SearchWalk-Flat-MicroDuck`)

上流の歩行タスクを、うつむいたまま歩く用途に合わせて追加学習したもの。

| 変更 | 理由 |
|---|---|
| 首の指令をうつむき側に偏らせる(head_pitch -0.1〜0.95 rad) | 上流の歩行は頭を 0.85 下げると一歩も歩けなかった |
| その場旋回の練習環境を 15% → 30% | 探索で向きを変えるため |
| 旋回速度の追従報酬を厳しく(標準偏差 0.71 → 0.39 rad/s、重み 2.0 → 3.5) | 上流の設定では、回らなくても報酬の約 4 割が入る |

その場旋回は最後まで学習できなかった。探索と位置合わせは「歩きながら曲がる」で代用している。

### 位置合わせに必要だった実測

| 項目 | 値 |
|---|---|
| 歩き出す最低速度 | 0.3 m/s(0.25 は歩くか止まるかの境目) |
| 0.3 m/s を 0.5 秒 | 前に約 6 cm |
| -0.3 m/s を 1 秒 | 後ろに約 13 cm |
| 拾える物体のずれ(v2_1000、木製キューブ) | 前後 -30〜+30 mm、横 -40〜0 mm が中心(`scripts/pick_window_map.py`) |
| CPU 推論の指令遅延 | 1 ステップ(20 ms)。学習時のアクチュエータ遅延に合わせないと、前進中に 117° 曲がる |

### 使うポリシーの比較(物 3 個 × 配置 3 通り)

| 組み合わせ | 片付け | 合計時間 |
|---|---|---|
| 拾う v2_1000 + 歩行 sw2_final | 9/9 | 330 秒(104 / 116 / 110) |
| 拾う v2_1000 + 歩行 sw2_6000 | 9/9 | 363 秒 |
| 拾う v3wide_final + 歩行 sw2_final | 8/9 | 342 秒(1 配置は時間切れ) |

v3wide は、置き位置を広げて追加学習したのに、木製キューブで拾える範囲は狭くなった(格子 30 点中 5 点、v2_1000 は 15 点)。
元にした 4000 反復版で、くわえた直後に勢いよく立ち上がり物を落とす癖がついており、それを引き継いだと考えられる。

## 構成

```
src/microduck_tidy/beak.py          くちばし関節の追加(add_beak)とスクリプト制御サーボ(BeakServo)
src/microduck_tidy/sim.py           BAM サーボ付きローダー(上流 infer_policy と同じ処理+くちばし)、物体
src/microduck_tidy/props.py         把持候補の物体カタログ
src/microduck_tidy/pick_profile.py  BeakPick の位相・規定位置(学習と推論で共有)
src/microduck_tidy/runner.py        CPU 推論ランナー(歩行・立ちポリシー+BeakPick+くちばし、指令遅延 20 ms)
src/microduck_tidy/fetch.py         知覚(前向きカメラ)と探索〜片付けの状態機械
src/microduck_tidy/tasks/           BeakPick / SearchWalk タスク定義(環境設定・報酬・観測・リセット)
scripts/mjlab_cli.py                 タスクを登録してから mjlab の train / play / export を呼ぶ
scripts/                             検証・評価・動画作成スクリプト
onnx/                                学習済みポリシー
run.sh                               上流 venv の python で実行するラッパー
```

## クレジット

- ロボットモデル・学習環境・学習済みポリシー: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)(Apache-2.0、3D モデルは CC BY-NC-SA)
- サーボ特性モデル: [Rhoban/bam](https://github.com/Rhoban/bam)
- 学習基盤: [mujocolab/mjlab](https://github.com/mujocolab/mjlab)
