# microduck-tidy-up

小型二足ロボット Microduck が、頭のカメラで床の物を探し、くちばしで拾って片付けエリアへ運ぶ(MuJoCo シミュレーション)。
Physical AI 応用 1 講座の最終課題。

- デモ動画:(YouTube の URL)
- 結果:物 3 個 × 配置 3 通りの片付けで 9/9

強化学習(PPO)で 2 つの動作を学習した。

- **BeakPick**:しゃがむ → くちばしで挟む → 持ったまま立ち上がる
- **SearchWalk**:うつむいたまま歩く(上流の歩行に追加学習)

## セットアップ

Linux(WSL2 可)と [uv](https://docs.astral.sh/uv/) が必要。学習する場合のみ CUDA GPU も必要。

```bash
# 上流を同じ階層に置いて環境を作る
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl && git checkout 29e887e
UV_PROJECT_ENVIRONMENT=~/.venvs/microduck_rl uv sync
cd ..

# 公式の歩行・立ちポリシー(alpha_walking.onnx, alpha_stand.onnx)を policies/ に置く
# 入手先: Hugging Face の pollen-robotics/microduck-policies

# このリポジトリを同じ階層に置く
git clone <このリポジトリの URL> microduck-tidy-up
cd microduck-tidy-up
```

## 実行

```bash
# 片付けデモ(動画を出力)
bash run.sh python scripts/fetch_demo.py onnx/v2_1000.onnx out.mp4 1 3 180 narrow onnx/sw2_final.onnx

# 物ごとの拾う成功率
bash run.sh python scripts/pick_eval.py onnx/v2_1000.onnx 10 all -

# 学習(BeakPick / SearchWalk)
bash run.sh python scripts/mjlab_cli.py train Mjlab-BeakPick-Flat-MicroDuck --env.scene.num-envs 2048
```

## 構成

```
src/microduck_tidy/beak.py    くちばしの関節の追加
src/microduck_tidy/tasks/     BeakPick / SearchWalk の学習タスク
src/microduck_tidy/fetch.py   カメラでの物の検出と、探す〜片付けるの状態機械
src/microduck_tidy/runner.py  学習済みポリシーの実行
scripts/                      デモ・評価・学習の実行スクリプト
onnx/                         学習済みポリシー(デモでは v2_1000 と sw2_final を使用)
```

## クレジット

- ロボットモデル・学習環境: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)(Apache-2.0、3D モデルは CC BY-NC-SA)
- サーボ特性モデル: [Rhoban/bam](https://github.com/Rhoban/bam)
- 学習基盤: [mujocolab/mjlab](https://github.com/mujocolab/mjlab)
