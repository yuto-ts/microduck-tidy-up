"""Assemble the 1-3 minute submission video from the rendered clips.

usage: python scripts/make_submission_video.py <montage.mp4> <fetch_demo.mp4> <out.mp4> [beak.png]

Sections: title card -> what was built -> 14 objects picked once each (0.5x) ->
search, pick and tidy up 3 objects (1x) -> results. Every section states its playback
speed on screen (the assignment requires a sped-up/slowed part to be labelled).
No personal information is written anywhere in the video.
"""
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MONTAGE, DEMO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BEAK = sys.argv[4] if len(sys.argv) > 4 else None
W, H, FPS = 1280, 720, 25
FONT = os.environ.get("JP_FONT", "/mnt/c/Windows/Fonts/meiryo.ttc")
F = {s: ImageFont.truetype(FONT, s) for s in (22, 26, 30, 36, 48)}
FB = {s: ImageFont.truetype(FONT.replace("meiryo.ttc", "meiryob.ttc"), s)
      if os.path.exists(FONT.replace("meiryo.ttc", "meiryob.ttc")) else F[s] for s in (36, 48)}
BG, INK, QUIET, ACCENT = (24, 28, 36), (245, 245, 245), (175, 180, 190), (255, 170, 60)

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
n_frames = 0


def emit(img, times=1):
    global n_frames
    arr = np.asarray(img.convert("RGB"))
    for _ in range(times):
        writer.append_data(arr)
        n_frames += 1


def card(lines, secs, image=None):
    """lines: list of (text, size, colour, bold). Centred vertically, left-aligned at x=120."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    top = 120
    if image is not None:
        pic = Image.open(image).convert("RGB")
        pic.thumbnail((520, 520))
        im.paste(pic, (W - pic.width - 80, (H - pic.height) // 2))
    y = top
    for text, size, colour, bold in lines:
        font = FB[size] if bold and size in FB else F[size]
        d.text((120, y), text, font=font, fill=colour)
        y += int(size * 1.7)
    emit(im, int(secs * FPS))


def tag(img, text):
    d = ImageDraw.Draw(img, "RGBA")
    w = d.textlength(text, font=F[22]) + 28
    d.rounded_rectangle([W - w - 16, H - 52, W - 16, H - 14], 8, fill=(0, 0, 0, 170))
    d.text((W - w - 2, H - 47), text, font=F[22], fill=ACCENT)
    return img


def fit(frame):
    """Scale a frame into the 1280x720 canvas, letterboxed on the dark background."""
    pic = Image.fromarray(frame)
    pic.thumbnail((W, H))
    im = Image.new("RGB", (W, H), BG)
    im.paste(pic, ((W - pic.width) // 2, (H - pic.height) // 2))
    return im


# 1. title
card([("Microduck が床の物を探して拾い、片付ける", 48, INK, True),
      ("小型二足ロボットに、強化学習で「くちばしで拾う」「うつむいて歩く」を学習させた", 26, QUIET, False),
      ("", 26, QUIET, False),
      ("MuJoCo シミュレーション(実機のサーボ特性モデル BAM 付き)", 26, QUIET, False),
      ("Physical AI 応用 1 講座 最終課題", 26, QUIET, False)], 5)

# 2. what was built
card([("作ったもの", 36, ACCENT, True),
      ("1. くちばしの関節を後付けし、ポリシーの外から開閉する", 26, INK, False),
      ("2. BeakPick:しゃがむ・挟む・持ち上げるを強化学習(PPO)", 26, INK, False),
      ("3. SearchWalk:うつむいたまま歩く歩行を追加学習", 26, INK, False),
      ("4. 前向きカメラで物の位置を推定(誤差 9 mm 以内)", 26, INK, False),
      ("5. 状態機械が 探す → 近づく → 拾う → 運ぶ → 置く を繰り返す", 26, INK, False),
      ("", 22, QUIET, False),
      ("上流 pollen-robotics/microduck_rl のモデルと学習基盤を利用", 22, QUIET, False)], 9, BEAK)

# 3. montage of 14 objects, 0.5x, header cropped (it says 等速 for the native speed)
card([("拾う動作:候補 14 種の物を 1 回ずつ", 36, ACCENT, True),
      ("緑 = 成功、赤 = 失敗。下段の [限界例] は大きすぎる・低すぎる物", 26, INK, False),
      ("0.5 倍速で再生", 26, QUIET, False)], 3)
reader = imageio.get_reader(MONTAGE)
frames = [f for f in reader]
for f in frames:
    img = tag(fit(f[44:]), "0.5 倍速")
    emit(img, 2)
emit(img, FPS)

# 4. full fetch demo at 1x
card([("探して拾って片付ける:床の物 3 個", 36, ACCENT, True),
      ("右下は頭のカメラの映像、赤枠は検出した物", 26, INK, False),
      ("等速で再生(約 1 分 45 秒)", 26, QUIET, False)], 3)
reader = imageio.get_reader(DEMO)
for f in reader:
    img = tag(fit(f), "等速")
    emit(img)

# 5. results
card([("結果", 36, ACCENT, True),
      ("片付け:物 3 個 × 配置 3 通りで 9/9 個", 30, INK, False),
      ("拾う動作:小物 6 種(立方体・球・寝かせた直方体)で 59/60 回", 30, INK, False),
      ("課題:その場旋回は学習できず、立てた円柱は拾えない", 26, QUIET, False),
      ("", 22, QUIET, False),
      ("コードと説明資料は提出物を参照", 22, QUIET, False)], 6)
writer.close()
print(f"saved {OUT}: {n_frames / FPS:.1f} s")
