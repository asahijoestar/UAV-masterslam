#!/usr/bin/env python3
# opencv_to_ram_ring.py
# TelloのUDPストリームをOpenCVで受け取り、RAMディスクに「最新を吐き続ける」軽量連番PNGとして保存。
# - 書きかけファイルは見せない（アトミック書き）
# - 解像度は最初のフレームから決めて固定（形状不一致による落ちを回避）
# - PNG圧縮は最軽量（IMWRITE_PNG_COMPRESSION=1）で書き込み負荷を最小化
# - 過去フレームは十分古いものだけ間引く（GUARD幅で安全側）

import os
import cv2
import time
import glob
import signal

# ====== 調整ポイント ======
SRC         = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=131072"  # ため込みすぎないFIFO
OUT         = "/mnt/ram/frames"      # RAMディスクの保存先
TARGET_FPS  = 10                     # 保存fps（高すぎるとSLAMが詰まる）
SCALE       = 0.5                    # 解像度倍率（0.5 推奨、0.75 でも可）
GUARD       = 1000                   # 現在フレーム番号 i からこれだけ古いものだけ削除
REOPEN_FAILS= 120                    # 読み失敗がこの回数続いたら再オープン
PREVIEW     = False                  # Trueで軽量プレビューを出す（必要時のみ）

# OpenCVのスレッド数（必要なら調整。1にすると安定することがある）
try:
    cv2.setNumThreads(0)
except Exception:
    pass

os.makedirs(OUT, exist_ok=True)

# 既存の .tmp を掃除
for f in glob.glob(os.path.join(OUT, "*.tmp")):
    try:
        os.remove(f)
    except Exception:
        pass

def open_cap():
    cap = cv2.VideoCapture(SRC, cv2.CAP_FFMPEG)  # FFmpegバックエンドで受信
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # ため込み最小化（効かない環境もあり）
    return cap

cap = open_cap()

# Ctrl+C で安全終了
_running = True
def _sigint(_sig, _frm):
    global _running
    _running = False
signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)

ts_path = os.path.join(OUT, "timestamps.txt")
ts = open(ts_path, "w", buffering=1)

# ====== Warmup：安定フレームが来るまで捨てる ======
t0 = time.time()
first = None
while True:
    ok, frame = cap.read()
    if ok and frame is not None:
        first = frame
        break
    if time.time() - t0 > 3.0:
        break

if first is None:
    print("[WARN] 初期フレームを取得できませんでしたが継続します。")
else:
    # ====== 固定解像度を決める（16の倍数に丸めるとNNが安定）======
    bh, bw = first.shape[:2]
    if SCALE <= 0 or SCALE > 2:
        SCALE = 1.0
    tgt_w = max(160, (int(bw * SCALE) // 16) * 16)
    tgt_h = max(120, (int(bh * SCALE) // 16) * 16)
    # 先頭フレームを基準サイズに合わせて先に吐いておく（任意）
    ok_enc, buf = cv2.imencode(".png", cv2.resize(first, (tgt_w, tgt_h)),
                               [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok_enc:
        tmp0 = os.path.join(OUT, "000000.png.tmp")
        with open(tmp0, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp0, os.path.join(OUT, "000000.png"))
        ts.write(f"000000 {time.time():.6f}\n")
        start_idx = 1
    else:
        start_idx = 0

# ====== ループ本体 ======
i = start_idx
t_last = 0.0
fail = 0
cleanup_every = 60   # 60フレームに1回だけ掃除して負荷を下げる
cleanup_tick = 0

try:
    while _running:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail += 1
            if fail >= REOPEN_FAILS:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = open_cap()
                fail = 0
            continue
        fail = 0

        now = time.time()
        # 保存fpsを上限に間引く
        if TARGET_FPS > 0 and (now - t_last) < (1.0 / TARGET_FPS):
            continue
        t_last = now

        # 必ず固定解像度へ（INTER_AREAは縮小に最適）
        if first is None:
            # 初回に基準が取れなかった場合はここで動的に決める
            bh, bw = frame.shape[:2]
            tgt_w = max(160, (int(bw * SCALE) // 16) * 16)
            tgt_h = max(120, (int(bh * SCALE) // 16) * 16)
            first = True  # 以後は固定
        frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

        # ---- アトミック書き（PNGをメモリエンコード→.tmp→rename）----
        final = os.path.join(OUT, f"{i:06d}.png")
        tmp   = final + ".tmp"
        ok_enc, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok_enc:
            continue
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, final)

        ts.write(f"{i:06d} {now:.6f}\n")
        i += 1

        # ---- 定期的に“十分古い”ファイルだけ掃除 ----
        cleanup_tick += 1
        if cleanup_tick >= cleanup_every and i > GUARD:
            cleanup_tick = 0
            cutoff = i - GUARD
            # 先頭側を軽くスキャン（重くしない）
            # ファイル数が多いときは head 的に200〜400件だけ見る
            for old in sorted(glob.glob(os.path.join(OUT, "*.png")))[:300]:
                base = os.path.basename(old).split('.')[0]
                try:
                    idx = int(base)
                except Exception:
                    continue
                if idx < cutoff:
                    try:
                        os.remove(old)
                    except Exception:
                        pass

        # 軽量プレビュー（必要なときだけ）
        if PREVIEW:
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    try:
        cap.release()
    except Exception:
        pass
    try:
        ts.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
