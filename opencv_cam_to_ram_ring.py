#!/usr/bin/env python3
# USBカメラ→/mnt/ram/frames に連番PNGを吐き続ける（固定解像度・低圧縮・アトミック書き）
import os, cv2, time, glob, signal

CAM_INDEX      = 0                 # /dev/video0
FORCE_SIZE     = (640, 480)        # 最速狙いの基準サイズ（W,H）※必要なら (512,384),(480,360) 等
TARGET_FPS     = 18                # 保存上限fps（負荷に応じて 12〜20で調整）
OUT            = "/mnt/ram/frames"
GUARD          = 1200              # これより古い連番だけ間引き
PREVIEW        = False

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.tmp")): 
    try: os.remove(f)
    except: pass

def open_cap():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    # 希望値（効かない場合もある）
    w,h = FORCE_SIZE
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # ため込み抑制（効かない環境あり）
    return cap

cap = open_cap()

_running=True
signal.signal(signal.SIGINT,  lambda *a: globals().__setitem__('_running', False))
signal.signal(signal.SIGTERM, lambda *a: globals().__setitem__('_running', False))

ts_path = os.path.join(OUT, "timestamps.txt")
ts = open(ts_path, "w", buffering=1)

# 先頭フレーム確保（ウォームアップ）
t0=time.time(); first=None
while time.time()-t0 < 1.5:
    ok,frame = cap.read()
    if ok and frame is not None:
        first = frame
        break
if first is None:
    print("[WARN] 初期フレームが得られませんでした。続行します。")

w,h = FORCE_SIZE
i = 0
t_last = 0.0
cleanup_tick=0

try:
    while _running:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        now = time.time()
        if TARGET_FPS>0 and (now - t_last) < (1.0/TARGET_FPS):
            continue
        t_last = now

        # 必ず固定解像度に
        frame = cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)

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

        # 古いフレームを安全に間引く（重くならない範囲で）
        cleanup_tick += 1
        if cleanup_tick >= 60 and i > GUARD:
            cleanup_tick = 0
            cutoff = i - GUARD
            for old in sorted(glob.glob(os.path.join(OUT, "*.png")))[:300]:
                try:
                    idx = int(os.path.basename(old).split('.')[0])
                    if idx < cutoff:
                        os.remove(old)
                except: pass

        if PREVIEW:
            cv2.imshow("cam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    pass
finally:
    try: cap.release()
    except: pass
    try: ts.close()
    except: pass
    cv2.destroyAllWindows()
