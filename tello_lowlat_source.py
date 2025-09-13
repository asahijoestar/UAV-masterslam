#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tello_lowlat.py
- TelloのH.264ストリームを超低遅延で受信し、常に「最新フレームだけ」をMASt3R-SLAMに流す完成版
- 特徴:
  * FFmpeg低遅延オプション & 最新1枚バッファ(AsyncCapture) → 「届くまで」を最小化
  * ステイル(古い)フレーム即廃棄 / フレームスキップ / 可視化＆GO間引き → 体感遅延を最小化
  * FP16(autocast)で推論を高速化
  * 起動時 streamoff→on (任意) でIフレームを促進
  * 端末到着→可視化投入までのE2E遅延をログ出力
"""

import os, time, socket, threading, argparse, yaml
import numpy as np
import cv2
import torch, lietorch
import torch.multiprocessing as mp
from contextlib import nullcontext

# ========== MASt3R-SLAM imports ==========
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization

# ---------- 低遅延：FFmpeg設定 ----------
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "protocol_whitelist;file,udp,rtp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
    "|max_delay;0"
    "|min_delay;1"
    "|use_wallclock_as_timestamps;1"
    "|avioflags;direct"
    "|flush_packets;1"
    "|probesize;32"
    "|analyzeduration;0"
)

TELLO_URL  = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000&reuse=1"
TELLO_ADDR = ("192.168.10.1", 8889)


# ---------- 低遅延受信（最新1枚だけ保持） ----------
class _AsyncCaptureLatest:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError("Tello UDPストリームを開けません（streamon後か、ポート競合確認）")

        self._lock = threading.Lock()
        self._latest = None  # (ts, frame_bgr)
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            ok, fr = self.cap.read()
            if not ok or fr is None or fr.size == 0:
                time.sleep(0.001)
                continue
            ts = time.time()
            with self._lock:
                self._latest = (ts, fr)

    def read_latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        try: self._t.join(timeout=0.5)
        except: pass
        try: self.cap.release()
        except: pass


def reset_stream():
    """Iフレームを早く引き出すために streamoff→on"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.sendto(b"command", TELLO_ADDR);    time.sleep(0.2)
        s.sendto(b"streamoff", TELLO_ADDR);  time.sleep(0.4)
        s.sendto(b"streamon", TELLO_ADDR);   time.sleep(0.5)
        s.close()
    except Exception:
        pass


# --- 置き換え: TelloLowLatencySource.__init__ ---
class TelloLowLatencySource:
    def __init__(self, url=TELLO_URL, warmup_sec=2.0, do_reset=False):
        url_candidates = [
            url,  # 例: "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000&reuse=1"
            "udp://@0.0.0.0:11111",                    # FFmpegの別表記
            "udp://0.0.0.0:11111?reuse=1",             # シンプル
            # ローカルIPを明示したい環境向け（例: 192.168.10.2 は自分のIPに置換）
            # "udp://0.0.0.0:11111?localaddr=192.168.10.2&reuse=1",
        ]
        attempts = 0
        last_err = None
        if do_reset:
            reset_stream()

        while attempts < 3:
            for u in url_candidates:
                try:
                    self.async_cap = _AsyncCaptureLatest(u)
                except Exception as e:
                    last_err = e
                    continue

                # ウォームアップ: 有効フレームを待つ
                t0 = time.time()
                ok = False
                while time.time() - t0 < max(2.0, warmup_sec):
                    if self.async_cap.read_latest() is not None:
                        ok = True; break
                    time.sleep(0.02)
                if ok:
                    return

                # ダメならクローズして次の候補へ
                try: self.async_cap.stop()
                except: pass

            # 1ラウンド失敗 → stream をリセットして再試行
            attempts += 1
            reset_stream()
            time.sleep(0.5)

        raise RuntimeError(f"Telloから有効なフレームが届きません（PPS/Iフレーム未着 or 受信側バインド問題）。last_err={last_err}")



# ---------- Backend ----------
def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = retrieval_database.update(
            frame, add_after_query=False,
            k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
        )
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            frame_idx = [n_kf - 1] * len(kf_idx)
            if factor_graph.add_factors(
                frame_idx, list(kf_idx),
                config["reloc"]["min_match_frac"], is_reloc=config["reloc"]["strict"]
            ):
                retrieval_database.update(
                    frame, add_after_query=True,
                    k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
                )
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[list(kf_idx)[0]].clone()
                if config["use_calib"]: factor_graph.solve_GN_calib()
                else:                   factor_graph.solve_GN_rays()
                return True
            else:
                keyframes.pop_last()
        return False


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    while states.get_mode() is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01); continue

        if mode == Mode.RELOC:
            frame = states.get_frame()
            if relocalization(frame, keyframes, factor_graph, retrieval_database):
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue

        idx = -1
        with states.lock:
            if states.global_optimizer_tasks:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01); continue

        frame = keyframes[idx]
        kf_idx = retrieval_database.update(
            frame, add_after_query=True,
            k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
        )
        if kf_idx:
            factor_graph.add_factors(
                list(kf_idx), [idx]*len(kf_idx), config["local_opt"]["min_match_frac"]
            )

        if config["use_calib"]: factor_graph.solve_GN_calib()
        else:                   factor_graph.solve_GN_rays()

        with states.lock:
            if states.global_optimizer_tasks:
                states.global_optimizer_tasks.pop(0)


# ---------- メイン ----------
def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config/base.yaml")
    ap.add_argument("--calib",   default="")
    ap.add_argument("--proc_size", type=int, default=224, choices=[224, 512],
                    help="モデル入力の正方形解像度（224推奨=軽い / 512=高精度）")
    ap.add_argument("--no-viz",  action="store_true")
    ap.add_argument("--reset_stream", type=int, default=0,
                    help="1で開始時に streamoff→on を送ってIフレームを促す")
    # 遅延最適化の追加スイッチ
    ap.add_argument("--frame_skip", type=int, default=0, help="N>0で N+1枚中1枚だけ処理（例:1→半分）")
    ap.add_argument("--viz_interval", type=int, default=2, help="可視化に流すフレーム間引き（1=毎フレーム）")
    ap.add_argument("--go_every_kf", type=int, default=2, help="グローバル最適化は M 枚に1回だけ走らせる")
    ap.add_argument("--stale_thresh_ms", type=float, default=45.0,
                    help="この閾値より古いフレームは捨てる（端末到着→受信時刻の差）")
    args = ap.parse_args()

    load_config(args.config)

    # 1) 低遅延受信
    src = TelloLowLatencySource(do_reset=bool(args.reset_stream))
    print(f"[Live/Tello] url={TELLO_URL} size={args.proc_size}")

    # 最初の有効フレームで入力形状を確定
    t0 = time.time()
    first = None
    while time.time() - t0 < 2.0:
        ts0, fr0 = src.get()
        if fr0 is not None:
            first = fr0
            break
        time.sleep(0.01)
    if first is None:
        raise RuntimeError("初期フレームを取得できませんでした。")

    def resize_bgr(fr, size):
        return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)

    img_bgr0 = resize_bgr(first, args.proc_size)
    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]
    print(f"[Init] Shared buffers: {H_real}x{W_real}")

    # 2) 共有
    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause()
    states.set_mode(Mode.INIT)

    # 3) 可視化
    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start()
        print(f"[Viz] pid={viz.pid} started")

    # 4) モデル
    model = load_mast3r(device=device)
    model.share_memory()

    # 5) キャリブ（任意）
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intr = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        intrinsics = Intrinsics.from_calib((H_real, W_real), intr["width"], intr["height"], intr["calibration"])
        K = torch.from_numpy(intrinsics.K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)

    tracker  = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # FP16コンテキスト
    amp = torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

    def make_uimg_rgb_from_bgr(img_bgr, h, w):
        u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if u.shape[0] != h or u.shape[1] != w:
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
        u = (u.astype(np.float32) / 255.0).copy()
        return torch.from_numpy(u)

    # 6) ループ
    i = 0
    cap_time_acc = inf_time_acc = tot_time_acc = 0.0
    avg_inf_ms_last = 0.0
    go_mod = max(1, args.go_every_kf)

    try:
        while True:
            loop_start = time.time()

            # 可視化メッセージ
            mode = states.get_mode()
            msg  = try_get_msg(viz2main)
            if msg: last_msg = msg
            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED); break
            if last_msg.is_paused and not last_msg.next:
                states.pause(); time.sleep(0.01); continue
            else:
                states.unpause()

            # 取り込み（最新のみ）
            cap_start = time.time()
            ts, img_bgr = src.get()
            if img_bgr is None:
                time.sleep(0.001); continue
            cap_time_acc += (time.time() - cap_start)

            # 古いフレームは捨てる（ステイル対策）
            stale_ms = (time.time() - ts) * 1000.0
            if stale_ms > args.stale_thresh_ms:
                # できるだけ新しいものに置き換える（最大5回）
                retried = 0
                while retried < 5:
                    ts2, fr2 = src.get()
                    if fr2 is not None and (time.time() - ts2) * 1000.0 <= args.stale_thresh_ms:
                        ts, img_bgr = ts2, fr2
                        stale_ms = (time.time() - ts) * 1000.0
                        break
                    retried += 1
                    time.sleep(0.001)
                else:
                    # 今回は処理せず次へ
                    continue

            # フレームスキップ
            if args.frame_skip > 0 and (i % (args.frame_skip + 1)) != 0:
                i += 1
                continue

            # 前処理
            img_bgr = cv2.resize(img_bgr, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            prev_for_T = states.get_frame()
            T_WC = lietorch.Sim3.Identity(1, device=device) if (i == 0 or prev_for_T is None) else prev_for_T.T_WC
            frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
            frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

            # INIT復帰ガード
            if len(keyframes) == 0 and mode != Mode.INIT:
                states.set_mode(Mode.INIT)
                mode = Mode.INIT

            # 推論／追跡（FP16）
            inf_start = time.time()
            did_geometry_update = False

            if mode == Mode.INIT:
                with amp:
                    X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                if (len(keyframes) % go_mod) == 0:
                    states.queue_global_optimization(len(keyframes)-1)
                states.set_mode(Mode.TRACKING)
                did_geometry_update = True

            elif mode == Mode.TRACKING:
                with amp:
                    add_new_kf, _, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                if add_new_kf:
                    keyframes.append(frame)
                    if (len(keyframes) % go_mod) == 0:
                        states.queue_global_optimization(len(keyframes)-1)

            elif mode == Mode.RELOC:
                with amp:
                    X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.queue_reloc()
                did_geometry_update = True

            inf_ms = (time.time() - inf_start) * 1000.0
            inf_time_acc += (time.time() - inf_start)
            avg_inf_ms_last = inf_ms

            # セーフガード（幾何未更新なら前フレームから継承）
            if not did_geometry_update:
                prev = states.get_frame()
                if prev is not None:
                    for attr in ("X_canon", "C_canon", "feat", "pos", "pointmap"):
                        if getattr(prev, attr, None) is not None and getattr(frame, attr, None) is None:
                            setattr(frame, attr, getattr(prev, attr).clone())

            # 可視化投入（間引き可能）
            if args.viz_interval <= 1 or (i % args.viz_interval) == 0:
                states.set_frame(frame)

            # 統計
            tot_time_acc += (time.time() - loop_start)
            if i > 0 and (i % 60 == 0):
                def avg_ms_fps(total_s, n=60):
                    avg_ms = 1000.0 * (total_s / n)
                    fps = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
                    return avg_ms, fps
                cap_ms, cap_fps = avg_ms_fps(cap_time_acc)
                inf_ms_avg, inf_fps = avg_ms_fps(inf_time_acc)
                tot_ms, tot_fps = avg_ms_fps(tot_time_acc)
                e2e_ms = (time.time() - ts) * 1000.0
                print(f"[Last 60] Capture: {cap_ms:.2f} ms ({cap_fps:.1f} FPS) | "
                      f"Infer(avg): {inf_ms_avg:.2f} ms ({inf_fps:.1f} FPS) | "
                      f"Loop: {tot_ms:.2f} ms ({tot_fps:.1f} FPS) | "
                      f"[Latency] recv->set_frame(last): {e2e_ms:.1f} ms | stale_in: {stale_ms:.1f} ms | infer(last): {avg_inf_ms_last:.1f} ms")
                cap_time_acc = inf_time_acc = tot_time_acc = 0.0

            i += 1

    finally:
        # 終了処理
        try:
            states.set_mode(Mode.TERMINATED)
        except Exception:
            pass
        if not args.no_viz:
            try: viz.join(timeout=1.0)
            except: pass
        try: src.close()
        except: pass


if __name__ == "__main__":
    main()
