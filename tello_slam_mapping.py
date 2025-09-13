#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tello_slam_mapping.py
- Telloの映像から MASt3R-SLAM を「必ず開始」&「必ず地図が生える」ようチューニングした実機向け完成版
- 重要ポイント:
  * 強制KF (--force_kf_every) で特徴が薄くても前進
  * KF追加前に必ず mast3r_inference_mono で幾何(点群/pointmap)を埋めてから共有
  * retrieval/matching 閾値を起動時に緩めて因子を入れやすく
  * キャリブ未指定でも暫定Kを自動設定（後で実測に差し替えてください）
  * 低遅延: PyAVで最新1枚のみ使用（古いフレームは即破棄）
"""

import os, sys, time, threading, socket, argparse, yaml
import numpy as np
import cv2
import torch, lietorch
import torch.multiprocessing as mp
from contextlib import nullcontext

# ===== MASt3R-SLAM =====
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization

TELLO_ADDR = ("192.168.10.1", 8889)
TELLO_URL  = "udp://0.0.0.0:11111"

# -----------------------------
# 受信：PyAVで最新フレームのみ
# -----------------------------
class PyAVLatest:
    def __init__(self, url=TELLO_URL, reset_stream=False):
        import av
        self.av = av
        if reset_stream:
            self._reset_stream()
        self._open(url)
        self._lock = threading.Lock()
        self._latest = None  # (ts, frame_bgr)
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _reset_stream(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1.0)
            for cmd in (b"command", b"streamoff", b"streamon"):
                s.sendto(cmd, TELLO_ADDR); time.sleep(0.25)
            s.close(); time.sleep(0.4)
        except Exception:
            pass

    def _open(self, url):
        # 低遅延オプション
        self.container = self.av.open(
            url, format="h264", mode="r",
            options={
                "fflags": "nobuffer",
                "flags": "low_delay",
                "probesize": "32",
                "analyzeduration": "0",
                "max_delay": "0",
                "reorder_queue_size": "0",
                "avioflags": "direct",
                "use_wallclock_as_timestamps": "1",
                "threads": "1",
                "refcounted_frames": "1",
            }
        )
        self.stream = next(s for s in self.container.streams if s.type == "video")
        self.stream.thread_type = "AUTO"

    def _loop(self):
        last_frame = None
        while self._run:
            try:
                for pkt in self.container.demux(self.stream):
                    if not self._run: break
                    if pkt.dts is None:
                        continue
                    got = False
                    for frm in pkt.decode():
                        img = frm.to_ndarray(format="bgr24")
                        last_frame = img
                        got = True
                    if got and last_frame is not None:
                        with self._lock:
                            self._latest = (time.time(), last_frame)
                time.sleep(0.001)
            except self.av.AVError:
                # 再オープン（Iフレーム待ちなど）
                try:
                    self.container.close()
                except Exception:
                    pass
                time.sleep(0.2)
                self._open(TELLO_URL)

    def get(self):
        with self._lock:
            return self._latest

    def close(self):
        self._run = False
        try: self._t.join(timeout=0.5)
        except: pass
        try: self.container.close()
        except: pass


# -----------------------------
# Backend（標準）
# -----------------------------
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


# -----------------------------
# メイン
# -----------------------------
def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--proc_size", type=int, default=224, choices=[224, 512])
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--reset_stream", type=int, default=1)
    ap.add_argument("--calib", default="", help="Tello用キャリブYAML")
    # ここから “確実に地図を生やす” オプション
    ap.add_argument("--force_kf_every", type=int, default=12,
                    help="N>0: Nフレ毎に強制KF（特徴が薄くても前進）")
    ap.add_argument("--viz_interval", type=int, default=2,
                    help="可視化に流す間引き（1=毎フレーム）")
    ap.add_argument("--go_every_kf", type=int, default=2,
                    help="グローバル最適化をこの間隔で実行")
    ap.add_argument("--stale_thresh_ms", type=float, default=35.0,
                    help="古いフレーム破棄の閾値")
    ap.add_argument("--clahe", action="store_true",
                    help="VチャンネルCLAHEで特徴を少し増やす")
    ap.add_argument("--relax_matching", action="store_true",
                    help="retrieval/マッチ閾値を起動時に緩める（地図が生えない時にON推奨）")
    args = ap.parse_args()

    # 設定のロード
    load_config(args.config)

    # --- しきい値を緩める（必要なら）-----------------
    if args.relax_matching:
        # 既存値を壊さないよう .get で取り出し、無ければ追加
        config.setdefault("retrieval", {})
        config.setdefault("local_opt", {})
        config.setdefault("reloc", {})
        config["retrieval"]["k"] = max(3, int(config["retrieval"].get("k", 5)))
        config["retrieval"]["min_thresh"] = min(0.18, float(config["retrieval"].get("min_thresh", 0.3)))
        config["local_opt"]["min_match_frac"] = min(0.06, float(config["local_opt"].get("min_match_frac", 0.15)))
        config["reloc"]["strict"] = False
        print("[Config] relaxed thresholds:", {
            "retrieval.k": config["retrieval"]["k"],
            "retrieval.min_thresh": config["retrieval"]["min_thresh"],
            "local_opt.min_match_frac": config["local_opt"]["min_match_frac"],
            "reloc.strict": config["reloc"]["strict"],
        })

    # --- 受信を開始（PyAV 最新1枚のみ） ---------------
    rx = PyAVLatest(reset_stream=bool(args.reset_stream))
    print(f"[Live/Tello] receiver=pyav proc_size={args.proc_size}")

    # 最初の有効フレーム
    t0 = time.time()
    first = None
    while time.time() - t0 < 3.0:
        item = rx.get()
        if item is not None and item[1] is not None:
            first = item[1]
            break
        time.sleep(0.01)
    if first is None:
        rx.close()
        raise RuntimeError("初期フレームを取得できません。")

    # ----- 共有・初期フレーム作成 ----------------------
    def resize_bgr(fr, size):
        return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)

    img_bgr0 = resize_bgr(first, args.proc_size)
    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]
    print(f"[Init] Shared buffers: {H_real}x{W_real}")

    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause(); states.set_mode(Mode.INIT)

    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start(); print(f"[Viz] pid={viz.pid} started")

    # ----- モデル・キャリブ ----------------------------
    model = load_mast3r(device=device)
    model.share_memory()

    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intr = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        intrinsics = Intrinsics.from_calib((H_real, W_real), intr["width"], intr["height"], intr["calibration"])
        K = torch.from_numpy(intrinsics.K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)
        print("[Calib] loaded from", args.calib)
    else:
        # 暫定Tello内パラ（後で実測に差し替え推奨）
        width, height = 960, 720
        fx = fy = 920.0; cx, cy = width/2, height/2
        calib = {"width": width, "height": height, "calibration": {"fx": fx,"fy": fy,"cx": cx,"cy": cy,"distortion":[-0.10,0.05,0.0,0.0,0.0]}}
        intrinsics = Intrinsics.from_calib((H_real, W_real), width, height, calib["calibration"])
        K = torch.from_numpy(intrinsics.K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)
        config["use_calib"] = True
        print("[Calib] using provisional intrinsics (update later for best mapping)")

    tracker  = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # ----- 補助 ---------------------------------------
    amp = torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
    def make_uimg_rgb_from_bgr(img_bgr, h, w):
        u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if u.shape[0] != h or u.shape[1] != w:
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
        u = (u.astype(np.float32) / 255.0).copy()
        return torch.from_numpy(u)

    # ----- ループ -------------------------------------
    i = 0
    last_kf_i = -1
    cap_s = inf_s = loop_s = 0.0

    try:
        while True:
            t_loop = time.time()

            # viz メッセージ
            mode = states.get_mode()
            m = try_get_msg(viz2main)
            if m: last_msg = m
            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED); break
            if last_msg.is_paused and not last_msg.next:
                states.pause(); time.sleep(0.01); continue
            else:
                states.unpause()

            # 取り込み（最新のみ）
            t_cap = time.time()
            item = rx.get()
            if item is None:
                time.sleep(0.001); continue
            ts, img_bgr = item
            cap_s += (time.time() - t_cap)

            # ステイル破棄
            if (time.time() - ts) * 1000.0 > args.stale_thresh_ms:
                continue

            # 前処理
            img_bgr = cv2.resize(img_bgr, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            if args.clahe:
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                h_, s_, v_ = cv2.split(hsv)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                v_ = clahe.apply(v_)
                hsv = cv2.merge([h_, s_, v_])
                img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            prev_for_T = states.get_frame()
            T_WC = lietorch.Sim3.Identity(1, device=device) if (i == 0 or prev_for_T is None) else prev_for_T.T_WC
            frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
            frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

            # KF=0 なら INIT に戻す
            if len(keyframes) == 0 and mode != Mode.INIT:
                states.set_mode(Mode.INIT); mode = Mode.INIT

            # 推論/追跡
            t_inf = time.time()
            did_geom = False

            if mode == Mode.INIT:
                with amp:
                    X0, C0 = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X0, C0)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes)-1)
                states.set_mode(Mode.TRACKING)
                last_kf_i = i
                did_geom = True
                print(f"[KF] INIT appended (i={i})")

            elif mode == Mode.TRACKING:
                with amp:
                    add_kf, _, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)

                force_kf = (args.force_kf_every > 0) and (i - last_kf_i >= args.force_kf_every)
                if add_kf or force_kf:
                    # 幾何を確実に埋める
                    need_geom = any(getattr(frame, a, None) is None for a in ("pointmap","X_canon","C_canon"))
                    if need_geom:
                        with amp:
                            Xf, Cf = mast3r_inference_mono(model, frame)
                        frame.update_pointmap(Xf, Cf)
                        did_geom = True
                    keyframes.append(frame)
                    if (len(keyframes) % max(1, args.go_every_kf)) == 0:
                        states.queue_global_optimization(len(keyframes)-1)
                    last_kf_i = i
                    print(f"[KF] appended (i={i}, force={force_kf}, tracker_add={add_kf})")

            elif mode == Mode.RELOC:
                with amp:
                    Xr, Cr = mast3r_inference_mono(model, frame)
                frame.update_pointmap(Xr, Cr)
                states.queue_reloc()
                did_geom = True

            inf_s += (time.time() - t_inf)

            # 幾何継承（保険）
            if not did_geom:
                prev = states.get_frame()
                if prev is not None:
                    for a in ("X_canon","C_canon","feat","pos","pointmap"):
                        if getattr(prev, a, None) is not None and getattr(frame, a, None) is None:
                            setattr(frame, a, getattr(prev, a).clone())

            # 可視化に流す
            if args.viz_interval <= 1 or (i % args.viz_interval) == 0:
                states.set_frame(frame)

            # ログ
            loop_s += (time.time() - t_loop)
            if i > 0 and (i % 60 == 0):
                def pr(s): 
                    ms = 1000.0*(s/60); fps = 1000.0/ms if ms>0 else 0.0
                    return ms, fps
                cap_ms, cap_fps = pr(cap_s); inf_ms, inf_fps = pr(inf_s); loop_ms, loop_fps = pr(loop_s)
                print(f"[Last 60] Capture: {cap_ms:.2f} ms ({cap_fps:.1f} FPS) | "
                      f"Infer(avg): {inf_ms:.2f} ms ({inf_fps:.1f} FPS) | Loop: {loop_ms:.2f} ms ({loop_fps:.1f} FPS) | "
                      f"KFs={len(keyframes)}")
                cap_s = inf_s = loop_s = 0.0

            i += 1

    finally:
        try: states.set_mode(Mode.TERMINATED)
        except: pass
        try: rx.close()
        except: pass

if __name__ == "__main__":
    main()
