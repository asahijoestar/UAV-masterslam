#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, cv2, torch, yaml, lietorch
import numpy as np
import torch.multiprocessing as mp

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization


# -----------------------------
# Live camera (BGR)
# -----------------------------
class LiveRGBDataset:
    """Webカメラ/動画/UDP→BGR(OpenCV)に正規化。常に proc_size×proc_size を返す。"""
    def __init__(self, src, proc_size=512, fps=0.0, yuv_mode="auto"):
        self.proc_size = int(proc_size)
        self.yuv_mode = yuv_mode.upper() if isinstance(yuv_mode, str) else "AUTO"

        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # 実質BGRで取得
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self.proc_size))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.proc_size))
            if fps and fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        ok, fr = self.cap.read()
        if not ok or fr is None:
            raise RuntimeError("No frames from camera")

        fr = self._to_bgr(fr)
        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        self._shape = ((self.proc_size, self.proc_size, 3),)
        self.img_size = (self.proc_size, self.proc_size)

    def _to_bgr(self, fr):
        if fr.ndim == 2:
            return cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        if fr.ndim == 3 and fr.shape[2] == 2:
            if self.yuv_mode == "UYVY":
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_UYVY)
            elif self.yuv_mode in ("YUY2", "YUYV"):
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_YUY2)
            elif self.yuv_mode == "NV12":
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_NV12)
            else:
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_YUY2)
        return fr

    def __getitem__(self, i):
        ok, fr = self.cap.read()
        if not ok or fr is None:
            return time.time(), None
        fr = self._to_bgr(fr)
        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        return time.time(), fr

    def __len__(self): return 1 << 60
    def get_img_shape(self): return self._shape
    def subsample(self, n): pass
    def has_calib(self): return False


# -----------------------------
# Backend（既存のまま）
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
# main
# -----------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config",  default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz",  action="store_true")
    parser.add_argument("--calib",   default="")
    parser.add_argument("--camera",   type=str,  default=None,
                        help="Live source. '0' for /dev/video0, 'udp://0.0.0.0:11111?...', or video path")
    parser.add_argument("--proc_size", type=int, default=512, choices=[224, 512],
                        help="Model input size (square). 224 or 512")
    parser.add_argument("--cam_fps",  type=float, default=0.0,
                        help="Request camera FPS (0 = auto)")
    parser.add_argument("--yuv",      type=str,  default="auto",
                        choices=["auto", "YUY2", "YUYV", "UYVY", "NV12"],
                        help="If input is 2ch YUV, select conversion (default:auto=YUY2).")
    args = parser.parse_args()

    load_config(args.config)

    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # 1) データソース
    if args.camera is not None:
        dataset = LiveRGBDataset(src=args.camera, proc_size=args.proc_size, fps=args.cam_fps, yuv_mode=args.yuv)
        print(f"[Live] source={args.camera} size={args.proc_size} fps={args.cam_fps} yuv={args.yuv}")
    else:
        dataset = load_dataset(args.dataset)
        dataset.subsample(config["dataset"]["subsample"])

    # 2) 最初のフレームで実ネットサイズを取得（例: 384x512）
    ts0, img_bgr0 = dataset[0]
    if img_bgr0 is None:
        raise RuntimeError("Failed to grab the first frame.")

    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]  # CHW末尾

    # 3) 共有バッファ（H_real, W_real）で確保
    print(f"[Init] Shared buffers: {H_real}x{W_real}")
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause()

    # 4) 可視化
    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start()
        print(f"[Viz] pid={viz.pid} started")

    # 5) モデル
    model = load_mast3r(device=device)
    model.share_memory()

    # 6) キャリブ（ある場合）
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intr = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        if hasattr(dataset, "use_calibration"): dataset.use_calibration = True
        if hasattr(dataset, "camera_intrinsics"):
            dataset.camera_intrinsics = Intrinsics.from_calib(
                (H_real, W_real), intr["width"], intr["height"], intr["calibration"]
            )
            K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)
            keyframes.set_intrinsics(K)

    tracker  = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # 7) ヘルパ：uimg を (H_real, W_real) に整形（RGB/HWC/float32[0..1]）
    def make_uimg_rgb_from_bgr(img_bgr, h, w):
        u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HxW x3 (uint8)
        if u.shape[0] != h or u.shape[1] != w:
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
        u = (u.astype(np.float32) / 255.0).copy()     # 書込可
        return torch.from_numpy(u)                    # CPU, HWC, float32

    # 8) メインループ（60フレーム平均の時間＆FPSを出力）
    i = 0
    t0 = time.time()
    cap_time_acc = 0.0
    inf_time_acc = 0.0
    tot_time_acc = 0.0

    while True:
        loop_start = time.time()

        # 状態・メッセージ
        mode = states.get_mode()
        msg  = try_get_msg(viz2main)
        last_msg = msg if msg else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED); break
        if last_msg.is_paused and not last_msg.next:
            states.pause(); time.sleep(0.01); continue
        else:
            states.unpause()

        # 取り込み
        cap_start = time.time()
        ts, img_bgr = dataset[i]
        if img_bgr is None:
            time.sleep(0.002); continue
        cap_time_acc += (time.time() - cap_start)

        # Frame 準備
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        prev_for_T = states.get_frame()
        T_WC = lietorch.Sim3.Identity(1, device=device) if (i == 0 or prev_for_T is None) else prev_for_T.T_WC
        frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
        frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

        # 推論／追跡
        inf_start = time.time()
        did_geometry_update = False

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes)-1)
            states.set_mode(Mode.TRACKING)
            did_geometry_update = True

        elif mode == Mode.TRACKING:
            add_new_kf, _, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            if add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes)-1)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.queue_reloc()
            did_geometry_update = True

        inf_time_acc += (time.time() - inf_start)

        # 安全ガード：幾何が None のままなら前フレームから埋める
        if not did_geometry_update:
            prev = states.get_frame()
            if prev is not None:
                if getattr(prev, "X_canon", None) is not None and getattr(frame, "X_canon", None) is None:
                    frame.X_canon = prev.X_canon.clone()
                if getattr(prev, "C_canon", None) is not None and getattr(frame, "C_canon", None) is None:
                    frame.C_canon = prev.C_canon.clone()
                if getattr(prev, "feat", None) is not None and getattr(frame, "feat", None) is None:
                    frame.feat = prev.feat.clone()
                if getattr(prev, "pos", None) is not None and getattr(frame, "pos", None) is None:
                    frame.pos = prev.pos.clone()
                if getattr(prev, "pointmap", None) is not None and getattr(frame, "pointmap", None) is None:
                    frame.pointmap = prev.pointmap.clone()

        # 共有状態へ
        states.set_frame(frame)

        # 60フレームごとに平均時間＆FPSを表示
        tot_time_acc += (time.time() - loop_start)
        if i > 0 and (i % 60 == 0):
            avg_cap_ms = 1000.0 * (cap_time_acc / 60)
            avg_inf_ms = 1000.0 * (inf_time_acc / 60)
            avg_tot_ms = 1000.0 * (tot_time_acc / 60)
            cap_fps = 1000.0 / avg_cap_ms if avg_cap_ms > 0 else float('inf')
            inf_fps = 1000.0 / avg_inf_ms if avg_inf_ms > 0 else float('inf')
            tot_fps = 1000.0 / avg_tot_ms if avg_tot_ms > 0 else float('inf')
            print(f"[Last 60 frames] "
                  f"Capture: {avg_cap_ms:.2f} ms ({cap_fps:.2f} FPS) | "
                  f"Inference: {avg_inf_ms:.2f} ms ({inf_fps:.2f} FPS) | "
                  f"Total loop: {avg_tot_ms:.2f} ms ({tot_fps:.2f} FPS)")
            cap_time_acc = inf_time_acc = tot_time_acc = 0.0

        i += 1

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
