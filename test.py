import os, time, argparse, subprocess, threading, re
import cv2
import numpy as np
from ultralytics import YOLO


def build_ffmpeg_cmd(src, out_w, out_h, out_fps, ffmpeg_path):
    vf_parts = []
    if out_w > 0 and out_h > 0: vf_parts.append(f"scale={out_w}:{out_h}")
    if out_fps > 0:              vf_parts.append(f"fps={out_fps}")
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "warning"]
    if src.lower().startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp", "-fflags", "+nobuffer+discardcorrupt"]
    cmd += ["-i", src, "-an", "-sn", "-dn"]
    if vf_parts: cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd


_UNREC_RE = re.compile(r"Unrecognized option '([^']+)'\.", re.IGNORECASE)


class FFmpegLatestFrameReader:
    def __init__(self, src, width, height, out_fps, ffmpeg_path="ffmpeg"):
        self.src = src; self.w = int(width); self.h = int(height)
        self.out_fps = float(out_fps); self.ffmpeg_path = ffmpeg_path
        self.frame_size = self.w * self.h * 3
        self.proc = None; self.running = False; self.thread = None
        self.lock = threading.Lock(); self.latest = None
        self.latest_ts = 0.0; self.frame_id = 0
        self.last_err = ""

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self): self.running = False; self._kill_proc()

    def get_latest(self):
        with self.lock:
            if self.latest is None: return None, 0.0, 0
            return self.latest.copy(), self.latest_ts, self.frame_id

    def _kill_proc(self):
        p, self.proc = self.proc, None
        if p is None: return
        try: p.kill()
        except: pass

    def _stderr_loop(self, proc):
        try:
            while self.running and proc and proc.stderr:
                line = proc.stderr.readline()
                if not line: break
                s = line.decode(errors="ignore").strip()
                if s: self.last_err = s; print("[FFMPEG]", s)
        except: pass

    def _start_proc(self):
        cmd = build_ffmpeg_cmd(self.src, self.w, self.h, self.out_fps, self.ffmpeg_path)
        cf = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        print("\n[FFMPEG CMD]\n" + " ".join(cmd) + "\n")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=10**8, creationflags=cf)
        threading.Thread(target=self._stderr_loop, args=(self.proc,), daemon=True).start()

    def _restart_proc(self):
        self._kill_proc(); time.sleep(0.4); self._start_proc()

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n and self.running and self.proc and self.proc.stdout:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk: return None
            buf += chunk
        return buf if len(buf) == n else None

    def _loop(self):
        self._start_proc()
        while self.running:
            if self.proc is None or self.proc.poll() is not None:
                self._restart_proc(); continue
            raw = self._read_exact(self.frame_size)
            if raw is None: self._restart_proc(); continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
            with self.lock:
                self.latest = frame; self.latest_ts = time.time(); self.frame_id += 1


def main():
    ap = argparse.ArgumentParser("Det Model Tester")
    ap.add_argument("--rtsp",  default="rtsp://admin:KOEYHZ@192.168.2.72:554/cam/realmonitor?channel=1&subtype=1")
    ap.add_argument("--video", default="")
    ap.add_argument("--det_model",  required=True)
    ap.add_argument("--imgsz",     type=int,   default=640)
    ap.add_argument("--conf",      type=float, default=0.4)
    ap.add_argument("--det_fps",   type=float, default=6.0)
    ap.add_argument("--ffmpeg_fps",type=float, default=12.0)
    ap.add_argument("--width",     type=int,   default=640)
    ap.add_argument("--height",    type=int,   default=360)
    ap.add_argument("--ffmpeg",    default="ffmpeg")
    ap.add_argument("--track",     action="store_true", help="Dùng tracking thay vì predict")
    args = ap.parse_args()

    src = args.rtsp if args.rtsp else args.video
    if not src: raise RuntimeError("Provide --rtsp or --video")

    model = YOLO(args.det_model)

    reader = FFmpegLatestFrameReader(src, args.width, args.height, args.ffmpeg_fps, args.ffmpeg)
    reader.start()

    win = "Det Model Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    next_det_ts = 0.0
    last_items  = []
    fps_t  = time.time(); fps_cnt = 0; fps_val = 0.0

    # Màu theo class index
    rng = np.random.default_rng(42)
    CLASS_COLORS = {}

    def get_color(cls_id):
        if cls_id not in CLASS_COLORS:
            CLASS_COLORS[cls_id] = tuple(int(c) for c in rng.integers(80, 255, 3))
        return CLASS_COLORS[cls_id]

    try:
        while True:
            frame, _, fid = reader.get_latest()
            now = time.time()

            if frame is None:
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "WAITING FOR STREAM...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(win, blank)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")): break
                time.sleep(0.01); continue

            vis = frame.copy()
            H, W = vis.shape[:2]
            fs = max(0.35, min(0.55, H / 900))

            # ── Inference ──────────────────────────────────────────────────────
            det_iv = 1.0 / args.det_fps if args.det_fps > 0 else 0.0
            if det_iv == 0.0 or now >= next_det_ts:
                next_det_ts = now + det_iv
                try:
                    if args.track:
                        res = model.track(vis, imgsz=args.imgsz, conf=args.conf,
                                          persist=True, verbose=False, iou=0.3)
                    else:
                        res = model.predict(vis, imgsz=args.imgsz, conf=args.conf,
                                            verbose=False)
                except Exception as e:
                    print(f"[ERR] {e}")
                    res = None

                last_items = []
                if res:
                    r = res[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes  = r.boxes.xyxy.cpu().numpy()
                        confs  = r.boxes.conf.cpu().numpy()
                        clss   = r.boxes.cls.cpu().numpy().astype(int)
                        names  = r.names or {}
                        tids   = (r.boxes.id.cpu().numpy().astype(int)
                                  if r.boxes.id is not None else [None] * len(boxes))
                        for i in range(len(boxes)):
                            last_items.append({
                                "bbox":     boxes[i],
                                "conf":     float(confs[i]),
                                "cls":      int(clss[i]),
                                "name":     str(names.get(int(clss[i]), clss[i])),
                                "track_id": tids[i],
                            })

            # ── Vẽ boxes ───────────────────────────────────────────────────────
            for it in last_items:
                x1, y1, x2, y2 = [int(v) for v in it["bbox"]]
                color = get_color(it["cls"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                tid  = it["track_id"]
                tid_str = f"#{tid} " if tid is not None else ""
                lbl  = f"{tid_str}{it['name']}  {it['conf']:.2f}"

                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ly = max(th + 4, y1 - 4)
                cv2.rectangle(vis, (x1, ly - th - 4), (min(W-1, x1 + tw + 6), ly + 2),
                              (0, 0, 0), -1)
                cv2.putText(vis, lbl, (x1 + 3, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

            # ── FPS + count ────────────────────────────────────────────────────
            fps_cnt += 1
            if now - fps_t >= 1.0:
                fps_val = fps_cnt / (now - fps_t)
                fps_cnt = 0; fps_t = now

            info = f"FPS:{fps_val:.1f}  Objects:{len(last_items)}  conf>={args.conf}"
            cv2.rectangle(vis, (4, 4), (len(info)*8 + 10, 24), (0, 0, 0), -1)
            cv2.putText(vis, info, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")): break

    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()