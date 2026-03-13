# zone_editor_rtsp.py (AUTO-LOAD + EDIT SAVED ZONES + AUTO-CLOSE SNAP)
import os, json, time, argparse
import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;500000")

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def clamp_pt(pt, w, h):
    x = max(0, min(w-1, int(pt[0])))
    y = max(0, min(h-1, int(pt[1])))
    return (x, y)

def point_in_poly(pt_xy, poly_pts):
    poly = np.array(poly_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt_xy[0]), float(pt_xy[1])), False) >= 0

class ZoneEditor:
    def __init__(self, file_path):
        self.file_path = file_path

        # zones stored in memory as pixels (easy edit)
        # each: {"name": str, "points_px": [(x,y),...]}
        self.zones = []

        # currently drawing zone (new)
        self.cur_zone = {"name": None, "points_px": []}

        self.dragging = False
        self.drag_target = None  # ("saved", zone_idx, pt_idx) or ("cur", pt_idx)

        self.freeze = True
        self.sel_r = 14   # radius to pick/drag point
        self.snap_r = 18  # radius to auto-close near first point

        self.w = None
        self.h = None
        self.mouse_xy = (0, 0)

        # selected zone index (for delete / highlight). None means no selection.
        self.selected_zone = None

    # ---------- load / save ----------
    def load_if_exists(self, frame_w, frame_h):
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            zones = data.get("zones", [])
            loaded = []
            for z in zones:
                name = z.get("name", f"zone{len(loaded)+1}")
                pts_norm = z.get("points", [])
                pts_px = []
                for p in pts_norm:
                    if not isinstance(p, (list, tuple)) or len(p) != 2:
                        continue
                    x = int(float(p[0]) * frame_w)
                    y = int(float(p[1]) * frame_h)
                    pts_px.append(clamp_pt((x, y), frame_w, frame_h))
                if len(pts_px) >= 3:
                    loaded.append({"name": name, "points_px": pts_px})
            self.zones = loaded
            if self.zones:
                self.selected_zone = 0
            print(f"[OK] Loaded {len(self.zones)} zones from {self.file_path}")
        except Exception as e:
            print(f"[WARN] Cannot load zones file: {e}")

    def save(self, frame_w, frame_h):
        data = {
            "version": 1,
            "frame_size": {"width": frame_w, "height": frame_h},
            "zones": []
        }
        for z in self.zones:
            pts_norm = [[p[0]/frame_w, p[1]/frame_h] for p in z["points_px"]]
            data["zones"].append({"name": z["name"], "points": pts_norm})

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved zones -> {self.file_path}")

    # ---------- zone ops ----------
    def start_new_zone(self):
        zid = len(self.zones) + 1
        self.cur_zone = {"name": f"zone{zid}", "points_px": []}
        self.dragging = False
        self.drag_target = None

    def finalize_cur_zone(self):
        pts = self.cur_zone["points_px"]
        if len(pts) >= 3:
            self.zones.append({"name": self.cur_zone["name"], "points_px": pts[:]})
            self.selected_zone = len(self.zones) - 1
            self.start_new_zone()

    def delete_selected_zone(self):
        if self.selected_zone is None:
            return
        if 0 <= self.selected_zone < len(self.zones):
            self.zones.pop(self.selected_zone)
            if not self.zones:
                self.selected_zone = None
            else:
                self.selected_zone = min(self.selected_zone, len(self.zones)-1)

    def select_zone_by_point(self, x, y):
        # If click inside polygon -> select that zone
        for idx, z in enumerate(self.zones):
            if len(z["points_px"]) >= 3 and point_in_poly((x, y), z["points_px"]):
                self.selected_zone = idx
                return idx
        return None

    # ---------- geometry helpers ----------
    def nearest_point_index(self, pts, x, y, radius):
        if not pts:
            return None
        best_i, best_d = None, 10**18
        for i, p in enumerate(pts):
            d = dist2(p, (x, y))
            if d < best_d:
                best_d = d
                best_i = i
        if best_d <= radius * radius:
            return best_i
        return None

    def is_near_first_point(self, pts, x, y):
        if len(pts) < 3:
            return False
        return dist2(pts[0], (x, y)) <= self.snap_r * self.snap_r

    # ---------- mouse ----------
    def mouse_cb(self, event, x, y, flags, param):
        if self.w is None or self.h is None:
            return

        self.mouse_xy = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            cur_pts = self.cur_zone["points_px"]

            # 0) Auto-close CURRENT zone FIRST (ưu tiên trước drag)
            if self.is_near_first_point(cur_pts, x, y):
                self.finalize_cur_zone()
                return

            # 1) Try pick any point in CURRENT zone (drag)
            idx = self.nearest_point_index(cur_pts, x, y, self.sel_r)
            if idx is not None:
                self.dragging = True
                self.drag_target = ("cur", idx)
                return

            # 2) Try pick any point in SAVED zones (drag)
            for zi, z in enumerate(self.zones):
                pts = z["points_px"]
                pi = self.nearest_point_index(pts, x, y, self.sel_r)
                if pi is not None:
                    self.selected_zone = zi
                    self.dragging = True
                    self.drag_target = ("saved", zi, pi)
                    return

            # 3) If click inside a SAVED zone polygon -> select it
            picked = self.select_zone_by_point(x, y)
            if picked is not None:
                return

            # 4) Else add point to CURRENT zone
            cur_pts.append(clamp_pt((x, y), self.w, self.h))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_target is not None:
                if self.drag_target[0] == "cur":
                    _, pi = self.drag_target
                    self.cur_zone["points_px"][pi] = clamp_pt((x, y), self.w, self.h)
                else:
                    _, zi, pi = self.drag_target
                    self.zones[zi]["points_px"][pi] = clamp_pt((x, y), self.w, self.h)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_target = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.cur_zone["points_px"]) >= 3:
                self.finalize_cur_zone()
    # ---------- draw ----------
    def draw(self, frame):
        h, w = frame.shape[:2]
        self.w, self.h = w, h

        overlay = frame.copy()

        # Draw saved zones (highlight selected)
        for idx, z in enumerate(self.zones):
            pts = np.array(z["points_px"], dtype=np.int32)
            is_sel = (self.selected_zone == idx)
            color = (0, 200, 255) if is_sel else (0, 255, 255)
            thick = 3 if is_sel else 2
            cv2.polylines(overlay, [pts], True, color, thick)

            if z["points_px"]:
                tx, ty = z["points_px"][0]
                cv2.putText(overlay, z["name"], (tx+6, ty-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # points
            for (px, py) in z["points_px"]:
                cv2.circle(overlay, (px, py), 5, color, -1)

        # Draw current zone (in progress)
        pts = self.cur_zone["points_px"]
        if pts:
            p = np.array(pts, dtype=np.int32)
            cv2.polylines(overlay, [p], False, (255, 0, 255), 2)
            for i, (px, py) in enumerate(pts):
                cv2.circle(overlay, (px, py), 6, (255, 0, 255), -1)
                cv2.putText(overlay, str(i+1), (px+8, py+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,255), 1, cv2.LINE_AA)

            mx, my = self.mouse_xy
            last = pts[-1]

            # Preview snap close line
            if self.is_near_first_point(pts, mx, my):
                cv2.circle(overlay, pts[0], self.snap_r, (255, 0, 255), 2)
                cv2.line(overlay, last, pts[0], (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, "Click near FIRST point to CLOSE", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2, cv2.LINE_AA)
            else:
                cv2.line(overlay, last, (mx, my), (200, 0, 200), 1, cv2.LINE_AA)

        # UI
        help1 = "SPACE: Freeze/Live | N: New zone | TAB: Select next | DEL/BS: Delete selected | S: Save | ESC/Q: Quit"
        help2 = "Left click: add point | Drag any point: move | Click near FIRST point: auto-close | Right click: close"
        cv2.rectangle(overlay, (10, 10), (w-10, 78), (0,0,0), -1)
        cv2.putText(overlay, help1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, help2, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)

        sel_name = self.zones[self.selected_zone]["name"] if (self.selected_zone is not None and self.zones) else "None"
        status = f"MODE: {'FREEZE' if self.freeze else 'LIVE'} | Drawing: {self.cur_zone['name']} pts={len(pts)} | Selected: {sel_name} | saved={len(self.zones)}"
        cv2.putText(overlay, status, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(overlay, status, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", default="rtsp://admin:KOEYHZ@192.168.2.72:554/cam/realmonitor?channel=1&subtype=1")
    ap.add_argument("--file", default="zones.json", help="Load/save zones json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Cannot open RTSP")

    # Grab first frame to know W/H for loading normalized coords
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Cannot read first frame from RTSP")
    H, W = first.shape[:2]

    editor = ZoneEditor(args.file)
    editor.load_if_exists(W, H)
    editor.start_new_zone()

    win = "Zone Editor (RTSP)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, editor.mouse_cb)

    last_frame = first

    while True:
        if (not editor.freeze) or (last_frame is None):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue
            last_frame = frame
        else:
            frame = last_frame

        vis = editor.draw(frame)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            editor.freeze = not editor.freeze
        elif key in (ord('n'), ord('N')):
            editor.start_new_zone()
        elif key == 9:  # TAB
            if editor.zones:
                if editor.selected_zone is None:
                    editor.selected_zone = 0
                else:
                    editor.selected_zone = (editor.selected_zone + 1) % len(editor.zones)
        elif key in (ord('s'), ord('S')):
            h, w = last_frame.shape[:2]
            editor.save(w, h)
        elif key in (8, 127):  # Backspace / Delete
            editor.delete_selected_zone()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()