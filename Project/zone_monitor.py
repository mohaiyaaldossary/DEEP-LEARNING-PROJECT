"""Person detection + tracking + restricted zone entry events (YOLO + ByteTrack).
Usage: python zone_monitor.py video.mp4 [--redraw] [--model yolo11n.pt] [--conf 0.3]
"""
import argparse, csv, json, os, time, uuid, platform
from pathlib import Path
from zone_state import ZoneEvents, validate_polygon
import cv2
import numpy as np
from ultralytics import YOLO

EXIT_CONFIRM = 15  # consecutive frames outside before re-arm



# ---------- zone editor ----------
def _ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])

def _seg_intersect(a, b, c, d):
    return (_ccw(a, c, d) > 0) != (_ccw(b, c, d) > 0) and (_ccw(a, b, c) > 0) != (_ccw(a, b, d) > 0)

def validate(pts):
    return validate_polygon(pts)


def edit_zone(frame):
    pts, msg = [], ""
    selected = [None]
    def on_mouse(ev, x, y, flags, _):
        nearest = min(range(len(pts)), key=lambda i:(pts[i][0]-x)**2+(pts[i][1]-y)**2) if pts else None
        close = nearest is not None and (pts[nearest][0]-x)**2+(pts[nearest][1]-y)**2 <= 144
        if ev == cv2.EVENT_LBUTTONDOWN:
            if close:
                selected[0] = nearest
            else:
                pts.append((x,y))
        elif ev == cv2.EVENT_MOUSEMOVE and selected[0] is not None:
            pts[selected[0]] = (max(0,min(x,frame.shape[1]-1)),max(0,min(y,frame.shape[0]-1)))
        elif ev == cv2.EVENT_LBUTTONUP:
            selected[0] = None
        elif ev == cv2.EVENT_RBUTTONDOWN and close:
            pts.pop(nearest)
            selected[0] = None
    cv2.namedWindow("zone"); cv2.setMouseCallback("zone", on_mouse)
    while True:
        img = frame.copy()
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], True, (0, 0, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, f"P{i+1}", (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        help_lines = ["Click:add / drag:move / right:delete", "u:undo / Enter:save / q:quit", msg]
        for line_number,line in enumerate(help_lines):
            cv2.putText(img,line,(6,16+18*line_number),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3)
            cv2.putText(img,line,(6,16+18*line_number),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        cv2.imshow("zone", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == 13:
            err = validate(pts)
            if err:
                msg = "INVALID: " + err
            else:
                break
        elif k == ord("q"):
            raise SystemExit
    cv2.destroyWindow("zone")
    return pts

def load_or_make_zone(video, frame, redraw, headless=False, zone_path=None):
    path = zone_path or os.path.splitext(video)[0] + "_zone.json"
    h, w = frame.shape[:2]
    if os.path.exists(path) and not redraw:
        z = json.load(open(path))
        if z["frame_w"] == w and z["frame_h"] == h and z["video"] == os.path.basename(video):
            err = validate(z["points"])
            if err:
                raise ValueError("Invalid saved zone: " + err)
            if any(not (0 <= x < w and 0 <= y < h) for x,y in z["points"]):
                raise ValueError("Saved zone extends outside frame")
            return z
    if headless:
        raise ValueError("Headless mode requires a valid saved zone for this video")
    pts = edit_zone(frame)
    z = {"zone_name": "Zone A", "video": os.path.basename(video),
         "frame_w": w, "frame_h": h, "points": pts}
    json.dump(z, open(path, "w"), indent=2)
    return z


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--redraw", action="store_true")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--zone", help="Saved polygon JSON")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--exit-confirm", type=int, default=15)
    ap.add_argument("--track-ttl", type=int, default=90)
    ap.add_argument("--save-video", action="store_true")
    a = ap.parse_args()
    if not 0 < a.conf <= 1 or a.imgsz < 32 or a.exit_confirm < 1 or a.track_ttl < 1:
        ap.error("Invalid confidence, image size, or state duration")

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    zone = load_or_make_zone(a.video, first, a.redraw, a.headless, a.zone)
    poly = np.array(zone["points"], dtype=np.int32)

    run_id = "R" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = os.path.join(a.out, run_id); os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "events.csv"), "w", newline="")
    tf = open(os.path.join(out, "tracks.csv"), "w", newline="")
    tw = csv.writer(tf)
    tw.writerow(["frame", "time_seconds", "track_id", "x1", "y1", "x2", "y2", "inside"])
    wr = csv.writer(f)
    wr.writerow(["Run ID", "Event ID", "Video filename", "Elapsed video time", "Track ID",
                 "Zone label", "Event type", "Screenshot path", "Frame", "Time seconds"])

    # video info for report
    info = {"file": os.path.basename(a.video), "fps": fps,
            "w": first.shape[1], "h": first.shape[0],
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "model": a.model, "conf": a.conf,
            "imgsz": a.imgsz, "device": a.device, "tracker": a.tracker,
            "exit_confirm": a.exit_confirm, "track_ttl": a.track_ttl,
            "platform": platform.platform(), "python": platform.python_version()}
    json.dump(info, open(os.path.join(out, "video_info.json"), "w"), indent=2)

    model = YOLO(a.model)
    engine = ZoneEvents(a.exit_confirm, a.track_ttl)
    json.dump(zone, open(os.path.join(out, "zone_used.json"), "w"), indent=2)
    video_writer = None
    if a.save_video:
        video_writer = cv2.VideoWriter(os.path.join(out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1],first.shape[0]))
        if not video_writer.isOpened():
            raise RuntimeError("Cannot create annotated video")
    n_events, frame_idx, t0 = 0, 0, time.time()
    warn_until = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, classes=[0], conf=a.conf, tracker=a.tracker, imgsz=a.imgsz, device=a.device,
                          persist=True, verbose=False)[0]
        observations, boxes = {}, {}
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                x1,y1,x2,y2 = box
                pt = (float((x1+x2)/2),float(y2))
                observations[tid] = cv2.pointPolygonTest(poly,pt,False) >= 0
                boxes[tid] = box
        events_now, any_inside = engine.update(frame_idx,observations)
        event_types = dict(events_now)
        for tid, box in boxes.items():
                x1,y1,x2,y2 = box
                pt = (float((x1+x2)/2),float(y2))
                tw.writerow([frame_idx,frame_idx/fps,tid,*map(float,box),int(observations[tid])])
                etype = event_types.get(tid)
                if etype:
                    n_events += 1
                    eid = f"E{n_events:03d}"
                    shot = os.path.join(out, f"{eid}.jpg")
                    snap = frame.copy()
                    cv2.rectangle(snap, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.polylines(snap, [poly], True, (0, 0, 255), 2)
                    if not cv2.imwrite(shot, snap):
                        raise OSError("Could not save event screenshot")
                    t = frame_idx / fps
                    wr.writerow([run_id, eid, info["file"], f"{int(t//60):02d}:{t%60:04.1f}",
                                 tid, zone["zone_name"], etype, os.path.basename(shot), frame_idx, f"{t:.6f}"])
                    f.flush()
                    warn_until = frame_idx + int(fps)
                col = (0, 0, 255) if observations[tid] else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, col, -1)

        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        if any_inside or frame_idx <= warn_until:
            cv2.putText(frame, "WARNING: RESTRICTED AREA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, min(.9, frame.shape[1]/620), (0, 0, 255), 2)
        frame_idx += 1
        if video_writer is not None:
            video_writer.write(frame)
        if not a.headless:
            cv2.imshow("monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord(" "),ord("q")):
                        break
                if key == ord("q"):
                    break

    el = time.time() - t0
    json.dump({"frames": frame_idx, "seconds": el, "avg_fps": frame_idx/el if el else 0,
               "events": n_events}, open(os.path.join(out, "run_stats.json"), "w"), indent=2)
    f.close(); tf.close(); cap.release()
    if video_writer is not None:
        video_writer.release()
    if not a.headless:
        cv2.destroyAllWindows()
    print(f"{run_id}: {n_events} events, {frame_idx/el:.1f} FPS -> {out}")

if __name__ == "__main__":
    main()
