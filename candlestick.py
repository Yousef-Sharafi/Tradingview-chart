

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from tkcalendar import DateEntry
    HAS_CAL = True
except Exception:
    HAS_CAL = False

# ---------- Helpers ----------
def is_intraday(interval: str) -> bool:
    interval = (interval or "").lower().strip()
    return interval.endswith("m") or interval.endswith("h")

def safe_parse_date(s: str):
    try:
        return pd.to_datetime(s)
    except Exception:
        return None

# ---------- Data ----------
def fetch_prices(ticker, timeframe, start_str, end_str):
    try:
        import yfinance as yf
    except Exception:
        messagebox.showerror("Missing dependency", "Please install yfinance:\n\npip install yfinance")
        return None

    interval = timeframe.strip()
    start_dt = safe_parse_date(start_str.strip()) if start_str else None
    end_dt   = safe_parse_date(end_str.strip()) if end_str else None

    default_period_map = {
        "1m": "7d", "2m": "14d", "5m": "30d", "15m": "60d", "30m": "60d",
        "1h": "3mo", "2h": "6mo", "4h": "1y", "1d": "1y", "1wk": "5y", "1mo": "10y",
    }
    period = default_period_map.get(interval, "1y")

    if start_dt and end_dt and is_intraday(interval):
        max_days = 729
        if (end_dt - start_dt).days > max_days:
            start_dt = end_dt - pd.Timedelta(days=max_days)

    try:
        if start_dt is not None and end_dt is not None:
            df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"),
                             end=end_dt.strftime("%Y-%m-%d"),
                             interval=interval, auto_adjust=False, progress=False)
        else:
            if is_intraday(interval) and period.endswith("y"):
                period = "730d"
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=False, progress=False)
    except Exception as e:
        if is_intraday(interval):
            try:
                df = yf.download(ticker, period="730d", interval=interval,
                                 auto_adjust=False, progress=False)
            except Exception as ee:
                messagebox.showerror("Download error", f"Failed to download: {ee}")
                return None
        else:
            messagebox.showerror("Download error", f"Failed to download: {e}")
            return None

    if df is None or df.empty:
        messagebox.showwarning("No data", "No data returned. Check ticker/timeframe/date range.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if x]) for col in df.columns]

    def find(cols, key):
        for c in cols:
            if key in str(c).lower():
                return c
        return None

    cols = list(df.columns)
    o = find(cols, "open"); h = find(cols, "high")
    l = find(cols, "low");  c = find(cols, "close")
    if not all([o, h, l, c]):
        messagebox.showerror("Format error", "Downloaded data missing OHLC columns.")
        return None

    out = df[[o, h, l, c]].copy()
    out.columns = ["Open", "High", "Low", "Close"]
    return out

# ---------- Geometry helpers ----------
def dist_point_to_segment_pixels(ax, x0, y0, x1, y1, xp, yp):
    trans = ax.transData.transform
    (X0, Y0) = trans((x0, y0)); (X1, Y1) = trans((x1, y1)); (XP, YP) = trans((xp, yp))
    vx, vy = X1 - X0, Y1 - Y0
    wx, wy = XP - X0, YP - Y0
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return np.hypot(XP - X0, YP - Y0)
    t = max(0, min(1, (wx*vx + wy*vy) / seg_len2))
    projx, projy = X0 + t*vx, Y0 + t*vy
    return np.hypot(XP - projx, YP - projy)

def dist_point_to_point_pixels(ax, xa, ya, xb, yb):
    trans = ax.transData.transform
    (XA, YA) = trans((xa, ya)); (XB, YB) = trans((xb, yb))
    return np.hypot(XA - XB, YA - YB)

# ---------- Items ----------
class LineItem:
    def __init__(self, ax, x0, y0, x1, y1, color="black", lw=2.0):
        self.ax = ax
        (self.line,) = ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw)
        (self.h0,) = ax.plot([x0], [y0], marker='o', linestyle='None', markersize=7, alpha=0.0, color=color)
        (self.h1,) = ax.plot([x1], [y1], marker='o', linestyle='None', markersize=7, alpha=0.0, color=color)
        self.selected = False; self.hovered = False
        self.color = color; self.linewidth = lw
        self.auto_extend = True

    def get_data(self):
        xd, yd = self.line.get_data(); return [xd[0], xd[1]], [yd[0], yd[1]]
    def set_data(self, x0, y0, x1, y1):
        self.line.set_data([x0, x1], [y0, y1]); self.h0.set_data([x0], [y0]); self.h1.set_data([x1], [y1])
    def set_color(self, color):
        self.color = color; self.line.set_color(color); self.h0.set_color(color); self.h1.set_color(color)
    def set_linewidth(self, lw):
        self.linewidth = float(lw); self._apply_style()
    def set_selected(self, val):
        self.selected = val; self._apply_style()
    def set_hovered(self, val):
        self.hovered = val; self._apply_style()
    def _apply_style(self):
        base = self.linewidth
        lw = base + (0.8 if (self.hovered or self.selected) else 0.0)
        self.line.set_linewidth(lw)
        alpha = 1.0 if (self.hovered or self.selected) else 0.0
        self.h0.set_alpha(alpha); self.h1.set_alpha(alpha)
    def remove(self):
        self.line.remove(); self.h0.remove(); self.h1.remove()

class HLineItem:
    def __init__(self, ax, y, xmin, xmax, color="black", lw=2.0):
        self.ax = ax
        (self.line,) = ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw)
        (self.hc,) = ax.plot([(xmin+xmax)/2], [y], marker='s', linestyle='None', markersize=7, alpha=0.0, color=color)
        self.selected = False; self.hovered = False
        self.color = color; self.linewidth = lw

    def get_data(self):
        xd, yd = self.line.get_data(); return [xd[0], xd[1]], [yd[0], yd[1]]
    def set_y(self, y):
        xd, _ = self.line.get_data()
        self.line.set_data(xd, [y, y]); self.hc.set_data([(xd[0]+xd[1])/2], [y])
    def set_span(self, xmin, xmax):
        y = self.line.get_data()[1][0]
        self.line.set_data([xmin, xmax], [y, y]); self.hc.set_data([(xmin+xmax)/2], [y])
    def set_color(self, color):
        self.color = color; self.line.set_color(color); self.hc.set_color(color)
    def set_linewidth(self, lw):
        self.linewidth = float(lw); self._apply_style()
    def set_selected(self, val):
        self.selected = val; self._apply_style()
    def set_hovered(self, val):
        self.hovered = val; self._apply_style()
    def _apply_style(self):
        base = self.linewidth
        lw = base + (0.8 if (self.hovered or self.selected) else 0.0)
        self.line.set_linewidth(lw)
        alpha = 1.0 if (self.hovered or self.selected) else 0.0
        self.hc.set_alpha(alpha)
    def remove(self):
        self.line.remove(); self.hc.remove()

class PenStroke:
    def __init__(self, ax, color="black"):
        self.ax = ax
        (self.line,) = ax.plot([], [], color=color, linewidth=1.8)
        self.color = color
    def add_point(self, x, y):
        xd, yd = self.line.get_data(); xdl = list(xd); ydl = list(yd)
        xdl.append(x); ydl.append(y); self.line.set_data(xdl, ydl)
    def get_data(self):
        xd, yd = self.line.get_data(); return list(xd), list(yd)
    def set_color(self, c):
        self.color = c; self.line.set_color(c)
    def remove(self): self.line.remove()

# ---------- Tools ----------
def dist_point_to_segment_pixels(ax, x0, y0, x1, y1, xp, yp):  # already defined above; kept to avoid NameError
    trans = ax.transData.transform
    (X0, Y0) = trans((x0, y0)); (X1, Y1) = trans((x1, y1)); (XP, YP) = trans((xp, yp))
    vx, vy = X1 - X0, Y1 - Y0
    wx, wy = XP - X0, YP - Y0
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return np.hypot(XP - X0, YP - Y0)
    t = max(0, min(1, (wx*vx + wy*vy) / seg_len2))
    projx, projy = X0 + t*vx, Y0 + t*vy
    return np.hypot(XP - projx, YP - projy)

def dist_point_to_point_pixels(ax, xa, ya, xb, yb):  # already defined above; kept to avoid NameError
    trans = ax.transData.transform
    (XA, YA) = trans((xa, ya)); (XB, YB) = trans((xb, yb))
    return np.hypot(XA - XB, YA - YB)

class DrawEditLineTool:
    HANDLE_RADIUS_PX = 12
    LINE_PICK_RADIUS_PX = 10
    def __init__(self, ax, snap_x, get_current_color_cb, get_current_lw_cb):
        self.ax = ax
        self.snap_x = snap_x
        self.get_color = get_current_color_cb
        self.get_lw = get_current_lw_cb
        self.lines = []; self.hlines = []
        self.creating_start = None; self.preview = None
        self.hover_item=None; self.selected_item=None
        self.dragging=False; self.drag_mode=None; self.drag_idx=None; self.last_xy=None
        self.mode = "pointer"
        c = ax.figure.canvas
        self.cid_press   = c.mpl_connect("button_press_event", self.on_press)
        self.cid_move    = c.mpl_connect("motion_notify_event", self.on_move)
        self.cid_release = c.mpl_connect("button_release_event", self.on_release)

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "line":
            self._clear_preview(); self.creating_start=None

    def clear_all(self):
        for it in self.lines: it.remove()
        for it in self.hlines: it.remove()
        self.lines.clear(); self.hlines.clear()
        self.hover_item=None; self.selected_item=None
        self._clear_preview(); self.creating_start=None
        self.ax.figure.canvas.draw_idle()

    def delete_selected(self):
        if self.selected_item is None: return
        self._delete_item(self.selected_item); self.selected_item=None; self.ax.figure.canvas.draw_idle()

    def delete_at(self, x, y):
        item, _ = self._hit_test(x, y, hover=False)
        if item is None: return False
        self._delete_item(item)
        if self.selected_item is item: self.selected_item=None
        self.ax.figure.canvas.draw_idle(); return True

    def _delete_item(self, item):
        if isinstance(item, LineItem):
            item.remove(); self.lines = [l for l in self.lines if l is not item]
        else:
            item.remove(); self.hlines = [h for h in self.hlines if h is not item]

    def apply_color_to_selected(self, color: str):
        if self.selected_item is None: return
        self.selected_item.set_color(color); self.ax.figure.canvas.draw_idle()

    def apply_lw_to_selected(self, lw: float):
        if self.selected_item is None: return
        self.selected_item.set_linewidth(lw); self.ax.figure.canvas.draw_idle()

    def _clear_preview(self):
        if self.preview is not None: self.preview.remove(); self.preview=None

    def _set_hover(self, it_new):
        if self.hover_item is it_new: return
        if self.hover_item is not None and self.hover_item is not self.selected_item:
            self.hover_item.set_hovered(False)
        self.hover_item = it_new
        if self.hover_item is not None: self.hover_item.set_hovered(True)

    def _set_selected(self, it_new):
        if self.selected_item is it_new: return
        if self.selected_item is not None: self.selected_item.set_selected(False)
        self.selected_item = it_new
        if self.selected_item is not None: self.selected_item.set_selected(True)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3:
            self.delete_at(event.xdata, event.ydata); return
        if event.button != 1 or self.dragging: return

        if self.mode == "hline":
            xlim = self.ax.get_xlim()
            color = self.get_color(); lw = float(self.get_lw())
            it = HLineItem(self.ax, event.ydata, xlim[0], xlim[1], color=color, lw=lw)
            self.hlines.append(it); self._set_selected(it)
            self.ax.figure.canvas.draw_idle(); return

        if self.mode == "line":
            if self.creating_start is None:
                xi = self.snap_x(event.xdata); 
                if xi is None: return
                self.creating_start=(xi, event.ydata)
                self._clear_preview()
                (self.preview,) = self.ax.plot([xi, xi], [event.ydata, event.ydata],
                                               color=self.get_color(), linewidth=float(self.get_lw()))
                self.ax.figure.canvas.draw_idle(); return
            else:
                xi = self.snap_x(event.xdata); 
                if xi is None: return
                x0,y0 = self.creating_start; x1,y1 = xi, event.ydata
                self._clear_preview()
                it = LineItem(self.ax, x0,y0,x1,y1, color=self.get_color(), lw=float(self.get_lw()))
                self.lines.append(it); self._set_selected(it)
                self.creating_start=None; self.ax.figure.canvas.draw_idle(); return

        it_hit, which = self._hit_test(event.xdata, event.ydata)
        if it_hit is not None:
            self._set_selected(it_hit); self.dragging=True; self.last_xy=(event.xdata, event.ydata)
            if isinstance(it_hit, LineItem):
                self.drag_mode = "endpoint" if which in ("h0","h1") else "line"
                self.drag_idx  = 0 if which=="h0" else (1 if which=="h1" else None)
            else:
                self.drag_mode="hline"; self.drag_idx=None

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.mode=="line" and self.creating_start is not None and self.preview is not None:
            xi=self.snap_x(event.xdata); 
            if xi is None: return
            x0,y0=self.creating_start; self.preview.set_data([x0, xi],[y0, event.ydata])
            self.ax.figure.canvas.draw_idle()

        if not self.dragging:
            it_hover,_ = self._hit_test(event.xdata, event.ydata, hover=True)
            self._set_hover(it_hover); self.ax.figure.canvas.draw_idle()

        if self.dragging and self.selected_item is not None and event.xdata is not None and event.ydata is not None:
            dx = event.xdata - self.last_xy[0]; dy = event.ydata - self.last_xy[1]
            if isinstance(self.selected_item, LineItem):
                xdata,ydata = self.selected_item.get_data()
                if self.drag_mode=="line":
                    nx0=self.snap_x(xdata[0]+dx); nx1=self.snap_x(xdata[1]+dx)
                    if nx0 is None or nx1 is None: return
                    ny0=ydata[0]+dy; ny1=ydata[1]+dy
                    self.selected_item.set_data(nx0,ny0,nx1,ny1)
                else:
                    nx=[xdata[0],xdata[1]]; ny=[ydata[0],ydata[1]]
                    nx[self.drag_idx]=self.snap_x(xdata[self.drag_idx]+dx)
                    ny[self.drag_idx]=ydata[self.drag_idx]+dy
                    if nx[self.drag_idx] is None: return
                    self.selected_item.set_data(nx[0],ny[0],nx[1],ny[1])
                self.selected_item.set_selected(True); self.selected_item.set_hovered(True)
            else:
                xd, yd = self.selected_item.get_data()
                self.selected_item.set_y(yd[0] + dy)
                self.selected_item.set_selected(True); self.selected_item.set_hovered(True)
            self.last_xy=(event.xdata, event.ydata); self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging:
            self.dragging=False; self.drag_mode=None; self.drag_idx=None; self.last_xy=None
            self.ax.figure.canvas.draw_idle()

    def _hit_test(self, x, y, hover=False):
        if x is None or y is None: return None, None
        handle_r = 12 + (4 if hover else 0)
        line_r   = 10 + (2 if hover else 0)

        best=(None,None,1e12)
        for li in self.lines:
            xd,yd=li.get_data()
            d0=dist_point_to_point_pixels(self.ax, xd[0], yd[0], x, y)
            d1=dist_point_to_point_pixels(self.ax, xd[1], yd[1], x, y)
            if d0<best[2]: best=(li,"h0",d0)
            if d1<best[2]: best=(li,"h1",d1)
        if best[2] <= handle_r: return best[0], best[1]

        best=(None,None,1e12)
        for li in self.lines:
            xd,yd=li.get_data()
            d=dist_point_to_segment_pixels(self.ax, xd[0], yd[0], xd[1], yd[1], x, y)
            if d<best[2]: best=(li,"line",d)
        for hi in self.hlines:
            xd,yd=hi.get_data()
            d=dist_point_to_segment_pixels(self.ax, xd[0], yd[0], xd[1], yd[1], x, y)
            if d<best[2]: best=(hi,"line",d)
        if best[2] <= line_r: return best[0], "line"
        return None, None

    def snapshot(self):
        snap = {"lines": [], "hlines": []}
        for li in self.lines:
            (xd, yd) = li.get_data()
            snap["lines"].append({"x0": xd[0], "y0": yd[0], "x1": xd[1], "y1": yd[1],
                                  "color": li.color, "lw": li.linewidth, "auto_extend": li.auto_extend})
        for hi in self.hlines:
            (xd, yd) = hi.get_data()
            snap["hlines"].append({"y": yd[0], "color": hi.color, "lw": hi.linewidth})
        return snap

    def restore(self, snap, chart_len):
        self.lines=[]; self.hlines=[]; self.hover_item=None; self.selected_item=None
        for rec in snap.get("lines", []):
            li = LineItem(self.ax, rec["x0"], rec["y0"], rec["x1"], rec["y1"],
                          color=rec.get("color","black"), lw=rec.get("lw",2.0))
            li.auto_extend = rec.get("auto_extend", True)
            self.lines.append(li)
        xmin, xmax = 0, max(0, chart_len-1)
        for rec in snap.get("hlines", []):
            hi = HLineItem(self.ax, rec["y"], xmin, xmax,
                           color=rec.get("color","black"), lw=rec.get("lw",2.0))
            self.hlines.append(hi)
        self.ax.figure.canvas.draw_idle()

    def span_to_xlim(self):
        xmin, xmax = self.ax.get_xlim()
        right = int(round(xmax))
        for li in self.lines:
            (xd, yd) = li.get_data()
            x0, x1 = xd[0], xd[1]; y0, y1 = yd[0], yd[1]
            if not li.auto_extend: continue
            if x1 == x0:
                li.set_data(x0, y0, right, y1)
            else:
                m = (y1 - y0) / (x1 - x0)
                new_y1 = y0 + m * (right - x0)
                li.set_data(x0, y0, right, new_y1)
        for hi in self.hlines:
            hi.set_span(int(round(xmin)), right)

class PenTool:
    def __init__(self, ax, get_current_color_cb):
        self.ax=ax; self.get_color=get_current_color_cb
        self.active=False; self.drawing=False; self.current=None; self.strokes=[]
        c = ax.figure.canvas
        self.cid_press  = c.mpl_connect("button_press_event", self.on_press)
        self.cid_move   = c.mpl_connect("motion_notify_event", self.on_move)
        self.cid_release= c.mpl_connect("button_release_event", self.on_release)

    def set_active(self, val: bool):
        self.active=val
        if not val and self.drawing:
            if self.current is not None: self.current.remove(); self.current=None
            self.drawing=False; self.ax.figure.canvas.draw_idle()

    def clear_all(self):
        for s in self.strokes: s.remove()
        self.strokes.clear()
        if self.current is not None: self.current.remove(); self.current=None
        self.drawing=False; self.ax.figure.canvas.draw_idle()

    def delete_at(self, x, y):
        if not self.strokes: return False
        best=(None,1e12)
        for s in self.strokes:
            xd, yd = s.get_data()
            if len(xd) == 0: continue
            cx, cy = np.mean(xd), np.mean(yd)
            d=dist_point_to_point_pixels(self.ax, cx, cy, x, y)
            if d<best[1]: best=(s,d)
        if best[0] is not None and best[1] <= 25:
            best[0].remove()
            self.strokes = [st for st in self.strokes if st is not best[0]]
            self.ax.figure.canvas.draw_idle()
            return True
        return False

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3:
            self.delete_at(event.xdata, event.ydata); return
        if not self.active or event.button != 1: return
        self.drawing=True
        color=self.get_color()
        self.current = PenStroke(self.ax, color=color)
        self.current.add_point(event.xdata, event.ydata)
        self.ax.figure.canvas.draw_idle()

    def on_move(self, event):
        if not self.active or not self.drawing or event.inaxes != self.ax: return
        self.current.add_point(event.xdata, event.ydata)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if not self.active or event.inaxes != self.ax:
            self.drawing=False; return
        if event.button != 1: return
        if self.current is not None:
            self.strokes.append(self.current); self.current=None
        self.drawing=False; self.ax.figure.canvas.draw_idle()

    def snapshot(self):
        snap = []
        for s in self.strokes:
            xd, yd = s.get_data(); snap.append({"x": xd, "y": yd, "color": s.color})
        return snap

    def restore(self, snap):
        self.strokes = []
        for rec in snap:
            ps = PenStroke(self.ax, color=rec.get("color","black"))
            ps.line.set_data(rec["x"], rec["y"])
            self.strokes.append(ps)
        self.ax.figure.canvas.draw_idle()

# ---------- Chart drawing ----------
def draw_candles(ax, df, log_scale=False, style="candle", bar_color="black",
                 x_full_len=None, show_upto_len=None):
    ax.clear()
    df = df.sort_index()
    n = len(df)
    x = np.arange(n)

    draw_n = n if (show_upto_len is None) else min(show_upto_len, n)
    xd = x[:draw_n]; slice_df = df.iloc[:draw_n]

    if draw_n > 0:
        o = slice_df["Open"].to_numpy()
        h = slice_df["High"].to_numpy()
        l = slice_df["Low"].to_numpy()
        c = slice_df["Close"].to_numpy()

        if style == "candle":
            up = (c >= o); down = ~up
            ax.vlines(xd[up], l[up], h[up], colors="green")
            ax.vlines(xd[down], l[down], h[down], colors="red")
            ax.bar(xd[up], (c[up]-o[up]), bottom=o[up], width=0.6, align="center", color="green")
            ax.bar(xd[down], (c[down]-o[down]), bottom=o[down], width=0.6, align="center", color="red")
        else:
            for i in range(draw_n):
                xi = xd[i]
                ax.vlines(xi, l[i], h[i], colors=bar_color)
                tw = 0.25
                ax.hlines(o[i], xi - tw, xi, colors=bar_color)
                ax.hlines(c[i], xi, xi + tw, colors=bar_color)

    # X ticks concise: YY-MM-DD
    step = max(1, max(1, n)//6)
    xticks = np.arange(0, n, step) if n else []
    if n:
        idx = pd.to_datetime(df.index)
        xticklabels = [idx[i].strftime("%y-%m-%d") for i in xticks]
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    else:
        ax.set_xticks([]); ax.set_xticklabels([])

    ax.set_xlabel(""); ax.set_ylabel("Price"); ax.grid(True, linestyle=":")
    try: ax.set_yscale("log" if log_scale else "linear")
    except Exception: ax.set_yscale("linear")

    if x_full_len is not None:
        ax.set_xlim(0, max(0, x_full_len-1))
    else:
        ax.set_xlim(0, max(0, n-1))

    def snap_x(xf):
        if xf is None or np.isnan(xf): return None
        i = int(np.round(xf))
        limit = (x_full_len-1) if x_full_len is not None else (n-1)
        limit = max(0, limit)
        return max(0, min(limit, i))

    ax.figure.canvas.draw_idle()
    return snap_x

# ---------- Main App ----------
class TVStyleApp:
    TIMEFRAMES = ["1m","2m","5m","15m","30m","1h","2h","4h","1d","1wk","1mo"]
    COLORS = ["red","green","black","blue"]
    LINEWIDTHS = ["1","2","3","4"]
    SPEEDS = {"Slow": 100, "Medium": 50, "Fast": 5}
    CHART_TYPES = ["Candlestick","Bar"]
    BAR_COLORS = ["black","blue","red","green","gray"]
    TICKERS_30 = [
        "EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURJPY=X",
        "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","TON-USD","AVAX-USD",
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","UNH","XOM","AMD",
        "XAUUSD=X",
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("TradingView-style Chart (Tk + Matplotlib)")
        try: self.root.state("zoomed")
        except Exception: self.root.attributes("-zoomed", True)

        today = dt.date.today()
        default_start = (today - dt.timedelta(days=365*2)).strftime("%Y-%m-%d")
        default_temp  = (today - dt.timedelta(days=365*1)).strftime("%Y-%m-%d")
        default_end   = today.strftime("%Y-%m-%d")

        # --- Top control bar ---
        top = ttk.Frame(root, padding=(6,6)); top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Line Color:").pack(side=tk.LEFT, padx=(0,4))
        self.color_var = tk.StringVar(value="black")
        self.color_cb = ttk.Combobox(top, values=self.COLORS, textvariable=self.color_var, width=8, state="readonly")
        self.color_cb.pack(side=tk.LEFT, padx=(0,8))
        self.color_cb.bind("<<ComboboxSelected>>", self.on_color_change)

        ttk.Label(top, text="Line Width:").pack(side=tk.LEFT, padx=(0,4))
        self.lw_var = tk.StringVar(value="2")
        self.lw_cb = ttk.Combobox(top, values=self.LINEWIDTHS, textvariable=self.lw_var, width=4, state="readonly")
        self.lw_cb.pack(side=tk.LEFT, padx=(0,12))
        self.lw_cb.bind("<<ComboboxSelected>>", self.on_lw_change)

        ttk.Label(top, text="Ticker:").pack(side=tk.LEFT, padx=(0,4))
        self.ticker_var = tk.StringVar(value="BTC-USD")
        self.ticker_cb = ttk.Combobox(top, values=self.TICKERS_30, textvariable=self.ticker_var, width=14)
        self.ticker_cb.pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(top, text="Timeframe:").pack(side=tk.LEFT, padx=(0,4))
        self.tf_var = tk.StringVar(value="4h")
        self.tf_cb = ttk.Combobox(top, values=self.TIMEFRAMES, textvariable=self.tf_var, width=6, state="readonly")
        self.tf_cb.pack(side=tk.LEFT, padx=(0,12))

        self.log_var = tk.BooleanVar(value=False)
        self.log_cb = ttk.Checkbutton(top, text="Log scale", variable=self.log_var, command=self.apply_refresh)
        self.log_cb.pack(side=tk.LEFT, padx=(0,12))

        ttk.Label(top, text="Start:").pack(side=tk.LEFT, padx=(0,4))
        if HAS_CAL:
            self.start_var = tk.StringVar(value=default_start)
            self.start_entry = DateEntry(top, textvariable=self.start_var, width=12, date_pattern="yyyy-mm-dd")
        else:
            self.start_var = tk.StringVar(value=default_start)
            self.start_entry = ttk.Entry(top, textvariable=self.start_var, width=12)
        self.start_entry.pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(top, text="End:").pack(side=tk.LEFT, padx=(0,4))
        if HAS_CAL:
            self.end_var = tk.StringVar(value=default_end)
            self.end_entry = DateEntry(top, textvariable=self.end_var, width=12, date_pattern="yyyy-mm-dd")
        else:
            self.end_var = tk.StringVar(value=default_end)
            self.end_entry = ttk.Entry(top, textvariable=self.end_var, width=12)
        self.end_entry.pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(top, text="Temp:").pack(side=tk.LEFT, padx=(0,4))
        if HAS_CAL:
            self.temp_var = tk.StringVar(value=default_temp)
            self.temp_entry = DateEntry(top, textvariable=self.temp_var, width=12, date_pattern="yyyy-mm-dd")
        else:
            self.temp_var = tk.StringVar(value=default_temp)
            self.temp_entry = ttk.Entry(top, textvariable=self.temp_var, width=12)
        self.temp_entry.pack(side=tk.LEFT, padx=(0,8))

        self.apply_btn = ttk.Button(top, text="Apply / Refresh", command=self.apply_refresh)
        self.apply_btn.pack(side=tk.LEFT, padx=(8,0))

        self.replay_btn = ttk.Button(top, text="Replay ▶", command=self.on_replay_button)
        self.replay_btn.pack(side=tk.LEFT, padx=(6,0))

        # Figure & toolbar
        self.fig = plt.Figure(figsize=(12,7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root); self.toolbar.update()

        # Tools
        self.snap_x = None
        self.line_tool = DrawEditLineTool(self.ax, snap_x=lambda x: None,
                                          get_current_color_cb=self.get_current_color,
                                          get_current_lw_cb=self.get_current_lw)
        self.pen_tool  = PenTool(self.ax, get_current_color_cb=self.get_current_color)

        self.canvas_widget.bind("<Button-3>", self.on_right_click_canvas)
        self.root.bind("<Delete>", self.on_delete_key)

        # ***** Initialize menu variables BEFORE building menus *****
        self.ctype_var = tk.StringVar(value="Candlestick")
        self.barcolor_var = tk.StringVar(value="black")
        self.speed_var = tk.StringVar(value="Medium")
        # ***********************************************************

        # Menus
        self._build_menus()

        # Replay state
        self.full_df = None
        self.current_df = None
        self.future_df = None
        self.replay_index = 0
        self.replay_state = "stopped"  # "stopped" | "playing" | "paused"

        # Initial draw
        self.apply_refresh()

    # ---- UI helpers ----
    def get_current_color(self):
        val = self.color_var.get().strip().lower()
        return val if val in self.COLORS else "black"

    def get_current_lw(self):
        try: return float(self.lw_var.get())
        except Exception: return 2.0

    def on_color_change(self, event=None):
        self.line_tool.apply_color_to_selected(self.get_current_color())

    def on_lw_change(self, event=None):
        self.line_tool.apply_lw_to_selected(self.get_current_lw())

    def _build_menus(self):
        menubar = tk.Menu(self.root)

        # Draw
        draw_menu = tk.Menu(menubar, tearoff=0)
        shape_menu = tk.Menu(draw_menu, tearoff=0)
        shape_menu.add_command(label="Line", command=lambda: self.line_tool.set_mode("line"))
        shape_menu.add_command(label="Horizontal Line", command=lambda: self.line_tool.set_mode("hline"))
        draw_menu.add_cascade(label="Shape", menu=shape_menu)
        draw_menu.add_command(label="Pen", command=lambda: self.pen_tool.set_active(True))
        draw_menu.add_command(label="Pointer", command=lambda: (self.pen_tool.set_active(False), self.line_tool.set_mode("pointer")))
        draw_menu.add_separator()
        draw_menu.add_command(label="Delete Selected", command=self.delete_selected_item)
        menubar.add_cascade(label="Draw", menu=draw_menu)

        # Chart
        chart_menu = tk.Menu(menubar, tearoff=0)
        type_menu = tk.Menu(chart_menu, tearoff=0)
        for t in self.CHART_TYPES:
            type_menu.add_radiobutton(label=t, variable=self.ctype_var, value=t,
                                      command=self.apply_refresh)
        chart_menu.add_cascade(label="Type", menu=type_menu)

        barc_menu = tk.Menu(chart_menu, tearoff=0)
        for c in self.BAR_COLORS:
            barc_menu.add_radiobutton(label=c, variable=self.barcolor_var, value=c,
                                      command=self.apply_refresh)
        chart_menu.add_cascade(label="Bar Color", menu=barc_menu)
        menubar.add_cascade(label="Chart", menu=chart_menu)

        # Replay
        rmenu = tk.Menu(menubar, tearoff=0)
        speed_menu = tk.Menu(rmenu, tearoff=0)
        for s in self.SPEEDS.keys():
            speed_menu.add_radiobutton(label=s, variable=self.speed_var, value=s)
        rmenu.add_cascade(label="Speed", menu=speed_menu)
        menubar.add_cascade(label="Replay", menu=rmenu)

        self.root.config(menu=menubar)

    def delete_selected_item(self):
        self.line_tool.delete_selected(); self.canvas.draw()

    def on_delete_key(self, event):
        self.line_tool.delete_selected(); self.canvas.draw()

    def on_right_click_canvas(self, event):
        inv = self.ax.transData.inverted()
        xdata, ydata = inv.transform((event.x, event.y))
        if self.line_tool.delete_at(xdata, ydata):
            self.canvas.draw(); return
        if self.pen_tool.delete_at(xdata, ydata):
            self.canvas.draw(); return

    # ---- Redraw preserving annotations; keep future blank ----
    def _redraw_preserving(self, df_past_only, full_len):
        snap_lines = self.line_tool.snapshot()
        snap_pens  = self.pen_tool.snapshot()

        style = "candle" if self.ctype_var.get()=="Candlestick" else "bar"
        self.snap_x = draw_candles(
            self.ax, df_past_only, log_scale=self.log_var.get(), style=style,
            bar_color=self.barcolor_var.get(), x_full_len=full_len, show_upto_len=len(df_past_only)
        )
        self.line_tool.snap_x = self.snap_x

        self.line_tool.restore(snap_lines, chart_len=full_len)
        self.pen_tool.restore(snap_pens)
        self.line_tool.span_to_xlim()

        self.canvas.draw()

    # ---- Data / drawing ----
    def apply_refresh(self):
        if self.replay_state == "playing":
            messagebox.showinfo("Replay", "Stop replay before refreshing.")
            return

        ticker = self.ticker_var.get().strip()
        timeframe = self.tf_var.get().strip()
        start_str = self.start_var.get().strip()
        end_str   = self.end_var.get().strip()
        temp_str  = self.temp_var.get().strip()
        if not ticker:
            messagebox.showwarning("Input", "Please enter a ticker."); return

        df = fetch_prices(ticker, timeframe, start_str, end_str)
        if df is None: return
        self.full_df = df.copy()

        temp_dt = safe_parse_date(temp_str)
        if temp_dt is None:
            past_df = df.copy(); future_df = df.iloc[0:0].copy()
        else:
            mask = df.index <= temp_dt
            if mask.sum() == 0:
                messagebox.showwarning("Temp out of range", "Temp date is before data start (showing full).")
                past_df = df.copy(); future_df = df.iloc[0:0].copy()
            elif mask.sum() == len(df):
                messagebox.showinfo("Temp at/after end", "Temp date at/after last bar; nothing to replay.")
                past_df = df.copy(); future_df = df.iloc[0:0].copy()
            else:
                past_df = df.loc[mask].copy()
                future_df = df.loc[~mask].copy()

        self.current_df = past_df.copy()
        self.future_df  = future_df.copy()
        self.replay_index = 0
        self.replay_state = "stopped"
        self.replay_btn.config(text="Replay ▶")

        full_len = len(self.full_df)
        self._redraw_preserving(self.current_df, full_len)

    # ---- Replay (single button state machine) ----
    def on_replay_button(self):
        if self.full_df is None:
            messagebox.showwarning("Replay", "Load data first."); return
        if self.replay_state == "stopped":
            if self.future_df is None or len(self.future_df) == 0:
                messagebox.showinfo("Replay", "No future bars to replay (check Temp)."); return
            self.replay_state = "playing"
            self.replay_btn.config(text="Stop ■")
            self._replay_step()
        elif self.replay_state == "playing":
            self.replay_state = "paused"
            self.replay_btn.config(text="Resume ▶")
        elif self.replay_state == "paused":
            self.replay_state = "playing"
            self.replay_btn.config(text="Stop ■")
            self._replay_step()

    def _replay_delay_ms(self):
        return self.SPEEDS.get(self.speed_var.get(), 200)

    def _replay_step(self):
        if self.replay_state != "playing": return
        if self.replay_index >= len(self.future_df):
            self.replay_state = "stopped"; self.replay_btn.config(text="Replay ▶"); return

        next_row = self.future_df.iloc[self.replay_index:self.replay_index+1]
        self.current_df = pd.concat([self.current_df, next_row], axis=0)
        self.replay_index += 1

        self._redraw_preserving(self.current_df, full_len=len(self.full_df))
        self.root.after(self._replay_delay_ms(), self._replay_step)

def main():
    root = tk.Tk()
    app = TVStyleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
