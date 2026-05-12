"""
=============================================================================
dashboard/app.py  –  UAV Mission Control Dashboard (Plotly Dash)
=============================================================================
Professional aerospace-grade real-time dashboard featuring:
  - Live 3D trajectory visualization
  - Sensor fusion telemetry panels
  - Battery / GNSS / AI status indicators
  - Obstacle map overlay
  - Flight envelope monitor
  - AI decision log
  - Camera feed simulation
  - Swarm overview (if enabled)
=============================================================================
"""

from __future__ import annotations

import math
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False

from utils.logger import get_logger

log = get_logger("DASHBOARD")


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry Store (thread-safe ring buffer)
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryStore:
    """Thread-safe ring buffer for live telemetry data."""

    MAX_POINTS = 2000

    def __init__(self):
        self._lock    = threading.Lock()
        self._history = deque(maxlen=self.MAX_POINTS)
        self._latest: Dict[str, Any] = {}
        self._events: deque = deque(maxlen=100)
        self._start_time = time.time()

    def push(self, telemetry: dict) -> None:
        with self._lock:
            telemetry["wall_time"] = time.time() - self._start_time
            self._history.append(telemetry.copy())
            self._latest = telemetry.copy()

    def push_event(self, event: str, detail: str = "") -> None:
        with self._lock:
            self._events.appendleft({
                "t": round(time.time() - self._start_time, 1),
                "event": event,
                "detail": detail,
            })

    def get_latest(self) -> dict:
        with self._lock:
            return self._latest.copy()

    def get_history(self, n: Optional[int] = None) -> List[dict]:
        with self._lock:
            data = list(self._history)
        return data[-n:] if n else data

    def get_events(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._events)[:n]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest = {}
            self._events.clear()


# Global telemetry store (singleton)
_store = TelemetryStore()


def get_store() -> TelemetryStore:
    return _store


# ──────────────────────────────────────────────────────────────────────────────
# Chart Builders
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0a0e1a"
PANEL_BG  = "#0f1626"
ACCENT    = "#00d4ff"
ACCENT2   = "#ff6b35"
GREEN     = "#00ff88"
YELLOW    = "#ffd700"
RED_ALERT = "#ff3355"
GRID_CLR  = "#1a2640"
TEXT_CLR  = "#c8d8e8"
FONT      = "JetBrains Mono, Courier New, monospace"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family=FONT, color=TEXT_CLR, size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
)


def build_trajectory_3d(history: List[dict], obstacles: List[dict] = None,
                         goal: List[float] = None) -> go.Figure:
    """3D flight trajectory with obstacles."""
    fig = go.Figure()

    if history:
        xs = [d["position"][0] for d in history]
        ys = [d["position"][1] for d in history]
        zs = [d["position"][2] for d in history]
        speeds = [d.get("speed_ms", 0.0) for d in history]

        # Trajectory line with speed coloring
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=speeds, colorscale="Plasma", width=4,
                      colorbar=dict(title="Speed<br>(m/s)", x=1.02, len=0.6)),
            name="Trajectory",
            hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>",
        ))

        # Current position marker
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
            mode="markers",
            marker=dict(size=12, color=ACCENT, symbol="diamond",
                        line=dict(color="white", width=2)),
            name="UAV",
        ))

        # Start marker
        if len(xs) > 1:
            fig.add_trace(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode="markers",
                marker=dict(size=8, color=GREEN, symbol="circle"),
                name="Start",
            ))

    # Goal marker
    if goal:
        fig.add_trace(go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]],
            mode="markers+text",
            marker=dict(size=14, color=YELLOW, symbol="cross",
                        line=dict(color="white", width=2)),
            text=["GOAL"],
            textfont=dict(color=YELLOW, size=12),
            name="Goal",
        ))

    # Obstacles
    if obstacles:
        for obs in obstacles[:30]:  # Limit for performance
            pos = obs["position"]
            r   = obs.get("radius", 2.0)
            clr = obs.get("color", "#555555")
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(size=max(3, r * 0.5), color=clr, opacity=0.7),
                showlegend=False,
                hovertemplate=f"Obstacle<br>r={r:.1f}m<extra></extra>",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        scene=dict(
            xaxis=dict(title="East (m)", backgroundcolor=DARK_BG,
                       gridcolor=GRID_CLR, showbackground=True),
            yaxis=dict(title="North (m)", backgroundcolor=DARK_BG,
                       gridcolor=GRID_CLR, showbackground=True),
            zaxis=dict(title="Altitude (m)", backgroundcolor=DARK_BG,
                       gridcolor=GRID_CLR, showbackground=True),
            bgcolor=DARK_BG,
        ),
        title=dict(text="3D Flight Trajectory", font=dict(color=ACCENT, size=14)),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_CLR),
        height=480,
    )
    return fig


def build_telemetry_chart(history: List[dict]) -> go.Figure:
    """Multi-panel telemetry time series."""
    if not history:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          title="Telemetry — waiting for data...")
        return fig

    ts       = [d.get("wall_time", i*0.02) for i, d in enumerate(history)]
    alts     = [d["position"][2] for d in history]
    speeds   = [d.get("speed_ms", 0) for d in history]
    batts    = [d.get("battery_pct", 100) for d in history]
    gnss_q   = [d.get("gnss_quality", 1) * 100 for d in history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Altitude (m)", "Speed (m/s)", "Battery (%)", "GNSS Quality (%)"),
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    colors = [GREEN, ACCENT, YELLOW, ACCENT2]
    data_sets = [alts, speeds, batts, gnss_q]
    names = ["Altitude", "Speed", "Battery", "GNSS"]

    for i, (vals, name, clr) in enumerate(zip(data_sets, names, colors)):
        row = i // 2 + 1; col = i % 2 + 1
        fig.add_trace(
            go.Scatter(x=ts, y=vals, mode="lines", line=dict(color=clr, width=2),
                       fill="tozeroy", fillcolor=clr.replace(")", ",0.1)").replace("rgb", "rgba")
                       if "rgb" in clr else clr + "22",
                       name=name, showlegend=False),
            row=row, col=col,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        title=dict(text="Real-Time Telemetry", font=dict(color=ACCENT, size=13)),
    )
    return fig


def build_lidar_polar(lidar_data: Optional[dict]) -> go.Figure:
    """360° LiDAR polar plot."""
    if lidar_data is None or "ranges" not in lidar_data:
        angles = np.linspace(0, 360, 360)
        ranges = np.full(360, 100.0)
    else:
        ranges = np.array(lidar_data["ranges"])
        angles = np.degrees(np.linspace(0, 2*math.pi, len(ranges)))

    # Color by distance
    norm_r = ranges / max(ranges.max(), 1)
    colors = [f"hsl({int(120*r)},100%,50%)" for r in norm_r]

    fig = go.Figure(go.Scatterpolar(
        r=ranges,
        theta=angles,
        mode="lines",
        line=dict(color=ACCENT, width=1.5),
        fill="toself",
        fillcolor=f"rgba(0, 212, 255, 0.1)",
        name="LiDAR",
    ))
    # Danger zone ring (2.5m safety margin)
    fig.add_trace(go.Scatterpolar(
        r=[2.5] * 361,
        theta=list(range(361)),
        mode="lines",
        line=dict(color=RED_ALERT, width=1, dash="dot"),
        name="Safety margin",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=DARK_BG,
        polar=dict(
            bgcolor=DARK_BG,
            angularaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                             tickcolor=TEXT_CLR, tickfont=dict(color=TEXT_CLR, size=9)),
            radialaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                            tickcolor=TEXT_CLR, tickfont=dict(color=TEXT_CLR, size=9),
                            range=[0, min(100, ranges.max() * 1.1)]),
        ),
        font=dict(family=FONT, color=TEXT_CLR),
        margin=dict(l=30, r=30, t=30, b=30),
        title=dict(text="LiDAR Obstacle Map", font=dict(color=ACCENT, size=12)),
        height=300,
        legend=dict(bgcolor=PANEL_BG),
    )
    return fig


def build_attitude_indicator(roll: float, pitch: float, yaw: float) -> go.Figure:
    """Artificial horizon / attitude indicator."""
    fig = go.Figure()

    # Background
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1,
                  fillcolor=PANEL_BG, line_color=GRID_CLR)

    # Sky
    horizon_y = math.sin(pitch)
    sky_pts = [[-1, horizon_y - math.tan(roll)],
               [ 1, horizon_y + math.tan(roll)],
               [ 1, 1.2], [-1, 1.2]]
    xs = [p[0] for p in sky_pts]; ys = [p[1] for p in sky_pts]
    fig.add_trace(go.Scatter(x=xs+[xs[0]], y=ys+[ys[0]],
                              fill="toself", fillcolor="#1a3a6b",
                              line=dict(color="#1a3a6b"), showlegend=False))

    # Ground
    gnd_pts = [[-1, horizon_y - math.tan(roll)],
               [ 1, horizon_y + math.tan(roll)],
               [ 1, -1.2], [-1, -1.2]]
    xs2 = [p[0] for p in gnd_pts]; ys2 = [p[1] for p in gnd_pts]
    fig.add_trace(go.Scatter(x=xs2+[xs2[0]], y=ys2+[ys2[0]],
                              fill="toself", fillcolor="#3d2b0f",
                              line=dict(color="#3d2b0f"), showlegend=False))

    # Horizon line
    fig.add_shape(type="line",
                  x0=-1, y0=horizon_y - math.tan(roll),
                  x1= 1, y1=horizon_y + math.tan(roll),
                  line=dict(color=YELLOW, width=2))

    # Aircraft symbol
    fig.add_trace(go.Scatter(
        x=[-0.3, 0, 0.3], y=[0, 0, 0],
        mode="lines", line=dict(color=YELLOW, width=3), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-0.15, 0.15],
        mode="lines", line=dict(color=YELLOW, width=3), showlegend=False
    ))

    # Annotations
    fig.add_annotation(x=0, y=-0.85, text=f"R:{math.degrees(roll):.1f}°",
                        showarrow=False, font=dict(color=TEXT_CLR, size=10, family=FONT))
    fig.add_annotation(x=0, y=-0.95, text=f"P:{math.degrees(pitch):.1f}°",
                        showarrow=False, font=dict(color=TEXT_CLR, size=10, family=FONT))

    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        margin=dict(l=5, r=5, t=25, b=5),
        height=220,
        title=dict(text="Attitude Indicator", font=dict(color=ACCENT, size=12)),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Layout Builders
# ──────────────────────────────────────────────────────────────────────────────

def _kpi_card(title: str, value: str, unit: str = "", color: str = ACCENT,
               icon: str = "") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(f"{icon} {title}", className="kpi-label",
                     style={"fontSize": "10px", "color": TEXT_CLR,
                            "fontFamily": FONT, "letterSpacing": "1px",
                            "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Div([
                html.Span(value, style={"fontSize": "22px", "fontWeight": "700",
                                        "color": color, "fontFamily": FONT}),
                html.Span(f" {unit}", style={"fontSize": "12px", "color": TEXT_CLR,
                                              "fontFamily": FONT}),
            ]),
        ], style={"padding": "10px 14px"}),
        style={
            "background": PANEL_BG,
            "border": f"1px solid {color}44",
            "borderLeft": f"3px solid {color}",
            "borderRadius": "4px",
        },
    )


def _status_badge(label: str, active: bool) -> html.Div:
    color = GREEN if active else RED_ALERT
    return html.Div(
        [html.Span("●", style={"color": color, "marginRight": "4px"}), label],
        style={"fontSize": "10px", "fontFamily": FONT, "color": TEXT_CLR,
               "marginBottom": "3px"},
    )


def build_layout() -> html.Div:
    """Full dashboard layout."""
    header = html.Div([
        html.Div([
            html.Img(src="", style={"display": "none"}),
            html.Div([
                html.H1("UAV MISSION CONTROL",
                        style={"color": ACCENT, "fontFamily": FONT, "fontSize": "18px",
                               "fontWeight": "700", "letterSpacing": "4px",
                               "margin": "0", "textShadow": f"0 0 20px {ACCENT}66"}),
                html.P("AUTONOMOUS NAVIGATION SYSTEM v1.0 | REAL-TIME TELEMETRY",
                       style={"color": TEXT_CLR, "fontFamily": FONT, "fontSize": "9px",
                              "letterSpacing": "2px", "margin": "0"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),

        # Connection / mission status
        html.Div([
            html.Div(id="mission-status", children=[
                _status_badge("SENSORS", True),
                _status_badge("GNSS RTK", True),
                _status_badge("AI ACTIVE", True),
                _status_badge("COMMS", True),
            ], style={"display": "flex", "gap": "16px"}),
            html.Div(id="wall-clock",
                     style={"fontFamily": FONT, "color": ACCENT2, "fontSize": "12px"}),
        ], style={"display": "flex", "gap": "20px", "alignItems": "center"}),
    ], style={
        "background": f"linear-gradient(90deg, {DARK_BG} 0%, {PANEL_BG} 100%)",
        "borderBottom": f"2px solid {ACCENT}44",
        "padding": "12px 20px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
    })

    kpi_row = dbc.Row([
        dbc.Col(_kpi_card("ALTITUDE", "—", "m", GREEN,      "↑"), width=2),
        dbc.Col(_kpi_card("SPEED",    "—", "m/s", ACCENT,   "→"), width=2),
        dbc.Col(_kpi_card("BATTERY",  "—", "%",   YELLOW,   "⚡"), width=2),
        dbc.Col(_kpi_card("DIST GOAL","—", "m",   ACCENT2,  "◎"), width=2),
        dbc.Col(_kpi_card("GNSS",     "—", "%",   GREEN,    "📡"), width=2),
        dbc.Col(_kpi_card("STEP",     "—", "",    TEXT_CLR, "#"), width=2),
    ], id="kpi-row", className="g-2", style={"margin": "10px 10px 0"})

    main_row = dbc.Row([
        # Left: 3D trajectory + telemetry
        dbc.Col([
            dbc.Card([
                dcc.Graph(id="traj-3d", config={"displayModeBar": False},
                          style={"height": "480px"}),
            ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                      "borderRadius": "4px", "marginBottom": "8px"}),

            dbc.Card([
                dcc.Graph(id="telemetry-chart", config={"displayModeBar": False},
                          style={"height": "320px"}),
            ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                      "borderRadius": "4px"}),
        ], width=7),

        # Right: LiDAR + Attitude + Status panels
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dcc.Graph(id="lidar-polar", config={"displayModeBar": False},
                                  style={"height": "300px"}),
                    ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                              "borderRadius": "4px"}),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dcc.Graph(id="attitude-indicator", config={"displayModeBar": False},
                                  style={"height": "300px"}),
                    ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                              "borderRadius": "4px"}),
                ], width=6),
            ], className="g-2", style={"marginBottom": "8px"}),

            # Sensor status panel
            dbc.Card([
                html.Div([
                    html.Div("SENSOR MATRIX", style={
                        "fontFamily": FONT, "color": ACCENT, "fontSize": "10px",
                        "letterSpacing": "2px", "marginBottom": "8px",
                    }),
                    html.Div(id="sensor-matrix"),
                ], style={"padding": "10px 14px"}),
            ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                      "borderRadius": "4px", "marginBottom": "8px"}),

            # AI Decision log
            dbc.Card([
                html.Div([
                    html.Div("AI DECISION LOG", style={
                        "fontFamily": FONT, "color": ACCENT2, "fontSize": "10px",
                        "letterSpacing": "2px", "marginBottom": "8px",
                    }),
                    html.Div(id="event-log",
                             style={"maxHeight": "200px", "overflowY": "auto"}),
                ], style={"padding": "10px 14px"}),
            ], style={"background": PANEL_BG, "border": f"1px solid {GRID_CLR}",
                      "borderRadius": "4px"}),
        ], width=5),
    ], className="g-2", style={"margin": "8px 10px"})

    return html.Div([
        header,
        html.Div(id="kpi-container", children=[kpi_row]),
        main_row,
        dcc.Interval(id="update-interval", interval=200, n_intervals=0),
        dcc.Store(id="sim-store"),
    ], style={"background": DARK_BG, "minHeight": "100vh", "padding": "0"})


# ──────────────────────────────────────────────────────────────────────────────
# Dash App Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_dashboard_app(config: dict) -> Optional[Any]:
    if not _HAS_DASH:
        log.warning("Dash not installed. Dashboard unavailable.")
        return None

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
        title="UAV Mission Control",
        update_title=None,
    )
    app.layout = build_layout()

    store = get_store()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        [
            Output("traj-3d",           "figure"),
            Output("telemetry-chart",   "figure"),
            Output("lidar-polar",       "figure"),
            Output("attitude-indicator","figure"),
            Output("kpi-container",     "children"),
            Output("sensor-matrix",     "children"),
            Output("event-log",         "children"),
            Output("wall-clock",        "children"),
        ],
        Input("update-interval", "n_intervals"),
    )
    def update_all(n):
        latest  = store.get_latest()
        history = store.get_history(n=500)
        events  = store.get_events(15)

        # ── KPI row ───────────────────────────────────────────────────
        alt   = f"{latest.get('altitude_m',    0.0):.1f}"
        spd   = f"{latest.get('speed_ms',      0.0):.1f}"
        bat   = f"{latest.get('battery_pct', 100.0):.0f}"
        dist  = f"{latest.get('distance_to_goal', 0.0):.1f}"
        gnss  = f"{latest.get('gnss_quality',  1.0)*100:.0f}"
        step  = str(latest.get('step', 0))

        bat_val   = float(bat)
        bat_color = GREEN if bat_val > 50 else YELLOW if bat_val > 20 else RED_ALERT
        gnss_val  = float(gnss)
        gnss_color= GREEN if gnss_val > 66 else YELLOW if gnss_val > 33 else RED_ALERT

        kpi_row = dbc.Row([
            dbc.Col(_kpi_card("ALTITUDE",  alt,  "m",   GREEN,      "↑"), width=2),
            dbc.Col(_kpi_card("SPEED",     spd,  "m/s", ACCENT,     "→"), width=2),
            dbc.Col(_kpi_card("BATTERY",   bat,  "%",   bat_color,  "⚡"), width=2),
            dbc.Col(_kpi_card("DIST GOAL", dist, "m",   ACCENT2,    "◎"), width=2),
            dbc.Col(_kpi_card("GNSS",      gnss, "%",   gnss_color, "📡"), width=2),
            dbc.Col(_kpi_card("STEP",      step, "",    TEXT_CLR,   "#"), width=2),
        ], className="g-2", style={"margin": "10px 10px 0"})

        # ── Figures ───────────────────────────────────────────────────
        obstacles = latest.get("obstacles", [])
        goal      = latest.get("goal")
        traj_fig  = build_trajectory_3d(history, obstacles, goal)

        tel_fig   = build_telemetry_chart(history)

        lidar_data = latest.get("lidar")
        lidar_fig  = build_lidar_polar(lidar_data)

        att        = latest.get("attitude", [0, 0, 0])
        att_fig    = build_attitude_indicator(att[0], att[1], att[2])

        # ── Sensor matrix ─────────────────────────────────────────────
        sensor_names = ["IMU", "GNSS", "BARO", "MAG", "LiDAR", "Camera"]
        sensor_active = [True, gnss_val > 0, True, True,
                         lidar_data is not None, True]
        sensor_matrix = html.Div(
            [_status_badge(n, a) for n, a in zip(sensor_names, sensor_active)],
            style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                   "gap": "4px"},
        )

        # ── Event log ────────────────────────────────────────────────
        log_items = []
        for ev in events:
            color = ACCENT if "WAYPOINT" in ev["event"].upper() else \
                    RED_ALERT if "COLLISION" in ev["event"].upper() else \
                    YELLOW if "WARNING" in ev["event"].upper() else TEXT_CLR
            log_items.append(html.Div(
                f"[{ev['t']:.1f}s] {ev['event']}: {ev['detail']}",
                style={"fontFamily": FONT, "fontSize": "10px",
                       "color": color, "marginBottom": "2px",
                       "borderBottom": f"1px solid {GRID_CLR}",
                       "paddingBottom": "2px"},
            ))
        if not log_items:
            log_items = [html.Div("Awaiting events...",
                                   style={"fontFamily": FONT, "fontSize": "10px",
                                          "color": GRID_CLR})]

        wall_clock = f"T+ {latest.get('t', 0.0):.1f}s"

        return (traj_fig, tel_fig, lidar_fig, att_fig,
                [kpi_row], sensor_matrix, log_items, wall_clock)

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Run
# ──────────────────────────────────────────────────────────────────────────────

def run_dashboard(config: dict, host: str = "0.0.0.0", port: int = 8050,
                  debug: bool = False) -> None:
    app = create_dashboard_app(config)
    if app is None:
        log.error("Cannot start dashboard: Dash not installed.")
        return
    log.info(f"Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
