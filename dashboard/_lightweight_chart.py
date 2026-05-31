"""TradingView-style live chart powered by TradingView's lightweight-charts.

The widget is a *renderer only*: it consumes the payload from
`GET /chart/intraday/{symbol}` (candles + overlay series + markers + per-marker
explanations) and draws it. All strategy logic stays on the backend.

Polling for live updates is driven by Streamlit (the page re-runs on its
own interval) — the widget itself is stateless across renders.
"""
from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


_LIBRARY_URL = "https://unpkg.com/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js"


# Single overlay-styling source so legend + line colours stay in sync.
_OVERLAY_STYLES: dict[str, dict[str, Any]] = {
    "ema9":       {"label": "EMA 9",   "color": "#26C6DA", "width": 1, "lineStyle": 0},
    "ema21":      {"label": "EMA 21",  "color": "#FF9800", "width": 1, "lineStyle": 0},
    "ema50":      {"label": "EMA 50",  "color": "#AB47BC", "width": 1, "lineStyle": 0},
    "vwap":       {"label": "VWAP",    "color": "#42A5F5", "width": 2, "lineStyle": 0},
    "bb_upper":   {"label": "BB Upper","color": "rgba(176,190,197,0.55)", "width": 1, "lineStyle": 2},
    "bb_middle":  {"label": "BB Mid",  "color": "rgba(176,190,197,0.35)", "width": 1, "lineStyle": 3},
    "bb_lower":   {"label": "BB Lower","color": "rgba(176,190,197,0.55)", "width": 1, "lineStyle": 2},
    "supertrend": {"label": "Supertrend", "color": "#FFCA28", "width": 2, "lineStyle": 0},
}


# Daily-chart overlay keys we can pull straight from the /strategy/chart payload's
# `indicators` block. Same styling source as the intraday chart.
_DAILY_OVERLAY_STYLES: dict[str, dict[str, Any]] = {
    "ema9":     {"label": "EMA 9",   "color": "#26C6DA", "width": 1, "lineStyle": 0},
    "ema21":    {"label": "EMA 21",  "color": "#FF9800", "width": 1, "lineStyle": 0},
    "ema50":    {"label": "EMA 50",  "color": "#AB47BC", "width": 1, "lineStyle": 0},
    "ema200":   {"label": "EMA 200", "color": "#EF5350", "width": 2, "lineStyle": 0},
    "sma50":    {"label": "SMA 50",  "color": "#FFD54F", "width": 1, "lineStyle": 0},
    "sma200":   {"label": "SMA 200", "color": "#90A4AE", "width": 1, "lineStyle": 0},
    "vwap":     {"label": "VWAP",    "color": "#42A5F5", "width": 2, "lineStyle": 0},
    "bb_upper": {"label": "BB Upper","color": "rgba(176,190,197,0.55)", "width": 1, "lineStyle": 2},
    "bb_middle":{"label": "BB Mid",  "color": "rgba(176,190,197,0.35)", "width": 1, "lineStyle": 3},
    "bb_lower": {"label": "BB Lower","color": "rgba(176,190,197,0.55)", "width": 1, "lineStyle": 2},
    "supertrend": {"label": "Supertrend", "color": "#FFCA28", "width": 2, "lineStyle": 0},
}


def render_daily_chart(
    payload: dict[str, Any],
    *,
    overlays_enabled: list[str],
    oscillators_enabled: list[str] | None = None,
    trades: list[dict[str, Any]] | None = None,
    data_source: str = "—",
    currency: str = "$",
    height: int = 720,
) -> None:
    """Render a daily candlestick chart with the SAME lightweight-charts engine
    the live tab uses, fed by the `/strategy/chart/{symbol}` payload.

    Works for US (Schwab/yfinance) and India (Upstox/yfinance .NS) alike — the
    backend provider decides the source; this is a pure renderer. `trades` is an
    optional list of {date, side, price} signal fills drawn as ▲/▼ markers.
    """
    dates = payload.get("dates", []) or []
    o = payload.get("open", []); h = payload.get("high", [])
    lo = payload.get("low", []); c = payload.get("close", [])
    vol = payload.get("volume", []) or []
    n = len(dates)

    # lightweight-charts accepts "YYYY-MM-DD" strings as time for daily series.
    candles = [
        {"time": dates[i], "open": o[i], "high": h[i], "low": lo[i], "close": c[i]}
        for i in range(n)
        if None not in (o[i] if i < len(o) else None,
                        h[i] if i < len(h) else None,
                        lo[i] if i < len(lo) else None,
                        c[i] if i < len(c) else None)
    ]
    volume = [
        {"time": dates[i],
         "value": vol[i] if i < len(vol) and vol[i] is not None else 0,
         "color": "rgba(38,166,154,0.5)" if (i < len(c) and i > 0 and c[i] is not None
                                              and c[i-1] is not None and c[i] >= c[i-1])
                  else "rgba(239,83,80,0.5)"}
        for i in range(n)
    ]

    indicators = payload.get("indicators", {}) or {}
    overlay_series: list[dict[str, Any]] = []
    for key in overlays_enabled:
        vals = indicators.get(key)
        style = _DAILY_OVERLAY_STYLES.get(key)
        if not vals or not style:
            continue
        data = [{"time": dates[i], "value": vals[i]}
                for i in range(min(n, len(vals))) if vals[i] is not None]
        if data:
            overlay_series.append({"key": key, "data": data, **style})

    # Oscillator sub-panes — each entry becomes a stacked pane below price.
    # Built from the same `indicators` block; line/histogram series per pane.
    def _line(key, color, width=1, style=0):
        vals = indicators.get(key)
        if not vals:
            return None
        data = [{"time": dates[i], "value": vals[i]}
                for i in range(min(n, len(vals))) if vals[i] is not None]
        return {"type": "line", "color": color, "width": width,
                "lineStyle": style, "data": data} if data else None

    def _hist(key, color):
        vals = indicators.get(key)
        if not vals:
            return None
        data = [{"time": dates[i], "value": vals[i],
                 "color": ("rgba(38,166,154,0.6)" if vals[i] >= 0 else "rgba(239,83,80,0.6)")}
                for i in range(min(n, len(vals))) if vals[i] is not None]
        return {"type": "histogram", "color": color, "base": 0, "data": data} if data else None

    _OSC_BUILDERS: dict[str, tuple[str, Any]] = {
        "rsi":   ("RSI 14",       lambda: [s for s in [_line("rsi14", "#AB47BC", 2)] if s]),
        "macd":  ("MACD",         lambda: [s for s in [
                        _hist("macd_hist", "#90A4AE"),
                        _line("macd", "#42A5F5", 2),
                        _line("macd_signal", "#FF9800", 1)] if s]),
        "stoch": ("Stochastic",   lambda: [s for s in [
                        _line("stoch_k", "#26C6DA", 2),
                        _line("stoch_d", "#FF9800", 1)] if s]),
        "atr":   ("ATR 14",       lambda: [s for s in [_line("atr14", "#FFCA28", 2)] if s]),
        "obv":   ("OBV",          lambda: [s for s in [_line("obv", "#66BB6A", 2)] if s]),
    }
    osc_panes: list[dict[str, Any]] = []
    for key in (oscillators_enabled or []):
        spec = _OSC_BUILDERS.get(key)
        if not spec:
            continue
        label, builder = spec
        series = builder()
        if series:
            osc_panes.append({"id": f"osc_{key}", "label": label, "series": series})

    # Adapt {date, side, price} fills into the marker shape the template draws.
    markers_all = []
    for t in (trades or []):
        d = t.get("date") or t.get("time")
        if not d:
            continue
        markers_all.append({
            "time": str(d)[:10],
            "side": str(t.get("side") or t.get("direction") or "").upper(),
            "strategy": t.get("strategy") or t.get("strategy_name") or "",
            "entry_price": t.get("price") or t.get("entry_price"),
            "reason": t.get("reason") or "",
            "_accepted": True,
        })

    config = {
        "library_url": _LIBRARY_URL,
        "candles": candles,
        "volume": volume,
        "overlays": overlay_series,
        "oscillators": osc_panes,
        "markers": markers_all,
        "show_levels": False,
        "symbol": payload.get("symbol", ""),
        "timeframe": "1D",
        "regime": "",
        "warning": "" if candles else "No candles to display.",
        "data_source": data_source,
        "fallback_anchored": 0,
        "price_precision": 2,
        "currency": currency,
    }
    cfg_json = json.dumps(config)
    components.html(_TEMPLATE.replace("__CFG__", cfg_json).replace("__HEIGHT__", str(height)),
                    height=height + 220, scrolling=False)


def render_strategy_chart(
    payload: dict[str, Any],
    *,
    overlays_enabled: list[str],
    show_trades: bool = True,
    show_rejected: bool = True,
    show_levels: bool = True,
    height: int = 720,
) -> None:
    """Render the live strategy chart.

    `payload` is the raw response from `/chart/intraday/{symbol}`.
    `overlays_enabled` is the subset of overlay keys to draw.
    """
    overlays_payload = payload.get("overlays", {}) or {}
    overlay_series: list[dict[str, Any]] = []
    for key in overlays_enabled:
        data = overlays_payload.get(key)
        if not data:
            continue
        style = _OVERLAY_STYLES.get(key)
        if not style:
            continue
        overlay_series.append({"key": key, "data": data, **style})

    markers_all = []
    if show_trades:
        for m in payload.get("markers") or []:
            markers_all.append({**m, "_accepted": True})
    if show_rejected:
        for m in payload.get("rejected_markers") or []:
            markers_all.append({**m, "_accepted": False})

    config = {
        "library_url": _LIBRARY_URL,
        "candles": payload.get("candles", []),
        "volume": payload.get("volume", []),
        "overlays": overlay_series,
        "markers": markers_all,
        "show_levels": bool(show_levels),
        "symbol": payload.get("symbol", ""),
        "timeframe": payload.get("timeframe", ""),
        "regime": payload.get("regime") or "",
        "warning": payload.get("warning") or "",
        "data_source": payload.get("data_source") or "—",
        "fallback_anchored": int(payload.get("fallback_anchored") or 0),
    }
    cfg_json = json.dumps(config)

    components.html(_TEMPLATE.replace("__CFG__", cfg_json).replace("__HEIGHT__", str(height)),
                    height=height + 220, scrolling=False)


# Self-contained HTML template — pure renderer with no embedded strategy logic.
# Everything it draws is whatever the backend sent.
_TEMPLATE = r"""
<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<style>
  html,body { margin:0; padding:0; background:#131722; color:#d1d4dc;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              font-size: 12px; height:100%; overflow:hidden; }
  #wrap { display:flex; flex-direction:column; height:100%; }
  #chart-host { width:100%; flex: 1 1 auto; min-height: 0; }
  #legend { padding: 6px 10px; background:#1a1f2c; border-bottom:1px solid #2a2e39;
            display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
  #legend .item { display:flex; align-items:center; gap:6px; }
  #legend .swatch { width:14px; height:3px; border-radius:1px; }
  #legend .meta { margin-left:auto; opacity:0.7; }
  #explain { background:#1a1f2c; border-top:1px solid #2a2e39; padding:10px 12px;
             min-height:170px; max-height:200px; overflow-y:auto; }
  #explain h4 { margin: 0 0 6px 0; font-size: 13px; color:#e6e6e6; }
  #explain .row { display:flex; gap:18px; flex-wrap:wrap; margin-bottom: 4px; }
  #explain .k { color:#9ba3b1; }
  #explain .v { color:#e6e6e6; font-variant-numeric: tabular-nums; }
  #explain .reason { margin-top: 6px; padding: 6px 8px; background:#0f1320;
                     border-left: 3px solid #42A5F5; border-radius: 2px;
                     font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: 11px; white-space: pre-wrap; }
  #explain .pill { display:inline-block; padding:1px 6px; border-radius:8px;
                   font-size:10px; margin-right:6px; }
  .pill.buy  { background:#1f3b35; color:#26a69a; border:1px solid #26a69a; }
  .pill.sell { background:#3e2326; color:#ef5350; border:1px solid #ef5350; }
  .pill.rej  { background:#3a3526; color:#ffca28; border:1px solid #ffca28; }
  #warn { color:#ffca28; padding: 4px 10px; }
</style>
</head><body>
<div id="wrap">
  <div id="legend"></div>
  <div id="warn"></div>
  <div id="chart-host"></div>
  <div id="explain">
    <h4>Click any ▲ / ▼ marker to see why the strategy did (or didn't) trade.</h4>
    <div class="row"><span class="k">Symbol</span><span id="hdr-symbol" class="v">—</span>
      <span class="k">Timeframe</span><span id="hdr-tf" class="v">—</span>
      <span class="k">Regime</span><span id="hdr-regime" class="v">—</span>
      <span class="k">Data source</span><span id="hdr-src" class="v">—</span>
      <span class="k">Fallback anchored</span><span id="hdr-fb" class="v">0</span></div>
  </div>
</div>

<script>
const CFG = __CFG__;

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = url; s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error("failed to load " + url));
    document.head.appendChild(s);
  });
}

function fmt(v, d=2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  return v.toFixed(d);
}

async function main() {
  document.getElementById("hdr-symbol").textContent = CFG.symbol || "—";
  document.getElementById("hdr-tf").textContent = CFG.timeframe || "—";
  document.getElementById("hdr-regime").textContent = CFG.regime || "—";
  document.getElementById("hdr-src").textContent = CFG.data_source || "—";
  const fbEl = document.getElementById("hdr-fb");
  fbEl.textContent = String(CFG.fallback_anchored || 0);
  if ((CFG.fallback_anchored || 0) > 0) fbEl.style.color = "#ffca28";
  if (CFG.warning) document.getElementById("warn").textContent = "⚠ " + CFG.warning;

  await loadScript(CFG.library_url);
  const host = document.getElementById("chart-host");
  const chart = LightweightCharts.createChart(host, {
    layout: { background: { type: "solid", color: "#131722" }, textColor: "#d1d4dc" },
    grid:   { vertLines: { color: "rgba(255,255,255,0.05)" }, horzLines: { color: "rgba(255,255,255,0.05)" } },
    rightPriceScale: { borderColor: "#2a2e39" },
    timeScale: { borderColor: "#2a2e39", timeVisible: true, secondsVisible: false },
    crosshair: { mode: 1 },
    autoSize: true,
  });

  const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
    upColor: "#26a69a", downColor: "#ef5350",
    borderUpColor: "#26a69a", borderDownColor: "#ef5350",
    wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });
  candleSeries.setData(CFG.candles || []);

  // Volume on its own scale at the bottom.
  const volSeries = chart.addSeries(LightweightCharts.HistogramSeries, {
    priceFormat: { type: "volume" },
    priceScaleId: "vol",
  });
  volSeries.setData(CFG.volume || []);
  chart.priceScale("vol").applyOptions({
    scaleMargins: { top: 0.82, bottom: 0 },
    visible: false,
  });

  // Oscillator sub-panes (RSI / MACD / Stochastic / ATR / OBV).
  //
  // Upgraded to lightweight-charts v5 native panes: pass `paneIndex` as the
  // 2nd arg to series factories and v5 creates a real stacked pane WITH ITS
  // OWN right-axis labels (e.g. 0/30/50/70/100 for RSI, signed values for
  // MACD). v4's custom-priceScaleId hack got us visually-separated bands but
  // overlay scales never render axis labels, which is what the user saw.
  const oscPanes = CFG.oscillators || [];
  if (oscPanes.length) {
    oscPanes.forEach((pane, pi) => {
      const paneIndex = pi + 1;  // 0 is the main price+volume pane
      (pane.series || []).forEach(ser => {
        const common = { priceLineVisible: false, lastValueVisible: true,
                         crosshairMarkerVisible: false };
        let s;
        if (ser.type === "histogram") {
          s = chart.addSeries(LightweightCharts.HistogramSeries,
                              { ...common, color: ser.color, base: ser.base || 0 }, paneIndex);
        } else {
          s = chart.addSeries(LightweightCharts.LineSeries,
                              { ...common, color: ser.color,
                                lineWidth: ser.width || 1, lineStyle: ser.lineStyle || 0 }, paneIndex);
        }
        s.setData(ser.data || []);
      });
    });
    // Give the price pane the majority of vertical space; share the rest among
    // the oscillator panes. v5's chart.panes() returns the live Pane objects.
    try {
      const panes = chart.panes();
      const totalH = host.clientHeight || 600;
      const oscShare = Math.min(0.40, 0.20 + oscPanes.length * 0.08);
      const oscH = Math.floor((totalH * oscShare) / oscPanes.length);
      const mainH = totalH - oscH * oscPanes.length;
      panes[0] && panes[0].setHeight && panes[0].setHeight(mainH);
      oscPanes.forEach((_, i) => {
        const p = panes[i + 1];
        if (p && p.setHeight) p.setHeight(oscH);
      });
    } catch (e) { /* setHeight is best-effort; ignore if v5 surface differs */ }
  }

  // Overlay line series — all backend-computed; we just draw what's sent.
  const legendEl = document.getElementById("legend");
  legendEl.innerHTML = "";
  (CFG.overlays || []).forEach(o => {
    const s = chart.addSeries(LightweightCharts.LineSeries, {
      color: o.color, lineWidth: o.width || 1, lineStyle: o.lineStyle || 0,
      priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData(o.data || []);
    const it = document.createElement("div");
    it.className = "item";
    it.innerHTML = `<div class="swatch" style="background:${o.color}"></div><span>${o.label}</span>`;
    legendEl.appendChild(it);
  });
  // meta on the right
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${CFG.candles ? CFG.candles.length : 0} bars · ${(CFG.markers||[]).length} markers`;
  legendEl.appendChild(meta);

  // Count markers per bar so we can badge ones that share a bar with others.
  const perBarCount = {};
  (CFG.markers || []).forEach(m => {
    if (!m.time) return;
    perBarCount[m.time] = (perBarCount[m.time] || 0) + 1;
  });

  // Markers — entries/rejections as TradingView-style shape markers.
  const markers = (CFG.markers || []).map((m, i) => {
    const isBuy  = m.side === "BUY";
    const isSell = m.side === "SELL" || m.side === "SELL_SHORT";
    const accepted = m._accepted;
    let color, position, shape, prefix;
    if (!accepted) {
      color = "#ffca28";
      position = isBuy ? "belowBar" : "aboveBar";
      shape = "circle";
      prefix = "✕";
    } else if (isBuy) {
      color = "#26a69a";
      position = "belowBar";
      shape = "arrowUp";
      prefix = "▲";
    } else if (isSell) {
      color = "#ef5350";
      position = "aboveBar";
      shape = "arrowDown";
      prefix = "▼";
    } else {
      color = "#9ba3b1";
      position = "inBar";
      shape = "square";
      prefix = "•";
    }
    const sameBarN = perBarCount[m.time] || 1;
    const badge = sameBarN > 1 ? ` ×${sameBarN}` : "";
    return {
      time: m.time,
      position, color, shape,
      text: `${prefix} ${m.strategy || ""}${badge}`,
      id: String(i),
      __payload: m,
    };
  }).filter(m => m.time);
  // v5: series.setMarkers was removed in favour of the createSeriesMarkers
  // primitive. The marker objects' shape is unchanged.
  LightweightCharts.createSeriesMarkers(candleSeries, markers);

  // Click-to-explain panel — cycles through multiple markers on the same bar.
  const panel = document.getElementById("explain");
  let lastClickBar = null;
  let lastClickIdx = 0;
  chart.subscribeClick(param => {
    if (!param || !param.time) return;
    const ms = markers.filter(m => m.time === param.time);
    if (!ms.length) return;
    if (param.time === lastClickBar) {
      lastClickIdx = (lastClickIdx + 1) % ms.length;
    } else {
      lastClickBar = param.time;
      lastClickIdx = 0;
    }
    renderExplain(ms[lastClickIdx].__payload, lastClickIdx + 1, ms.length);
  });

  // Tap-friendly: also surface the last marker by default so the panel isn't empty.
  if (markers.length) {
    const last = markers[markers.length - 1];
    const sameBar = markers.filter(m => m.time === last.time);
    renderExplain(last.__payload, sameBar.length, sameBar.length);
  }

  // Stop / target price lines for the most recent ACCEPTED marker, so the
  // trader can see live where the strategy wanted to risk and reward.
  if (CFG.show_levels) {
    const latest = (CFG.markers || []).filter(m => m._accepted).slice(-1)[0];
    if (latest) {
      if (latest.entry_price)
        candleSeries.createPriceLine({ price: latest.entry_price, color: "#90caf9",
          lineStyle: 0, lineWidth: 1, axisLabelVisible: true, title: "ENTRY" });
      if (latest.stop_price)
        candleSeries.createPriceLine({ price: latest.stop_price, color: "#ef5350",
          lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: "STOP" });
      if (latest.target_price)
        candleSeries.createPriceLine({ price: latest.target_price, color: "#26a69a",
          lineStyle: 2, lineWidth: 1, axisLabelVisible: true, title: "TARGET" });
    }
  }

  function fmtTs(t) {
    // Markers come from two backends: intraday (`time` is a unix epoch in
    // seconds) and daily (`time` is a "YYYY-MM-DD" string). The old version
    // multiplied unconditionally by 1000 and called toISOString(), which threw
    // "Invalid time value" on daily markers and was caught by main().catch as
    // "chart init failed: Invalid time value". Handle both shapes here.
    if (t === null || t === undefined) return "—";
    let d;
    if (typeof t === "number") {
      d = new Date(t * 1000);
    } else if (typeof t === "string") {
      d = new Date(t);
    } else {
      return String(t);
    }
    return Number.isNaN(d.getTime())
      ? String(t)
      : d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
  }
  function renderExplain(m, idx, total) {
    const pill = m._accepted
      ? (m.side === "BUY" ? `<span class="pill buy">BUY</span>` : `<span class="pill sell">${m.side||"SELL"}</span>`)
      : `<span class="pill rej">REJECTED</span>`;
    const ts = fmtTs(m.time);
    const counter = (total && total > 1)
      ? `<span style="opacity:.6;font-size:11px">  ·  marker ${idx} of ${total} on this bar (click again to cycle)</span>`
      : "";
    const fbWarn = m.anchor_fallback
      ? `<div style="margin-top:4px;color:#ffca28;font-size:11px">⚠ signal_time was unparseable — anchored to latest bar (raw: ${(m.signal_time_raw||"").replace(/&/g,"&amp;").replace(/</g,"&lt;")})</div>`
      : "";
    panel.innerHTML = `
      <h4>${pill} ${m.strategy || "—"} <span style="opacity:.6">· ${m.timeframe||""}</span>${counter}</h4>
      <div class="row">
        <span class="k">Time</span><span class="v">${ts}</span>
        <span class="k">Regime</span><span class="v">${m.regime || "—"}</span>
        <span class="k">Confidence</span><span class="v">${fmt(m.confidence, 2)}</span>
        <span class="k">R-multiple</span><span class="v">${fmt(m.r_multiple, 2)}</span>
      </div>
      <div class="row">
        <span class="k">Entry</span><span class="v">${fmt(m.entry_price)}</span>
        <span class="k">Stop</span><span class="v">${fmt(m.stop_price)}</span>
        <span class="k">Target</span><span class="v">${fmt(m.target_price)}</span>
      </div>
      ${fbWarn}
      <div class="reason">${(m.reason || m.explanation || "(no explanation supplied)")
            .replace(/&/g,"&amp;").replace(/</g,"&lt;")}</div>
    `;
  }
}

main().catch(err => {
  document.getElementById("warn").textContent = "⚠ chart init failed: " + err.message;
});
</script>
</body></html>
"""
