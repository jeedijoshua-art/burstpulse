"""
antenna_component.py
BurstPulse — Signature Antenna Signal Animation Component
Self-contained, accessible HTML5/SVG component rendered via Streamlit components.
"""

from typing import Optional
import streamlit as st
import streamlit.components.v1 as components


def get_antenna_signal_html(
    status_title: str,
    status_detail: str,
    is_animating: bool = True,
    signal_type: str = "builtin"
) -> str:
    """
    Generate a self-contained, valid HTML document with an animated antenna emitting expanding radio waves.
    Contains zero leading line indentation to prevent markdown preformatted block parsing.
    
    Args:
        status_title: Primary status message (e.g. "Signal Synchronized: Normal Network Traffic")
        status_detail: Secondary descriptive telemetry (e.g. "250 intervals · Baseline ready")
        is_animating: Whether signal waves are actively pulsing
        signal_type: "builtin" (violet/cyan), "upload" (emerald/cyan), or "alert" (amber)
    """
    accent_color = "#6366f1" if signal_type == "builtin" else ("#10b981" if signal_type == "upload" else "#f59e0b")
    beacon_color = "#38bdf8" if signal_type == "builtin" else ("#34d399" if signal_type == "upload" else "#fbbf24")
    anim_class = "active-signal" if is_animating else "idle-signal"
    status_text = "Signal Active · Real-Time Analysis" if is_animating else "Signal Synchronized"

    # Strictly column-0 aligned HTML document to prevent any markdown code-block conversion
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}
html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
@keyframes pulse-beacon {{
    0% {{ transform: scale(0.9); opacity: 0.6; filter: drop-shadow(0 0 2px {beacon_color}); }}
    50% {{ transform: scale(1.3); opacity: 1; filter: drop-shadow(0 0 8px {beacon_color}); }}
    100% {{ transform: scale(0.9); opacity: 0.6; filter: drop-shadow(0 0 2px {beacon_color}); }}
}}
@keyframes wave-expand-1 {{
    0% {{ opacity: 0; }}
    20% {{ opacity: 0.95; }}
    70% {{ opacity: 0.4; }}
    100% {{ opacity: 0.05; }}
}}
@keyframes wave-expand-2 {{
    0% {{ opacity: 0; }}
    25% {{ opacity: 0; }}
    50% {{ opacity: 0.9; }}
    80% {{ opacity: 0.35; }}
    100% {{ opacity: 0.05; }}
}}
@keyframes wave-expand-3 {{
    0% {{ opacity: 0; }}
    50% {{ opacity: 0; }}
    75% {{ opacity: 0.85; }}
    100% {{ opacity: 0.1; }}
}}
.antenna-signal-container {{
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(135deg, rgba(23, 29, 45, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid {accent_color};
    border-radius: 10px;
    padding: 8px 16px;
    height: 68px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}}
.antenna-svg-wrap {{
    flex-shrink: 0;
    width: 54px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.antenna-beacon {{
    transform-origin: 30px 10px;
    animation: pulse-beacon 1.8s infinite ease-in-out;
}}
.active-signal .wave-1 {{
    animation: wave-expand-1 2s infinite ease-out;
}}
.active-signal .wave-2 {{
    animation: wave-expand-2 2s infinite ease-out;
}}
.active-signal .wave-3 {{
    animation: wave-expand-3 2s infinite ease-out;
}}
.idle-signal .wave-1 {{ opacity: 0.7; }}
.idle-signal .wave-2 {{ opacity: 0.45; }}
.idle-signal .wave-3 {{ opacity: 0.25; }}

.antenna-text-wrap {{
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
}}
.antenna-status-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {beacon_color};
    margin-bottom: 2px;
}}
.antenna-status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: {beacon_color};
    box-shadow: 0 0 6px {beacon_color};
}}
.antenna-title {{
    font-size: 0.94rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.antenna-detail {{
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.3;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
@media (prefers-reduced-motion: reduce) {{
    .antenna-beacon,
    .active-signal .wave-1,
    .active-signal .wave-2,
    .active-signal .wave-3 {{
        animation: none !important;
    }}
}}
</style>
</head>
<body>
<div class="antenna-signal-container">
    <div class="antenna-svg-wrap {anim_class}">
        <svg viewBox="0 0 60 54" width="54" height="48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Outer Waves -->
            <path class="wave-3" d="M10 12 A22 22 0 0 0 10 38" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
            <path class="wave-3" d="M50 12 A22 22 0 0 1 50 38" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
            <!-- Middle Waves -->
            <path class="wave-2" d="M16 16 A16 16 0 0 0 16 34" stroke="{beacon_color}" stroke-width="2" stroke-linecap="round"/>
            <path class="wave-2" d="M44 16 A16 16 0 0 1 44 34" stroke="{beacon_color}" stroke-width="2" stroke-linecap="round"/>
            <!-- Inner Waves -->
            <path class="wave-1" d="M22 20 A10 10 0 0 0 22 30" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
            <path class="wave-1" d="M38 20 A10 10 0 0 1 38 30" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
            <!-- Mast / Tower Body -->
            <path d="M22 49 L38 49" stroke="#94a3b8" stroke-width="2" stroke-linecap="round"/>
            <path d="M26 49 L30 35 L34 49" stroke="#64748b" stroke-width="1.8" stroke-linejoin="round"/>
            <line x1="28" y1="42" x2="32" y2="42" stroke="#64748b" stroke-width="1.5"/>
            <line x1="30" y1="35" x2="30" y2="12" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
            <line x1="25" y1="18" x2="35" y2="18" stroke="#cbd5e1" stroke-width="1.5" stroke-linecap="round"/>
            <!-- Emitter Beacon Bulb -->
            <circle class="antenna-beacon" cx="30" cy="10" r="3.5" fill="{beacon_color}"/>
        </svg>
    </div>
    <div class="antenna-text-wrap">
        <div class="antenna-status-tag">
            <span class="antenna-status-dot"></span>
            <span>{status_text}</span>
        </div>
        <div class="antenna-title">{status_title}</div>
        <div class="antenna-detail">{status_detail}</div>
    </div>
</div>
</body>
</html>"""


def render_antenna_signal(
    status_title: str,
    status_detail: str,
    is_animating: bool = True,
    signal_type: str = "builtin",
    height: int = 76
) -> None:
    """
    Render the animated antenna signal banner into Streamlit with explicit height.
    Guarantees isolated execution, zero markdown escaping, zero raw code display, and zero layout overflow/overlap.
    """
    html_content = get_antenna_signal_html(
        status_title=status_title,
        status_detail=status_detail,
        is_animating=is_animating,
        signal_type=signal_type
    )
    if hasattr(st, "iframe"):
        st.iframe(html_content, height=height, width="stretch", alt="BurstPulse Antenna Signal Indicator")
    else:
        components.html(html_content, height=height, scrolling=False)

