"""
Unicorn UI/UX Design System — Precision, Modern, Clean, Deliberate.
Linear/Stripe inspired. High contrast, refined typography, structured spacing.
Supports Light Mode (back_nor.png) and Dark Mode (back2.png).
"""

import base64
from pathlib import Path

_BG1_BASE64_CACHE = None
_BG2_BASE64_CACHE = None


def _get_bg1_base64() -> str:
    global _BG1_BASE64_CACHE
    if _BG1_BASE64_CACHE is None:
        bg_path = Path("back_nor.png")
        if not bg_path.exists():
            bg_path = Path(__file__).resolve().parent.parent / "back_nor.png"
        if bg_path.exists():
            try:
                with open(bg_path, "rb") as f:
                    _BG1_BASE64_CACHE = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                _BG1_BASE64_CACHE = ""
        else:
            _BG1_BASE64_CACHE = ""
    return _BG1_BASE64_CACHE


def _get_bg2_base64() -> str:
    global _BG2_BASE64_CACHE
    if _BG2_BASE64_CACHE is None:
        bg_path = Path("back2.png")
        if not bg_path.exists():
            bg_path = Path(__file__).resolve().parent.parent / "back2.png"
        if bg_path.exists():
            try:
                with open(bg_path, "rb") as f:
                    _BG2_BASE64_CACHE = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                _BG2_BASE64_CACHE = ""
        else:
            _BG2_BASE64_CACHE = ""
    return _BG2_BASE64_CACHE


def get_custom_css(dark_mode: bool = False, is_training: bool = False) -> str:
    bg_b64 = _get_bg2_base64() if dark_mode else _get_bg1_base64()
    if bg_b64:
        bg_rule = f'background-image: url("data:image/png;base64,{bg_b64}") !important;'
    else:
        bg_rule = 'background-color: #0F172A !important;' if dark_mode else 'background-color: #EEF2F6 !important;'

    # Base CSS template
    css_template = """
    <style>
    /* ---- Global Canvas (with back_nor.png or back2.png background image) ---- */
    html, body {
        background: transparent !important;
        background-color: transparent !important;
    }

    .stApp {
        background: transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        position: relative !important;
        isolation: isolate !important;
        background: transparent !important;
        background-color: transparent !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    /* Ambient slightly blurred background wallpaper layer */
    [data-testid="stAppViewContainer"]::before {
        content: "" !important;
        position: fixed !important;
        top: -15px !important;
        left: -15px !important;
        width: calc(100vw + 30px) !important;
        height: calc(100vh + 30px) !important;
        __BG_RULE__
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        filter: blur(6px) !important;
        -webkit-filter: blur(6px) !important;
        transform: scale(1.03) !important;
        z-index: -1 !important;
        pointer-events: none !important;
    }

    [data-testid="stHeader"],
    .main,
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottom"],
    .stApp > header {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Pull content right up and eliminate large empty gap at top of page */
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    div[class*="block-container"] {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
    }

    [data-testid="stHeader"] {
        height: 1rem !important;
        min-height: 0 !important;
        background: transparent !important;
    }

    /* ---- Header Hero Navbar ---- */
    .lab-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        padding: 18px 26px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .lab-hero-title {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }
    .lab-hero-sub {
        font-size: 13px;
        color: #94A3B8 !important;
        margin: 3px 0 0 0;
    }
    .lab-hero-sub b {
        color: #E2E8F0 !important;
    }
    .pill {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .pill-blue   { background: rgba(37,99,235,.2); color: #93C5FD !important; border: 1px solid rgba(147,197,253,.3); }
    .pill-indigo { background: rgba(99,102,241,.2); color: #A5B4FC !important; border: 1px solid rgba(165,180,252,.3); }
    .pill-amber  { background: rgba(217,119,6,.2);  color: #FCD34D !important; border: 1px solid rgba(252,211,77,.3); }

    /* ---- Stepper ---- */
    .stepper-wrap {
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 16px !important;
        padding: 14px 28px !important;
        margin-bottom: 20px !important;
        box-shadow: none !important;
    }
    .step-node {
        display: flex;
        align-items: center;
        flex: 1;
        gap: 16px;
    }
    .step-num {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 16px;
        flex-shrink: 0;
    }
    .step-node.done .step-num   { background: #10B981; color: #FFFFFF !important; }
    .step-node.active .step-num { background: #2563EB; color: #FFFFFF !important; }
    .step-node.idle .step-num   { background: rgba(226, 232, 240, 0.7); color: #64748B !important; }
    .step-label {
        font-size: 17px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .step-node.active .step-label { color: #2563EB; font-weight: 800; }
    .step-node.done .step-label   { color: #0F172A; }
    .step-node.idle .step-label   { color: #64748B; }
    .step-sub {
        font-size: 13px;
        color: #64748B;
        display: block;
        margin-top: 3px;
    }
    .step-arrow {
        color: #94A3B8;
        font-size: 26px;
        margin: 0 24px;
        flex-shrink: 0;
        font-weight: 300;
    }

    /* ---- Checkboxes & Labels ---- */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] label span {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 17px !important;
    }

    /* ---- Checkbox inline tags/badges beside names ---- */
    [data-testid="stCheckbox"] label code {
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 6px !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        padding: 3px 9px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        margin-left: 10px !important;
        letter-spacing: 0.02em !important;
        display: inline-block !important;
        vertical-align: middle !important;
    }

    /* ---- Playing Card Container Cards ---- */
    div[class*="st-key-card_"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 
            0 1px 3px rgba(15, 23, 42, 0.05),
            0 4px 8px -1px rgba(15, 23, 42, 0.08),
            0 12px 24px -4px rgba(15, 23, 42, 0.12),
            0 20px 32px -6px rgba(15, 23, 42, 0.06) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[class*="st-key-card_"]:hover {
        border-color: #64748B !important;
        box-shadow: 
            0 2px 5px rgba(15, 23, 42, 0.06),
            0 8px 16px -2px rgba(15, 23, 42, 0.12),
            0 20px 36px -4px rgba(15, 23, 42, 0.16),
            0 32px 56px -8px rgba(15, 23, 42, 0.10) !important;
        transform: translateY(-3px) !important;
    }
    div[class*="st-key-card_"]:has(input:checked),
    div[class*="st-key-card_"]:has([aria-checked="true"]) {
        border: 2px solid #10B981 !important;
        border-color: #10B981 !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 4px 12px -2px rgba(15, 23, 42, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    div[class*="st-key-card_"] [data-testid="stCheckbox"] label,
    div[class*="st-key-card_"] [data-testid="stCheckbox"] label div {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* ---- Sample Dataset Cards (Load button inside bottom-right with green outline) ---- */
    div[class*="st-key-sample_card_"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 
            0 1px 3px rgba(15, 23, 42, 0.05),
            0 4px 8px -1px rgba(15, 23, 42, 0.08),
            0 12px 24px -4px rgba(15, 23, 42, 0.12),
            0 20px 32px -6px rgba(15, 23, 42, 0.06) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[class*="st-key-sample_card_"]:hover {
        border-color: #64748B !important;
        box-shadow: 
            0 2px 5px rgba(15, 23, 42, 0.06),
            0 8px 16px -2px rgba(15, 23, 42, 0.12),
            0 20px 36px -4px rgba(15, 23, 42, 0.16),
            0 32px 56px -8px rgba(15, 23, 42, 0.10) !important;
        transform: translateY(-2px) !important;
    }
    div[class*="st-key-sample_card_"][class*="_active"] {
        border: 2px solid #10B981 !important;
        border-color: #10B981 !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 4px 12px -2px rgba(15, 23, 42, 0.08) !important;
    }
    /* Green outline Load Dataset Button inside sample card */
    div[class*="st-key-sample_card_"] button {
        border: 2px solid #10B981 !important;
        color: #0F172A !important;
        background: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(16, 185, 129, 0.1) !important;
    }
    div[class*="st-key-sample_card_"] button p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[class*="st-key-sample_card_"] button:hover {
        border-color: #059669 !important;
        background: #F0FDF4 !important;
        color: #0F172A !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    div[class*="st-key-sample_card_"] button:active {
        transform: translateY(0) !important;
    }

    /* ---- Global Pipeline Parameters Container (Less White Background) ---- */
    div[class*="st-key-pipeline_params_card"] {
        background: #F1F5F9 !important;
        background-color: #F1F5F9 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-top: 10px !important;
        margin-bottom: 26px !important;
        box-shadow: 
            0 1px 3px rgba(15, 23, 42, 0.05),
            0 4px 8px -1px rgba(15, 23, 42, 0.08),
            0 12px 24px -4px rgba(15, 23, 42, 0.06) !important;
    }
    div[class*="st-key-pipeline_params_card"]:empty,
    div[class*="st-key-pipeline_params_card"]:not(:has(div[data-testid="stElementContainer"])) {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        border: none !important;
        margin: 0 !important;
    }
    div[class*="st-key-pipeline_params_card"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }

    /* ---- Results KPI Summary Card (Executive Scientific Midnight Texture) ---- */
    div[class*="st-key-results_kpi_card"] {
        background: radial-gradient(circle at 85% 15%, rgba(59, 130, 246, 0.22) 0%, transparent 45%),
                    radial-gradient(circle at 15% 85%, rgba(16, 185, 129, 0.15) 0%, transparent 40%),
                    linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0B192E 100%) !important;
        background-color: #0F172A !important;
        border: 1.5px solid rgba(147, 197, 253, 0.25) !important;
        border-radius: 16px !important;
        padding: 24px 28px !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
        box-shadow: 
            0 4px 6px -1px rgba(15, 23, 42, 0.2),
            0 12px 28px -4px rgba(15, 23, 42, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }

    div[class*="st-key-results_kpi_card"] .results-kpi-header {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
        margin: 0 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---- KPI Strip Inside Results Card ---- */
    .kpi-strip {
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        width: 100%;
        padding: 4px 0;
    }
    .kpi-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 0 20px;
    }
    .kpi-col:first-child {
        padding-left: 0;
    }
    .kpi-col:last-child {
        padding-right: 0;
    }
    .kpi-sep {
        width: 1px;
        background: rgba(255, 255, 255, 0.12);
        align-self: stretch;
        flex-shrink: 0;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .kpi-sub {
        display: flex;
        align-items: center;
        min-height: 24px;
    }
    .kpi-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
        line-height: 1;
    }
    .kpi-badge-green {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .kpi-badge-blue {
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(147, 197, 253, 0.35);
    }
    .kpi-badge-slate {
        background: rgba(148, 163, 184, 0.2);
        color: #E2E8F0;
        border: 1px solid rgba(226, 232, 240, 0.25);
    }

    /* Reset internal blocks inside cards */
    div[class*="st-key-card_"] div,
    div[class*="st-key-card_"] [data-testid="stVerticalBlock"],
    div[class*="st-key-card_"] [data-testid="stHorizontalBlock"],
    div[class*="st-key-card_"] [data-testid="stElementContainer"],
    div[class*="st-key-pipeline_params_card"] [data-testid="stVerticalBlock"],
    div[class*="st-key-pipeline_params_card"] [data-testid="stHorizontalBlock"],
    div[class*="st-key-pipeline_params_card"] [data-testid="stElementContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
    }

    /* Preserve popover button inside card */
    div[class*="st-key-card_"] [data-testid="stPopover"] button,
    div[class*="st-key-card_"] [data-testid="stPopoverButton"] button {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }

    /* ---- Popover / ⚙️ Gear Button ---- */
    [data-testid="stPopover"] button,
    [data-testid="stPopoverButton"] button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopoverButton"] > button {
        background: #F8FAFC !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        min-width: 36px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stPopover"] button:hover,
    [data-testid="stPopoverButton"] button:hover {
        background: #EFF6FF !important;
        border-color: #2563EB !important;
        color: #2563EB !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.15) !important;
    }

    /* ---- Primary & Secondary Buttons ---- */
    .stButton > button[kind="primary"] {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button[kind="secondary"] {
        border: 1px solid #CBD5E1 !important;
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        background: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background: #EFF6FF !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.12) !important;
        transform: translateY(-1px) !important;
    }

    /* ---- Load Dataset Buttons: EXPLICIT GREEN BORDER, DARK TEXT INITIALLY ---- */
    div[class*="st-key-btn2"] button,
    div[class*="st-key-btn3"] button,
    div[class*="st-key-sample_card_"] button,
    div[class*="st-key-sample_card_"] .stButton > button,
    .green-load-btn button {
        border: 2px solid #10B981 !important;
        border-color: #10B981 !important;
        color: #0F172A !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        box-shadow: 0 1px 3px rgba(16, 185, 129, 0.15) !important;
        transition: all 0.2s ease !important;
    }
    div[class*="st-key-btn2"] button:hover,
    div[class*="st-key-btn3"] button:hover,
    div[class*="st-key-sample_card_"] button:hover,
    div[class*="st-key-sample_card_"] .stButton > button:hover,
    .green-load-btn button:hover {
        border: 2px solid #059669 !important;
        border-color: #059669 !important;
        background: #F0FDF4 !important;
        background-color: #F0FDF4 !important;
        color: #0F172A !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    div[class*="st-key-btn2"] button p,
    div[class*="st-key-btn3"] button p,
    div[class*="st-key-sample_card_"] button p,
    .green-load-btn button p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* ---- Loaded Dataset Button: GREEN ONLY ---- */
    .loaded-btn-wrap button,
    .loaded-btn-wrap button:disabled,
    .loaded-btn-wrap [data-testid="stBaseButton-secondary"]:disabled,
    .loaded-btn-wrap [data-testid="stBaseButton-primary"]:disabled,
    .stElementContainer.st-key-btn2_loaded button,
    .stElementContainer.st-key-btn3_loaded button,
    div[class*="st-key-btn2_loaded"] button,
    div[class*="st-key-btn3_loaded"] button,
    div[class*="btn2_loaded"] button,
    div[class*="btn3_loaded"] button {
        background: #10B981 !important;
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border: 1px solid #059669 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        opacity: 1 !important;
        cursor: default !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4) !important;
    }
    .loaded-btn-wrap button:hover,
    div[class*="btn2_loaded"] button:hover,
    div[class*="btn3_loaded"] button:hover {
        background: #059669 !important;
        background-color: #059669 !important;
        color: #FFFFFF !important;
    }
    .loaded-btn-wrap button *,
    .loaded-btn-wrap button p,
    div[class*="btn2_loaded"] button *,
    div[class*="btn3_loaded"] button * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* ---- Dark Preview Table Scrollbar ---- */
    .dark-preview-table-wrap::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    .dark-preview-table-wrap::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    .dark-preview-table-wrap::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 3px;
    }
    .dark-preview-table-wrap::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* ---- Cursor & Results Entrance Animations ---- */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    .term-cursor {
        animation: blink 0.9s infinite;
        display: inline-block;
        color: #38BDF8;
        font-weight: 700;
    }
    @keyframes resultsSlideIn {
        0% {
            opacity: 0;
            transform: translateY(24px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .results-entrance,
    div[class*="st-key-results_wrapper"] {
        animation: resultsSlideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }

    /* ---- Metric Tiles ---- */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 
            0 1px 3px rgba(15, 23, 42, 0.05),
            0 4px 8px -1px rgba(15, 23, 42, 0.08),
            0 12px 20px -4px rgba(15, 23, 42, 0.09) !important;
    }

    /* ---- Plot 1 (Confusion Matrix) Left Alignment ---- */
    div[class*="st-key-chart_cm_linked"],
    div[class*="st-key-chart_cm_dedicated"] {
        margin-left: -32px !important;
    }

    /* ---- Plot 2 Diagnostic View Radio (Cleanly Underneath Plot 2) ---- */
    div[class*="st-key-linked_p2_mode"] {
        margin-top: 10px !important;
        margin-bottom: 6px !important;
        display: flex !important;
        justify-content: flex-start !important;
    }
    div[class*="st-key-linked_p2_mode"] > div[data-testid="stRadio"] {
        width: 100% !important;
    }
    div[class*="st-key-linked_p2_mode"] div[role="radiogroup"] {
        justify-content: flex-start !important;
        gap: 18px !important;
    }

    /* ---- Scientific Software Artifact Manager Workspace ---- */
    div[class*="st-key-export_workspace"] > div[data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-export_workspace"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 20px 24px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }

    /* Artifact Manager Workspace Container (Pop-off Solid Surface) */
    div[class*="st-key-artifact_workspace"] {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 24px 28px !important;
        margin: 0 auto 24px auto !important;
        max-width: 1440px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 6px 16px -2px rgba(15, 23, 42, 0.08) !important;
    }

    /* Artifact Row Cards (Proper Individual Divs) */
    div[class*="st-key-art_card_"] {
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[class*="st-key-art_card_"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    }

    /* Segmented Artifact Category Tabs */
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 1.5px solid #E2E8F0 !important;
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 20px !important;
        height: 42px !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
        font-size: 14.5px !important;
        font-weight: 500 !important;
        color: #536176 !important;
        padding: 10px 20px !important;
        border-radius: 6px 6px 0 0 !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
        border-bottom: 2px solid transparent !important;
        height: 42px !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab"]:hover {
        color: #17243D !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563EB !important;
        background: transparent !important;
    }

    /* Artifact Manager Download Buttons (Subtle Outlined Neutral) */
    div[class*="st-key-btn_art_"] button {
        background: #FFFFFF !important;
        color: #17243D !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 0 14px !important;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[class*="st-key-btn_art_"] button:hover {
        background: #EFF6FF !important;
        border-color: #2563EB !important;
        color: #2563EB !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.18) !important;
    }
    div[class*="st-key-btn_art_"] button:active {
        background: #DBEAFE !important;
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
        transform: translateY(1px);
    }

    /* Artifact Manager Classifier Selectbox */
    div[class*="st-key-artifact_model_sel"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #17243D !important;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[class*="st-key-artifact_model_sel"] [data-baseweb="select"] > div:hover {
        border-color: #2563EB !important;
    }
    div[class*="st-key-artifact_model_sel"] [data-baseweb="select"] span {
        font-size: 13px !important;
        color: #17243D !important;
    }

    /* Artifact Manager Disclosure Expanders */
    div[class*="st-key-art_card_"] [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        margin-top: 14px !important;
        margin-bottom: 2px !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] summary {
        font-size: 12.5px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        padding: 7px 12px !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] summary:hover {
        color: #2563EB !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 8px 12px 12px 12px !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] pre {
        max-height: 400px !important;
        border-radius: 6px !important;
    }

    /* ---- Standalone Dark Mode Code Viewer (VS Code / GitHub Dark Theme) ---- */
    [data-testid="stCode"],
    [data-testid="stCodeBlock"],
    div[class*="st-key-art_card_"] [data-testid="stCode"],
    div[class*="st-key-art_card_"] [data-testid="stCodeBlock"],
    div[class*="st-key-artifact_workspace"] [data-testid="stCode"],
    div[class*="st-key-artifact_workspace"] [data-testid="stCodeBlock"] {
        background: #0D1117 !important;
        background-color: #0D1117 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
        position: relative !important;
    }

    [data-testid="stCode"] pre,
    [data-testid="stCodeBlock"] pre,
    div[class*="st-key-art_card_"] [data-testid="stExpander"] pre,
    div[class*="st-key-artifact_workspace"] [data-testid="stExpander"] pre {
        background: #0D1117 !important;
        background-color: #0D1117 !important;
        color: #E6EDF3 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
        max-height: 400px !important;
        overflow: auto !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stCode"] code,
    [data-testid="stCodeBlock"] code,
    div[class*="st-key-art_card_"] pre code,
    div[class*="st-key-artifact_workspace"] pre code,
    [data-testid="stExpander"] pre code {
        background: transparent !important;
        background-color: transparent !important;
        color: #E6EDF3 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
    }

    /* CRITICAL FIX: The copy button container overlay must ALWAYS be transparent */
    [data-testid="stCode"] > div,
    [data-testid="stCodeBlock"] > div,
    div[class*="StyledCopyButtonContainer"],
    div:has(> button[data-testid="stCodeCopyButton"]),
    div:has(> button[title*="Copy"]) {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        pointer-events: none !important;
    }

    /* Scrollbars inside code viewer */
    [data-testid="stCode"] pre::-webkit-scrollbar,
    [data-testid="stCodeBlock"] pre::-webkit-scrollbar,
    div[class*="st-key-art_card_"] pre::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    [data-testid="stCode"] pre::-webkit-scrollbar-track,
    [data-testid="stCodeBlock"] pre::-webkit-scrollbar-track,
    div[class*="st-key-art_card_"] pre::-webkit-scrollbar-track {
        background: #0D1117 !important;
    }
    [data-testid="stCode"] pre::-webkit-scrollbar-thumb,
    [data-testid="stCodeBlock"] pre::-webkit-scrollbar-thumb,
    div[class*="st-key-art_card_"] pre::-webkit-scrollbar-thumb {
        background: #30363D !important;
        border-radius: 4px !important;
    }
    [data-testid="stCode"] pre::-webkit-scrollbar-thumb:hover,
    [data-testid="stCodeBlock"] pre::-webkit-scrollbar-thumb:hover,
    div[class*="st-key-art_card_"] pre::-webkit-scrollbar-thumb:hover {
        background: #484F58 !important;
    }

    /* Copy button inside code block */
    [data-testid="stCode"] button,
    [data-testid="stCodeBlock"] button {
        background: rgba(48, 54, 61, 0.85) !important;
        color: #C9D1D9 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stCode"] button:hover,
    [data-testid="stCodeBlock"] button:hover {
        background: #21262D !important;
        color: #58A6FF !important;
        border-color: #58A6FF !important;
    }
    [data-testid="stCode"] button svg,
    [data-testid="stCodeBlock"] button svg {
        fill: #C9D1D9 !important;
        stroke: #C9D1D9 !important;
    }

    /* Prism Syntax Highlighting (Dark Theme Tokens) */
    [data-testid="stCode"] .token.comment,
    [data-testid="stCodeBlock"] .token.comment,
    [data-testid="stCode"] .token.prolog,
    [data-testid="stCodeBlock"] .token.prolog,
    [data-testid="stCode"] .token.doctype,
    [data-testid="stCodeBlock"] .token.doctype,
    [data-testid="stCode"] .token.cdata,
    [data-testid="stCodeBlock"] .token.cdata {
        color: #8B949E !important;
        font-style: italic !important;
    }
    [data-testid="stCode"] .token.string,
    [data-testid="stCodeBlock"] .token.string,
    [data-testid="stCode"] .token.triple-quoted-string,
    [data-testid="stCodeBlock"] .token.triple-quoted-string {
        color: #7EE787 !important;
    }
    [data-testid="stCode"] .token.keyword,
    [data-testid="stCodeBlock"] .token.keyword {
        color: #FF7B72 !important;
        font-weight: 600 !important;
    }
    [data-testid="stCode"] .token.function,
    [data-testid="stCodeBlock"] .token.function {
        color: #D2A8FF !important;
    }
    [data-testid="stCode"] .token.number,
    [data-testid="stCodeBlock"] .token.number,
    [data-testid="stCode"] .token.boolean,
    [data-testid="stCodeBlock"] .token.boolean {
        color: #79C0FF !important;
    }
    [data-testid="stCode"] .token.operator,
    [data-testid="stCodeBlock"] .token.operator,
    [data-testid="stCode"] .token.punctuation,
    [data-testid="stCodeBlock"] .token.punctuation {
        color: #C9D1D9 !important;
    }
    [data-testid="stCode"] .token.class-name,
    [data-testid="stCodeBlock"] .token.class-name {
        color: #FFA657 !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1.5px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 13px;
        color: #64748B;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background: transparent !important;
    }

    /* ---- Form & Widget Labels Typography (Bigger & High Contrast) ---- */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] span {
        font-size: 15.5px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 4px !important;
    }
    .stRadio div[role="radiogroup"] label p {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    __DARK_MODE_OVERRIDES__
    __TRAINING_PURGE_CSS__
    </style>
    """

    dark_overrides = """
    /* ============================================================== */
    /* ---- DARK MODE COMPREHENSIVE OVERRIDES (back2.png) ----------- */
    /* ============================================================== */
    body, .stApp {
        color: #F8FAFC !important;
    }

    /* Headings, text, labels, and spans */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: #F8FAFC !important;
    }

    /* Stepper / Breadcrumb in Dark Mode */
    .stepper-wrap {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        box-shadow: none !important;
    }
    .step-label {
        color: #F8FAFC !important;
        font-size: 17px !important;
        font-weight: 700 !important;
    }
    .step-sub {
        color: #94A3B8 !important;
        font-size: 13px !important;
    }
    .step-node.active .step-label {
        color: #60A5FA !important;
        font-weight: 800 !important;
    }
    .step-node.done .step-label {
        color: #F8FAFC !important;
    }
    .step-node.idle .step-label {
        color: #94A3B8 !important;
    }
    .step-node.idle .step-num {
        background: #1E293B !important;
        color: #94A3B8 !important;
    }
    .step-arrow {
        color: #475569 !important;
        font-size: 26px !important;
    }

    /* Playing Cards in Dark Mode */
    div[class*="st-key-card_"] {
        background: #111827 !important;
        background-color: #111827 !important;
        border: 1.5px solid #334155 !important;
        box-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.4),
            0 8px 16px -2px rgba(0, 0, 0, 0.5),
            0 20px 36px -4px rgba(0, 0, 0, 0.6) !important;
    }
    div[class*="st-key-card_"]:hover {
        border-color: #64748B !important;
        box-shadow: 
            0 4px 8px rgba(0, 0, 0, 0.5),
            0 12px 24px -2px rgba(0, 0, 0, 0.6),
            0 28px 48px -4px rgba(0, 0, 0, 0.7) !important;
        transform: translateY(-3px) !important;
    }
    div[class*="st-key-card_"]:has(input:checked),
    div[class*="st-key-card_"]:has([aria-checked="true"]) {
        border: 2px solid #10B981 !important;
        background: #111827 !important;
        background-color: #111827 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Results KPI Summary Card in Dark Mode */
    div[class*="st-key-results_kpi_card"] {
        background: #111827 !important;
        background-color: #111827 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 16px !important;
        padding: 24px 28px !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
        box-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.4),
            0 8px 16px -2px rgba(0, 0, 0, 0.5),
            0 20px 36px -4px rgba(0, 0, 0, 0.6) !important;
    }
    div[class*="st-key-results_kpi_card"] [data-testid="metric-container"],
    div[class*="st-key-results_kpi_card"] [data-testid="stMetric"] {
        background: #1E293B !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
    }

    /* Checkbox text & badge in Dark Mode */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] label span {
        color: #F8FAFC !important;
        font-size: 17px !important;
        font-weight: 700 !important;
    }
    [data-testid="stCheckbox"] label code {
        background: rgba(37, 99, 235, 0.25) !important;
        color: #93C5FD !important;
        border: 1px solid rgba(147, 197, 253, 0.3) !important;
        font-size: 12.5px !important;
        padding: 3px 9px !important;
    }

    /* Popovers & Gear Button in Dark Mode */
    [data-testid="stPopover"] button,
    [data-testid="stPopoverButton"] button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopoverButton"] > button {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stPopover"] button:hover,
    [data-testid="stPopoverButton"] button:hover {
        background: #334155 !important;
        border-color: #3B82F6 !important;
        color: #38BDF8 !important;
    }
    [data-testid="stPopoverBody"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stPopoverBody"] label,
    [data-testid="stPopoverBody"] span,
    [data-testid="stPopoverBody"] p {
        color: #F8FAFC !important;
    }

    /* Secondary Buttons in Dark Mode */
    .stButton > button[kind="secondary"] {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #334155 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
    }

    /* Dark Mode Toggle Switch */
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] label p {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    /* Metric Containers (Results KPI) in Dark Mode */
    [data-testid="metric-container"] {
        background: #111827 !important;
        border: 1.5px solid #334155 !important;
        box-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.4),
            0 8px 16px -2px rgba(0, 0, 0, 0.5),
            0 20px 36px -4px rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    /* Tabs in Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1.5px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Selectbox, Inputs & Labels in Dark Mode */
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    [data-baseweb="select"] > div {
        background: #1E293B !important;
        border-color: #475569 !important;
        color: #F8FAFC !important;
    }
    [data-baseweb="select"] span {
        color: #F8FAFC !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: #94A3B8 !important;
    }

    /* Results Diagnostic Box in Dark Mode */
    div[style*="background:#FFFFFF;border:1px solid #E2E8F0"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
    }
    div[style*="background:#FFFFFF;border:1px solid #E2E8F0"] div {
        color: #F8FAFC !important;
    }
    div[style*="background:#FFFFFF;border:1px solid #E2E8F0"] b {
        color: #60A5FA !important;
    }

    /* Artifact Manager in Dark Mode */
    div[class*="st-key-artifact_workspace"] {
        background: #111827 !important;
        border-color: #334155 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4), 0 8px 16px -2px rgba(0, 0, 0, 0.5) !important;
    }
    div[class*="st-key-art_card_"] {
        background: #1E293B !important;
        border-color: #334155 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    }
    div[class*="st-key-art_card_"]:hover {
        border-color: #475569 !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] {
        background: #0F172A !important;
        border-color: #334155 !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] summary {
        color: #94A3B8 !important;
    }
    div[class*="st-key-art_card_"] [data-testid="stExpander"] summary:hover {
        color: #38BDF8 !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1.5px solid #334155 !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
    }
    div[class*="st-key-artifact_workspace"] .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    div[class*="st-key-btn_art_"] button {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
    }
    div[class*="st-key-btn_art_"] button:hover {
        background: #334155 !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
    div[class*="st-key-artifact_model_sel"] [data-baseweb="select"] > div {
        background: #1E293B !important;
        border-color: #334155 !important;
    }
    div[class*="st-key-artifact_model_sel"] [data-baseweb="select"] span {
        color: #F8FAFC !important;
    }
    div[class*="st-key-artifact_workspace"] [data-testid="stExpander"] {
        background: #0F172A !important;
        border-color: #334155 !important;
    }
    div[class*="st-key-artifact_workspace"] [data-testid="stExpander"] summary {
        color: #94A3B8 !important;
    }
    div[class*="st-key-artifact_workspace"] [data-testid="stExpander"] summary:hover {
        color: #38BDF8 !important;
    }
    """ if dark_mode else ""

    training_purge_css = """
    /* ---- ISOLATED LOADING SCREEN PURGE (Wipes all stale widgets & cards) ---- */
    div[class*="st-key-card_"],
    div[class*="st-key-pipeline_params_card"],
    div[class*="st-key-sample_card_"],
    div[class*="st-key-results_kpi_card"],
    div[class*="st-key-results_wrapper"],
    div[data-testid="stCheckbox"],
    .stButton,
    div[data-testid="stPopover"],
    div[data-testid="stPopoverButton"],
    div[data-testid="stHorizontalBlock"],
    div[class*="st-key-btn_"],
    div[class*="st-key-chk_"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }
    """ if is_training else ""

    css = (
        css_template
        .replace("__BG_RULE__", bg_rule)
        .replace("__DARK_MODE_OVERRIDES__", dark_overrides)
        .replace("__TRAINING_PURGE_CSS__", training_purge_css)
    )
    return css
