import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import time
import urllib3
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

############################################
#                Core Bot                  #
############################################
class KISStockBot:
    """한국투자증권 OpenAPI를 이용한 코스피200 종목 분석 봇 (v11)"""

    def __init__(self, app_key, app_secret, acc_no, acc_prod_cd="01"):
        self.base_url  = "https://openapi.koreainvestment.com:9443"
        self.app_key   = app_key
        self.app_secret= app_secret
        self.acc_no    = acc_no
        self.acc_prod_cd = acc_prod_cd
        self.access_token = None

    # ── 인증 ──────────────────────────────────────
    def get_access_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        res = requests.post(url,
            headers={"content-type": "application/json"},
            data=json.dumps({"grant_type": "client_credentials",
                             "appkey": self.app_key,
                             "appsecret": self.app_secret}),
            verify=False)
        if res.status_code == 200:
            self.access_token = res.json().get("access_token")
            return self.access_token
        raise RuntimeError(f"토큰 발급 실패: {res.status_code} | {res.text}")

    # ── 종목 리스트 ────────────────────────────────
    def get_kospi200_list(self):
        return {
            "005930": "삼성전자",       "000660": "SK하이닉스",
            "035420": "NAVER",          "005380": "현대차",
            "051910": "LG화학",         "006400": "삼성SDI",
            "035720": "카카오",         "028260": "삼성물산",
            "012330": "현대모비스",     "068270": "셀트리온",
            "207940": "삼성바이오로직스","105560": "KB금융",
            "055550": "신한지주",       "003670": "포스코퓨처엠",
            "086790": "하나금융지주",
        }

    # ── 시세 조회 ──────────────────────────────────
    def get_stock_daily_price(self, stock_code):
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010400",
        }
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }
        res = requests.get(url, headers=headers, params=params, verify=False)
        if res.status_code == 200:
            return res.json().get("output")
        raise RuntimeError(f"시세 조회 실패: {stock_code} | {res.status_code}")

    # ── 데이터 전처리 & 지표 계산 ─────────────────
    @staticmethod
    def _to_num(series):
        return pd.to_numeric(series, errors="coerce")

    def _prep_df(self, price_data: list) -> pd.DataFrame:
        df = pd.DataFrame(price_data).copy()
        for col in ["stck_oprc","stck_hgpr","stck_lwpr","stck_clpr","acml_vol"]:
            df[col] = self._to_num(df[col])

        # 최신 데이터가 아래로 오도록 정렬
        df = df.iloc[::-1].reset_index(drop=True)

        # 날짜 변환
        if "stck_bsop_date" in df.columns:
            df["stck_bsop_date"] = pd.to_datetime(
                df["stck_bsop_date"], format="%Y%m%d", errors="coerce")

        c = df["stck_clpr"]

        # ── 이동평균선 ──
        df["ma5"]  = c.rolling(5).mean()
        df["ma20"] = c.rolling(20).mean()
        df["ma60"] = c.rolling(60).mean()          # ★ 60일선 추가

        # ── 수익률 / 모멘텀 ──
        df["ret1"] = c.pct_change()
        df["ret5"] = c.pct_change(5)

        # ── 볼린저밴드 (20일, 2σ) ★ 추가 ──
        df["bb_mid"]   = c.rolling(20).mean()
        df["bb_std"]   = c.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ── MACD (12, 26, 9) ★ 추가 ──
        ema12          = c.ewm(span=12, adjust=False).mean()
        ema26          = c.ewm(span=26, adjust=False).mean()
        df["macd"]     = ema12 - ema26
        df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]= df["macd"] - df["macd_sig"]

        # ── RSI (14) ──
        delta = c.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi14"] = 100 - (100 / (1 + gain / loss))

        # ── 거래량 ──
        df["vol_ma5"]  = df["acml_vol"].rolling(5).mean()
        df["vol_ma20"] = df["acml_vol"].rolling(20).mean()
        df["vol_z20"]  = ((df["acml_vol"] - df["vol_ma20"]) /
                          df["acml_vol"].rolling(20).std())

        # ── 변동성 ──
        df["volatility10"] = df["ret1"].rolling(10).std()

        # ── 거래대금 ──
        df["trd_val"]      = c * df["acml_vol"]
        df["trd_val_ma20"] = df["trd_val"].rolling(20).mean()

        return df

    # ── 개별 종목 분석 ────────────────────────────
    def analyze_stock(self, stock_code: str, stock_name: str,
                      min_trd_val: float, weights: dict):
        raw = self.get_stock_daily_price(stock_code)
        df  = self._prep_df(raw)
        if len(df) < 30:
            raise ValueError("분석에 필요한 데이터가 부족합니다")

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # 유동성 필터
        avg_trd_val20 = float(latest["trd_val_ma20"]) if pd.notna(latest["trd_val_ma20"]) else 0.0
        liquidity_pass = bool(avg_trd_val20 >= min_trd_val)

        score, signals = 0.0, []

        # 1) 상승 추세 (5일선 > 20일선)
        w = weights.get("trend", 1.0)
        if latest["ma5"] > latest["ma20"]:
            score += 4 * w
            signals.append("✅ 상승 추세 진입 (단기선이 장기선 돌파)")
        # ★ 중장기 추세 (20일선 > 60일선)
        if pd.notna(latest["ma60"]) and latest["ma20"] > latest["ma60"]:
            score += 1.5 * w
            signals.append("✅ 중장기 상승 추세 (20일선 > 60일선)")

        # 2) 최근 5일 상승 속도
        w    = weights.get("momentum", 1.0)
        mom5 = latest["ret5"] if pd.notna(latest["ret5"]) else 0.0
        if mom5 > 0.05:
            score += 3 * w
            signals.append(f"✅ 강한 상승세 지속 중 (+{mom5*100:.1f}%)")
        elif mom5 > 0.02:
            score += 2 * w
            signals.append(f"✅ 상승세 (+{mom5*100:.1f}%)")
        elif mom5 < -0.03:
            score -= 1 * w
            signals.append(f"⚠️ 최근 하락 중 ({mom5*100:.1f}%)")

        # 3) MACD ★ 추가
        w      = weights.get("macd", 1.0)
        mh     = latest["macd_hist"] if pd.notna(latest["macd_hist"]) else 0.0
        mh_prev= prev["macd_hist"]   if pd.notna(prev["macd_hist"])   else 0.0
        if mh > 0 and mh_prev <= 0:
            score += 2.5 * w
            signals.append("✅ MACD 골든크로스 (매수 전환 신호)")
        elif mh > 0:
            score += 1.0 * w
            signals.append("✅ MACD 상승 국면")
        elif mh < 0 and mh_prev >= 0:
            score -= 1.0 * w
            signals.append("⚠️ MACD 데드크로스 (하락 전환 신호)")

        # 4) 볼린저밴드 ★ 추가
        w  = weights.get("bollinger", 1.0)
        bb = latest["bb_pct"] if pd.notna(latest["bb_pct"]) else 0.5
        if bb < 0.2:
            score += 2.0 * w
            signals.append("⚡ 볼린저밴드 하단 근접 (반등 가능성)")
        elif 0.4 <= bb <= 0.6:
            score += 1.0 * w
            signals.append("✅ 볼린저밴드 중간 안정권")
        elif bb > 0.9:
            score -= 1.0 * w
            signals.append("⚠️ 볼린저밴드 상단 돌파 (과열 주의)")

        # 5) 과열도 지표 (RSI)
        w   = weights.get("rsi", 1.0)
        rsi = latest["rsi14"] if pd.notna(latest["rsi14"]) else np.nan
        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                score += 1.5 * w
                signals.append(f"✅ 적정 가격대 (과열도: {rsi:.0f})")
            elif rsi < 30:
                score += 0.5 * w
                signals.append(f"⚡ 저평가 구간, 반등 가능성 (과열도: {rsi:.0f})")
            elif rsi > 75:
                score -= 0.5 * w
                signals.append(f"⚠️ 고평가 구간, 조정 가능성 (과열도: {rsi:.0f})")

        # 6) 거래 활발도
        w    = weights.get("volume", 1.0)
        volz = latest["vol_z20"] if pd.notna(latest["vol_z20"]) else 0.0
        if volz >= 1.5:
            score += 2 * w
            signals.append("✅ 매우 활발한 거래 (평균의 1.5배 이상)")
        elif volz >= 0.5:
            score += 1 * w
            signals.append("✅ 거래 증가 중")

        # 7) 가격 변동성 페널티
        vol10 = latest["volatility10"] if pd.notna(latest["volatility10"]) else 0.0
        if vol10 >= 0.035:
            score -= 1.0
            signals.append("⚠️ 가격 변동 매우 큼 (위험 높음)")
        elif vol10 >= 0.025:
            score -= 0.5
            signals.append("⚠️ 가격 변동 다소 큼")

        # 8) 어제 대비 상승
        day_chg = (latest["stck_clpr"] - prev["stck_clpr"]) / prev["stck_clpr"]
        if day_chg > 0:
            score += 1
            signals.append(f"✅ 어제보다 {day_chg*100:.2f}% 상승")

        return {
            "code":          stock_code,
            "name":          stock_name,
            "price":         int(latest["stck_clpr"]),
            "ma5":           float(latest["ma5"])  if pd.notna(latest["ma5"])  else None,
            "ma20":          float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
            "ma60":          float(latest["ma60"]) if pd.notna(latest["ma60"]) else None,
            "rsi":           float(rsi)  if pd.notna(rsi)  else None,
            "macd_hist":     float(mh),
            "bb_pct":        float(bb),
            "vol_z20":       float(volz),
            "mom5":          float(mom5),
            "volatility10":  float(vol10),
            "avg_trd_val20": avg_trd_val20,
            "liquidity_pass":liquidity_pass,
            "score":         round(score, 2),
            "signals":       signals,
            "df":            df,
        }

    # ── 병렬 전체 분석 ★ 추가 ───────────────────
    def analyze_all_parallel(self, stock_dict, min_trd_val, weights, progress_cb=None):
        results, errors = [], []
        total = len(stock_dict)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(self.analyze_stock, code, name, min_trd_val, weights): (code, name)
                for code, name in stock_dict.items()
            }
            for i, fut in enumerate(as_completed(futures)):
                code, name = futures[fut]
                try:
                    r = fut.result()
                    if r["liquidity_pass"]:
                        results.append(r)
                except Exception as e:
                    errors.append((name, str(e)))
                if progress_cb:
                    progress_cb(i + 1, total, name)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results, errors


############################################
#              Chart Functions             #
############################################

def plot_stock_chart(df: pd.DataFrame, stock_name: str):
    """3단 차트: 주가+이평선+볼린저밴드 / MACD / 거래량"""
    df = df.copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f"{stock_name} 주가 차트", "MACD", "거래량"),
        row_heights=[0.55, 0.25, 0.20]
    )

    # ── 1행: 캔들 ──
    fig.add_trace(go.Candlestick(
        x=df["stck_bsop_date"],
        open=df["stck_oprc"], high=df["stck_hgpr"],
        low=df["stck_lwpr"],  close=df["stck_clpr"],
        name="주가",
        increasing_line_color="red", decreasing_line_color="blue"
    ), row=1, col=1)

    # 이동평균선
    for col, color, label in [
        ("ma5","orange","5일 평균선"),
        ("ma20","royalblue","20일 평균선"),
        ("ma60","green","60일 평균선"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["stck_bsop_date"], y=df[col],
                name=label, line=dict(width=1, color=color)
            ), row=1, col=1)

    # 볼린저밴드
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["stck_bsop_date"], y=df["bb_upper"],
            line=dict(width=0.8, color="gray", dash="dot"),
            name="볼린저 상단", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["stck_bsop_date"], y=df["bb_lower"],
            line=dict(width=0.8, color="gray", dash="dot"),
            name="볼린저 하단",
            fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
            showlegend=False
        ), row=1, col=1)

    # ── 2행: MACD ──
    if "macd_hist" in df.columns:
        macd_colors = ["red" if v >= 0 else "blue"
                       for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df["stck_bsop_date"], y=df["macd_hist"],
            name="MACD 히스토그램", marker_color=macd_colors, showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["stck_bsop_date"], y=df["macd"],
            name="MACD선", line=dict(width=1, color="orange")
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["stck_bsop_date"], y=df["macd_sig"],
            name="시그널선", line=dict(width=1, color="royalblue")
        ), row=2, col=1)

    # ── 3행: 거래량 ──
    vol_colors = ["red" if (c - o) >= 0 else "blue"
                  for c, o in zip(df["stck_clpr"], df["stck_oprc"])]
    fig.add_trace(go.Bar(
        x=df["stck_bsop_date"], y=df["acml_vol"],
        name="거래량", marker_color=vol_colors, showlegend=False
    ), row=3, col=1)
    if "vol_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["stck_bsop_date"], y=df["vol_ma20"],
            name="거래량 20일평균", line=dict(width=1, color="purple", dash="dash")
        ), row=3, col=1)

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02)
    )
    return fig


def plot_radar(stock: dict):
    """레이더 차트 - 6개 지표 한눈에 비교"""
    def norm(v, lo, hi):
        return max(0, min(10, (v - lo) / (hi - lo) * 10))

    vals = [
        norm((stock["ma5"] or 0) - (stock["ma20"] or 0), -5000, 5000),  # 추세
        norm((stock["mom5"] or 0) * 100, -5, 10),                        # 모멘텀
        norm(stock["macd_hist"] or 0, -500, 500),                        # MACD
        norm(1 - (stock["bb_pct"] or 0.5), 0, 1) * 10,                  # 볼린저
        norm(100 - abs((stock["rsi"] or 50) - 55), 40, 100),             # RSI
        norm(stock["vol_z20"] or 0, -1, 3),                              # 거래량
    ]
    cats = ["상승추세", "상승속도", "MACD", "볼린저밴드", "과열도", "거래활발도"]
    vals += [vals[0]]; cats += [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(255,100,100,0.2)",
        line_color="red", name=stock["name"]
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        showlegend=False, height=320,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig


def plot_comparison(results: list, top_n: int):
    """추천 종목 수익률 & 점수 비교 차트"""
    names  = [r["name"]                        for r in results[:top_n]]
    mom5   = [(r["mom5"] or 0) * 100           for r in results[:top_n]]
    scores = [r["score"]                        for r in results[:top_n]]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("최근 5일 수익률 (%)", "추천 점수"))
    fig.add_trace(go.Bar(
        x=names, y=mom5,
        marker_color=["red" if v >= 0 else "blue" for v in mom5],
        name="5일 수익률"
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=names, y=scores,
        marker_color="steelblue", name="추천 점수"
    ), row=1, col=2)
    fig.update_layout(height=360, showlegend=False, hovermode="x")
    return fig


def to_excel(results: list) -> bytes:
    """엑셀 다운로드용 바이트 생성"""
    rows = [{
        "순위":          i + 1,
        "종목명":        r["name"],
        "종목코드":      r["code"],
        "현재가":        r["price"],
        "추천점수":      r["score"],
        "최근5일수익률(%)": round((r["mom5"] or 0) * 100, 2),
        "과열도(RSI)":   round(r["rsi"], 1)       if r["rsi"]       else None,
        "MACD히스토그램":round(r["macd_hist"], 2) if r["macd_hist"] else None,
        "볼린저밴드위치":round(r["bb_pct"], 2)    if r["bb_pct"]    else None,
        "가격변동성(%)": round((r["volatility10"] or 0) * 100, 2),
        "거래활발도":    round(r["vol_z20"], 2)    if r["vol_z20"]   else None,
        "5일평균":       round(r["ma5"], 0)        if r["ma5"]       else None,
        "20일평균":      round(r["ma20"], 0)       if r["ma20"]      else None,
        "60일평균":      round(r["ma60"], 0)       if r["ma60"]      else None,
        "일평균거래액(억)": round(r["avg_trd_val20"] / 1e8, 1),
        "매수신호":      " | ".join(r["signals"]),
    } for i, r in enumerate(results)]
    buf = BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


############################################
#                Streamlit UI              #
############################################
def main():
    st.set_page_config(
        page_title="코스피200 주식 추천 봇 v11",
        layout="wide", page_icon="📈"
    )

    # ── Session State 초기화 ──
    for key, val in [("results", None), ("analysis_done", False),
                     ("top_n", 5), ("errors", [])]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── 헤더 ──
    st.title("📈 코스피200 주식 추천 시스템 v11")
    st.markdown("##### 초보자도 쉽게 이해하는 주식 분석 도구 | MACD · 볼린저밴드 · 60일선 · 병렬처리 · 레이더차트")
    st.markdown("---")

    # ── 사이드바 ──
    with st.sidebar:
        st.header("⚙️ 설정")

        with st.expander("🔑 API 인증 정보", expanded=True):
            app_key    = st.text_input("APP KEY",    type="password",
                                       help="한국투자증권에서 발급받은 앱 키")
            app_secret = st.text_input("APP SECRET", type="password",
                                       help="한국투자증권에서 발급받은 시크릿 키")
            acc_no     = st.text_input("계좌번호",   value="")

        st.markdown("---")
        st.header("📊 분석 설정")

        top_n = st.slider("추천받을 종목 개수",
                          min_value=3, max_value=10, value=5,
                          help="점수가 높은 상위 N개 종목을 추천합니다")

        min_trd_bil = st.number_input(
            "최소 거래 규모 (억원)",
            min_value=0, max_value=1000, step=10, value=100,
            help="20일 평균 거래대금 미만 종목은 제외합니다"
        )
        min_trd_val = min_trd_bil * 1e8

        st.markdown("---")
        st.subheader("🎚️ 지표 가중치 조정")
        st.caption("내 투자 스타일에 맞게 각 지표의 중요도를 조정하세요")
        weights = {
            "trend":    st.slider("📈 상승추세 (이동평균)",  0.5, 2.0, 1.0, 0.1),
            "momentum": st.slider("🚀 상승속도 (5일 수익률)", 0.5, 2.0, 1.0, 0.1),
            "macd":     st.slider("📉 MACD",                0.5, 2.0, 1.0, 0.1),
            "bollinger":st.slider("〰️ 볼린저밴드",          0.5, 2.0, 1.0, 0.1),
            "rsi":      st.slider("🌡️ 과열도 (RSI)",        0.5, 2.0, 1.0, 0.1),
            "volume":   st.slider("💹 거래 활발도",          0.5, 2.0, 1.0, 0.1),
        }

        st.markdown("---")
        analyze_btn = st.button("🔍 분석 시작하기",
                                type="primary", use_container_width=True)
        if st.session_state.analysis_done:
            if st.button("🔄 새로 분석하기", use_container_width=True):
                st.session_state.analysis_done = False
                st.session_state.results = None
                st.rerun()

    # ── 초기 화면 ──
    if not st.session_state.analysis_done and not analyze_btn:
        st.info("👈 왼쪽 메뉴에서 API 정보를 입력하고 '분석 시작하기' 버튼을 눌러주세요!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📌 이 도구는 무엇인가요?
            코스피200 종목을 자동으로 분석하여 **매수하기 좋은 종목**을 추천해드립니다.

            #### 🆕 v11 새로운 기능
            - 📊 **MACD** — 추세 전환 신호 포착
            - 〰️ **볼린저밴드** — 과열·침체 구간 파악
            - 📈 **60일 이동평균** — 중장기 추세 확인
            - ⚡ **병렬 처리** — 분석 속도 3~5배 향상
            - 🎚️ **가중치 슬라이더** — 나만의 스타일로 설정
            - 🕸️ **레이더 차트** — 6개 지표 한눈에 비교
            - 📥 **엑셀 다운로드** — 분석 결과 저장
            """)
        with col2:
            st.markdown("""
            ### 💯 추천 점수 계산 방법
            | 항목 | 최대 점수 |
            |------|---------|
            | ✅ 상승 추세 진입 (5일>20일선) | +4점 |
            | ✅ 중장기 상승 추세 (20일>60일선) | +1.5점 |
            | 🚀 강한 상승세 (5일 수익률) | +3점 |
            | 📉 MACD 골든크로스 | +2.5점 |
            | 〰️ 볼린저밴드 하단 반등 | +2점 |
            | 🌡️ 적정 가격대 (RSI 45~65) | +1.5점 |
            | 💹 거래량 급증 | +2점 |
            | ✅ 전일 대비 상승 | +1점 |
            | ⚠️ 가격 변동 큼 (페널티) | -1점 |

            **높은 점수 = 지금 사기 좋은 종목** 💡
            """)

        st.markdown("---")
        st.warning("""
        ⚠️ **투자 주의사항**
        - 이 도구는 참고용이며, 투자 손실에 대한 책임은 투자자 본인에게 있습니다.
        - 실제 투자 전에는 반드시 추가 조사를 하시기 바랍니다.
        """)
        return

    # ── 분석 실행 ──
    if analyze_btn and not st.session_state.analysis_done:
        if not app_key or not app_secret:
            st.error("❌ API KEY를 입력해주세요!")
            return

        bot = KISStockBot(app_key, app_secret, acc_no)

        with st.spinner("🔐 인증 중..."):
            try:
                bot.get_access_token()
                st.success("✅ 인증 완료! 병렬 분석을 시작합니다...")
            except Exception as e:
                st.error(f"❌ 인증 실패: {str(e)}")
                return

        prog = st.progress(0)
        stat = st.empty()

        def cb(done, total, name):
            prog.progress(done / total)
            stat.text(f"📊 {name} 분석 완료 ({done}/{total})")

        results, errors = bot.analyze_all_parallel(
            bot.get_kospi200_list(), min_trd_val, weights, cb)

        prog.empty(); stat.empty()

        for name, err in errors:
            st.warning(f"⚠️ {name} 분석 실패: {err}")

        if not results:
            st.error("❌ 조건을 만족하는 종목이 없습니다. '최소 거래 규모'를 낮춰보세요.")
            return

        st.session_state.results       = results
        st.session_state.top_n         = top_n
        st.session_state.analysis_done = True
        st.rerun()

    # ── 결과 표시 ──
    if not (st.session_state.analysis_done and st.session_state.results):
        return

    results = st.session_state.results
    top_n   = st.session_state.top_n

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["🎯 추천 종목", "📈 상세 차트", "📋 전체 결과"])

    # ═══════════════ Tab 1: 추천 종목 ═══════════════
    with tab1:
        st.subheader(f"🎯 추천 종목 TOP {min(top_n, len(results))}")
        st.caption("점수가 높을수록 지금 매수하기 좋은 종목입니다")

        # 비교 차트
        st.plotly_chart(plot_comparison(results, top_n), use_container_width=True)
        st.markdown("---")

        # 종목 카드
        cols = st.columns(min(3, max(1, min(top_n, len(results)))))
        for i, stock in enumerate(results[:top_n]):
            with cols[i % len(cols)]:
                score_emoji = ("🔥" if stock["score"] >= 9
                               else "⭐" if stock["score"] >= 6 else "✨")
                st.markdown(f"### {i+1}위. {stock['name']}")

                c1, c2 = st.columns(2)
                c1.metric("💰 현재가", f"{stock['price']:,}원")
                c2.metric(f"{score_emoji} 추천 점수", f"{stock['score']}점")

                avg_bil = int(stock["avg_trd_val20"] / 1e8)
                vol_lv  = ("높음"  if (stock["volatility10"] or 0) > 0.03
                           else "보통" if (stock["volatility10"] or 0) > 0.02 else "낮음")
                st.caption(f"💵 일평균 거래액: **{avg_bil}억원**")
                st.caption(f"📈 최근 5일 수익률: **{(stock['mom5'] or 0)*100:+.1f}%**")
                st.caption(f"📊 가격 변동성: **{vol_lv}**")

                # 레이더 차트
                st.plotly_chart(plot_radar(stock), use_container_width=True)

                if stock["signals"]:
                    with st.expander("💡 매수 신호 보기"):
                        for s in stock["signals"]:
                            st.markdown(f"- {s}")
                st.markdown("---")

        # 엑셀 다운로드
        st.download_button(
            label="📥 분석 결과 엑셀 다운로드",
            data=to_excel(results),
            file_name=f"kospi200_추천_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # ═══════════════ Tab 2: 상세 차트 ═══════════════
    with tab2:
        st.subheader("📈 종목별 상세 차트")

        options  = [f"{r['name']} ({r['code']})" for r in results[:top_n]]
        selected = st.selectbox("📊 차트를 볼 종목을 선택하세요", options=options)

        if selected:
            code = selected.split("(")[-1].split(")")[0]
            r    = next((x for x in results if x["code"] == code), None)

            if r is not None:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("💰 현재가",    f"{r['price']:,}원")
                c2.metric("📈 5일 평균",  f"{r['ma5']:,.0f}원"  if r["ma5"]  else "-")
                c3.metric("📉 20일 평균", f"{r['ma20']:,.0f}원" if r["ma20"] else "-")
                c4.metric("📊 60일 평균", f"{r['ma60']:,.0f}원" if r["ma60"] else "-")
                c5.metric("🌡️ 과열도",   f"{r['rsi']:.0f}"     if r["rsi"]  else "-")

                st.plotly_chart(plot_stock_chart(r["df"], r["name"]),
                                use_container_width=True)

                if r["signals"]:
                    st.success("**✅ 이 종목의 매수 신호**")
                    for signal in r["signals"]:
                        st.markdown(f"- {signal}")
                else:
                    st.info("현재 특별한 매수 신호가 없습니다.")

                with st.expander("💡 차트 보는 법"):
                    st.markdown("""
                    - **캔들차트**: 빨간색은 상승, 파란색은 하락
                    - **주황색 선 (5일 평균선)**: 최근 5일 평균 가격
                    - **파란색 선 (20일 평균선)**: 최근 20일 평균 가격
                    - **초록색 선 (60일 평균선)**: 최근 60일 평균 가격 (중장기 추세)
                    - **회색 점선**: 볼린저밴드 — 주가 예상 변동 범위
                    - **MACD**: 빨간 막대=상승력, 파란 막대=하락력
                    - **MACD 주황선 > 파란선**: 매수 신호 🔥
                    - **거래량 보라 점선**: 20일 평균 거래량
                    """)

    # ═══════════════ Tab 3: 전체 결과 ═══════════════
    with tab3:
        st.subheader("📋 전체 분석 결과")

        df_results = pd.DataFrame([{
            "순위":          i + 1,
            "종목명":        r["name"],
            "종목코드":      r["code"],
            "현재가":        f"{r['price']:,}원",
            "추천점수":      r["score"],
            "최근5일수익률": f"{(r['mom5'] or 0)*100:+.1f}%",
            "과열도(RSI)":   f"{r['rsi']:.0f}"      if r["rsi"]       else "-",
            "MACD":          f"{r['macd_hist']:.1f}" if r["macd_hist"] else "-",
            "볼린저밴드위치":f"{r['bb_pct']:.2f}"   if r["bb_pct"]    else "-",
            "가격변동성":    f"{(r['volatility10'] or 0)*100:.2f}%",
            "거래활발도":    f"{r['vol_z20']:.1f}"   if r["vol_z20"]   else "-",
            "5일평균":       f"{r['ma5']:,.0f}원"    if r["ma5"]       else "-",
            "20일평균":      f"{r['ma20']:,.0f}원"   if r["ma20"]      else "-",
            "60일평균":      f"{r['ma60']:,.0f}원"   if r["ma60"]      else "-",
            "일평균거래액":  f"{int(r['avg_trd_val20']/1e8)}억원",
        } for i, r in enumerate(results)])

        st.dataframe(df_results, use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 전체 결과 엑셀 다운로드",
            data=to_excel(results),
            file_name=f"kospi200_전체_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
