import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import urllib3

# SSL 경고 메시지 비활성화 (개발용)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

############################################
#                Core Bot                  #
############################################
class KISStockBot:
    """한국투자증권 OpenAPI를 이용한 코스피200 종목 분석 봇 (초보자 친화 버전)"""

    def __init__(self, app_key, app_secret, acc_no, acc_prod_cd="01"):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = app_key
        self.app_secret = app_secret
        self.acc_no = acc_no
        self.acc_prod_cd = acc_prod_cd
        self.access_token = None

    def get_access_token(self):
        """접근 토큰 발급"""
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        res = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        if res.status_code == 200:
            self.access_token = res.json().get("access_token")
            return self.access_token
        else:
            raise RuntimeError(f"토큰 발급 실패: {res.status_code} | {res.text}")

    def get_kospi200_list(self):
        """코스피200 종목 리스트"""
        return {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "035420": "NAVER",
            "005380": "현대차",
            "051910": "LG화학",
            "006400": "삼성SDI",
            "035720": "카카오",
            "028260": "삼성물산",
            "012330": "현대모비스",
            "068270": "셀트리온",
            "207940": "삼성바이오로직스",
            "105560": "KB금융",
            "055550": "신한지주",
            "003670": "포스코퓨처엠",
            "086790": "하나금융지주",
        }

    def get_stock_daily_price(self, stock_code):
        """종목 일별 시세 조회"""
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

    @staticmethod
    def _to_num(series):
        return pd.to_numeric(series, errors="coerce")

    def _prep_df(self, price_data: list) -> pd.DataFrame:
        df = pd.DataFrame(price_data).copy()
        # 수치형 변환
        for col in ["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"]:
            df[col] = self._to_num(df[col])
        
        # 최신 데이터가 아래로 오도록 정렬
        df = df.iloc[::-1].reset_index(drop=True)

        # 날짜 변환
        if "stck_bsop_date" in df.columns:
            df["stck_bsop_date"] = pd.to_datetime(df["stck_bsop_date"], format="%Y%m%d", errors="coerce")

        # 기본 지표 계산
        df["ret1"] = df["stck_clpr"].pct_change()
        df["ret5"] = df["stck_clpr"].pct_change(5)
        df["ma5"] = df["stck_clpr"].rolling(5).mean()
        df["ma20"] = df["stck_clpr"].rolling(20).mean()
        df["vol_ma5"] = df["acml_vol"].rolling(5).mean()
        df["vol_ma20"] = df["acml_vol"].rolling(20).mean()
        df["vol_z20"] = (df["acml_vol"] - df["vol_ma20"]) / df["acml_vol"].rolling(20).std()
        df["volatility10"] = df["ret1"].rolling(10).std()

        # 거래대금 계산
        df["trd_val"] = df["stck_clpr"] * df["acml_vol"]
        df["trd_val_ma20"] = df["trd_val"].rolling(20).mean()

        # RSI(14) 계산
        delta = df["stck_clpr"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi14"] = 100 - (100 / (1 + rs))

        return df

    def analyze_stock(self, stock_code: str, stock_name: str, min_trd_val: float):
        """개별 종목 분석 및 스코어 산출"""
        raw = self.get_stock_daily_price(stock_code)
        df = self._prep_df(raw)
        if len(df) < 25:
            raise ValueError("분석에 필요한 데이터가 부족합니다")

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 유동성 필터
        avg_trd_val20 = float(latest["trd_val_ma20"]) if pd.notna(latest["trd_val_ma20"]) else 0.0
        liquidity_pass = bool(avg_trd_val20 >= min_trd_val)

        # 스코어링
        score = 0.0
        signals = []

        # 1) 상승 추세 진입 (5일 평균선 > 20일 평균선)
        if latest["ma5"] > latest["ma20"]:
            score += 4
            signals.append("✅ 상승 추세 진입 (단기선이 장기선 돌파)")

        # 2) 최근 5일 상승 속도
        mom5 = latest["ret5"] if pd.notna(latest["ret5"]) else 0.0
        if mom5 > 0.05:
            score += 3
            signals.append(f"✅ 강한 상승세 지속 중 (+{mom5*100:.1f}%)")
        elif mom5 > 0.02:
            score += 2
            signals.append(f"✅ 상승세 (+{mom5*100:.1f}%)")
        elif mom5 < -0.03:
            score -= 1
            signals.append(f"⚠️ 최근 하락 중 ({mom5*100:.1f}%)")

        # 3) 거래 활발도
        volz = latest["vol_z20"] if pd.notna(latest["vol_z20"]) else 0.0
        if volz >= 1.5:
            score += 2
            signals.append("✅ 매우 활발한 거래 (평균의 1.5배 이상)")
        elif volz >= 0.5:
            score += 1
            signals.append("✅ 거래 증가 중")

        # 4) 과열도 지표 (RSI)
        rsi = latest["rsi14"] if pd.notna(latest["rsi14"]) else np.nan
        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                score += 1.5
                signals.append(f"✅ 적정 가격대 (과열도: {rsi:.1f})")
            elif rsi < 30:
                score += 0.5
                signals.append(f"⚡ 저평가 구간, 반등 가능성 (과열도: {rsi:.1f})")
            elif rsi > 75:
                score -= 0.5
                signals.append(f"⚠️ 고평가 구간, 조정 가능성 (과열도: {rsi:.1f})")

        # 5) 가격 변동성 (안정성)
        vol10 = latest["volatility10"] if pd.notna(latest["volatility10"]) else 0.0
        if vol10 >= 0.035:
            score -= 1.0
            signals.append("⚠️ 가격 변동 매우 큼 (위험 높음)")
        elif vol10 >= 0.025:
            score -= 0.5
            signals.append("⚠️ 가격 변동 다소 큼")

        # 6) 어제 대비 상승
        day_chg = (latest["stck_clpr"] - prev["stck_clpr"]) / prev["stck_clpr"]
        if day_chg > 0:
            score += 1
            signals.append(f"✅ 어제보다 {day_chg*100:.2f}% 상승")

        result = {
            "code": stock_code,
            "name": stock_name,
            "price": int(latest["stck_clpr"]),
            "ma5": float(latest["ma5"]) if pd.notna(latest["ma5"]) else None,
            "ma20": float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
            "rsi": float(rsi) if pd.notna(rsi) else None,
            "volume": int(latest["acml_vol"]) if pd.notna(latest["acml_vol"]) else None,
            "vol_z20": float(volz) if pd.notna(volz) else None,
            "mom5": float(mom5) if pd.notna(mom5) else None,
            "volatility10": float(vol10) if pd.notna(vol10) else None,
            "avg_trd_val20": avg_trd_val20,
            "liquidity_pass": liquidity_pass,
            "score": round(score, 2),
            "signals": signals,
            "df": df,
        }
        return result

############################################
#                Plotting                  #
############################################

def plot_stock_chart(df: pd.DataFrame, stock_name: str):
    """주가 차트 그리기"""
    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f"{stock_name} 주가 차트", "거래량"), row_heights=[0.7, 0.3]
    )

    # 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=df["stck_bsop_date"],
            open=df["stck_oprc"], high=df["stck_hgpr"], 
            low=df["stck_lwpr"], close=df["stck_clpr"],
            name="주가",
        ),
        row=1, col=1,
    )

    # 이동평균선
    fig.add_trace(
        go.Scatter(x=df["stck_bsop_date"], y=df["ma5"], 
                   name="5일 평균선", line=dict(width=1, color='orange')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["stck_bsop_date"], y=df["ma20"], 
                   name="20일 평균선", line=dict(width=1, color='blue')),
        row=1, col=1,
    )

    # 거래량
    colors = ["red" if (c - o) >= 0 else "blue" 
              for c, o in zip(df["stck_clpr"], df["stck_oprc"])]
    fig.add_trace(
        go.Bar(x=df["stck_bsop_date"], y=df["acml_vol"], 
               name="거래량", marker_color=colors),
        row=2, col=1,
    )

    fig.update_layout(
        height=600, 
        xaxis_rangeslider_visible=False, 
        showlegend=True, 
        hovermode="x unified"
    )
    return fig

############################################
#                Streamlit                 #
############################################

def main():
    st.set_page_config(page_title="코스피200 주식 추천 봇", layout="wide")
    
    # Session State 초기화
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # 헤더
    st.title("📈 코스피200 주식 추천 시스템")
    st.markdown("##### 초보자도 쉽게 이해하는 주식 분석 도구")
    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        with st.expander("🔑 API 인증 정보", expanded=True):
            app_key = st.text_input("APP KEY", type="password", 
                                   help="한국투자증권에서 발급받은 앱 키")
            app_secret = st.text_input("APP SECRET", type="password",
                                      help="한국투자증권에서 발급받은 시크릿 키")
            acc_no = st.text_input("계좌번호", value="")

        st.markdown("---")
        st.header("📊 분석 설정")
        
        top_n = st.slider("추천받을 종목 개수", 
                         min_value=3, max_value=10, value=5,
                         help="점수가 높은 상위 N개 종목을 추천합니다")
        
        min_trd_val_krw = st.number_input(
            "최소 거래 규모 (억원)",
            min_value=0, max_value=1000, step=10, value=100,
            help="거래가 너무 적은 종목은 제외합니다"
        ) * 100_000_000  # 억원을 원으로 변환
        
        st.markdown("---")
        analyze_btn = st.button("🔍 분석 시작하기", 
                               type="primary", 
                               use_container_width=True)

    # 메인 화면
    if not st.session_state.analysis_done:
        if not analyze_btn:
            st.info("👈 왼쪽 메뉴에서 API 정보를 입력하고 '분석 시작하기' 버튼을 눌러주세요!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 📌 이 도구는 무엇인가요?
                
                코스피200 종목을 자동으로 분석하여 **매수하기 좋은 종목**을 추천해드립니다.
                
                #### 분석 항목
                - 📈 **상승 추세**: 주가가 올라가는 흐름인지 확인
                - 🚀 **상승 속도**: 최근 며칠간 얼마나 빠르게 올랐는지
                - 💰 **거래 활발도**: 사람들이 얼마나 많이 거래하는지
                - 📊 **적정 가격**: 너무 오르거나 떨어지지 않았는지
                - ⚖️ **안정성**: 가격 변동이 크지 않은지
                """)
            
            with col2:
                st.markdown("""
                ### 💯 추천 점수는 어떻게 계산하나요?
                
                각 항목별로 점수를 부여하여 합산합니다:
                
                | 항목 | 점수 |
                |-----|------|
                | ✅ 상승 추세 진입 | +4점 |
                | ✅ 강한 상승세 | +2~3점 |
                | ✅ 거래 증가 | +1~2점 |
                | ✅ 적정 가격대 | +1.5점 |
                | ✅ 어제 대비 상승 | +1점 |
                | ⚠️ 가격 변동 큼 | -0.5~-1점 |
                
                **높은 점수 = 지금 사기 좋은 종목** 💡
                """)
            
            st.markdown("---")
            st.warning("""
            ⚠️ **투자 주의사항**
            - 이 도구는 참고용이며, 투자 손실에 대한 책임은 투자자 본인에게 있습니다.
            - 실제 투자 전에는 반드시 추가 조사를 하시기 바랍니다.
            """)
            return
        
        # API 키 검증
        if not app_key or not app_secret:
            st.error("❌ API KEY를 입력해주세요!")
            return

        # 봇 초기화
        bot = KISStockBot(app_key, app_secret, acc_no)

        with st.spinner("🔐 인증 중..."):
            try:
                bot.get_access_token()
                st.success("✅ 인증 완료!")
            except Exception as e:
                st.error(f"❌ 인증 실패: {str(e)}")
                return

        stock_dict = bot.get_kospi200_list()

        # 분석 진행
        st.subheader("🔍 종목 분석 중...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        total = len(stock_dict)

        for idx, (code, name) in enumerate(stock_dict.items()):
            status_text.text(f"📊 {name} 분석 중... ({idx+1}/{total})")
            try:
                r = bot.analyze_stock(code, name, min_trd_val=min_trd_val_krw)
                if r and r.get("liquidity_pass", False):
                    results.append(r)
            except Exception as e:
                st.warning(f"⚠️ {name} 분석 실패: {e}")
            time.sleep(0.05)
            progress_bar.progress((idx + 1) / total)

        status_text.empty()
        progress_bar.empty()

        if not results:
            st.error("❌ 조건을 만족하는 종목이 없습니다. '최소 거래 규모'를 낮춰보세요.")
            return

        # 점수 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Session State에 저장
        st.session_state.results = results
        st.session_state.top_n = top_n
        st.session_state.analysis_done = True
        st.rerun()
    
    # 분석 완료 후 결과 표시
    if st.session_state.analysis_done and st.session_state.results:
        results = st.session_state.results
        top_n = st.session_state.top_n
        
        # 새로 분석 버튼
        if st.sidebar.button("🔄 새로 분석하기", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.results = None
            st.rerun()

        # 추천 종목 카드
        st.markdown("---")
        st.subheader(f"🎯 추천 종목 TOP {min(top_n, len(results))}")
        st.markdown("##### 점수가 높을수록 지금 매수하기 좋은 종목입니다")

        cols = st.columns(min(3, max(1, min(top_n, len(results)))))
        for i, stock in enumerate(results[:top_n]):
            col = cols[i % len(cols)]
            with col:
                with st.container():
                    st.markdown(f"### {i+1}위. {stock['name']}")
                    
                    # 메트릭
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("💰 현재가", f"{stock['price']:,}원")
                    with c2:
                        score_emoji = "🔥" if stock['score'] >= 8 else "⭐" if stock['score'] >= 5 else "✨"
                        st.metric(f"{score_emoji} 추천 점수", f"{stock['score']}점")
                    
                    # 상세 정보
                    avg_trd_val_billion = int(stock['avg_trd_val20'] / 100_000_000)
                    mom5_pct = stock['mom5'] * 100 if stock['mom5'] is not None else 0
                    vol_level = "높음" if (stock['volatility10'] or 0) > 0.03 else "보통" if (stock['volatility10'] or 0) > 0.02 else "낮음"
                    
                    st.caption(f"💵 일평균 거래액: **{avg_trd_val_billion}억원**")
                    st.caption(f"📈 최근 5일 수익률: **{mom5_pct:+.1f}%**")
                    st.caption(f"📊 가격 변동성: **{vol_level}**")
                    
                    # 매수 신호
                    if stock["signals"]:
                        st.markdown("**💡 매수 신호**")
                        for s in stock["signals"][:3]:  # 상위 3개만
                            st.markdown(f"- {s}")

        # 전체 결과 테이블
        st.markdown("---")
        st.subheader("📋 전체 분석 결과")
        
        df_results = pd.DataFrame([
            {
                "순위": i + 1,
                "종목명": r["name"],
                "종목코드": r["code"],
                "현재가": f"{r['price']:,}원",
                "추천점수": r["score"],
                "최근5일수익률": f"{(r['mom5'] or 0) * 100:+.1f}%",
                "과열도": f"{r['rsi']:.0f}" if r["rsi"] is not None else "-",
                "가격변동성": f"{(r['volatility10'] or 0) * 100:.2f}%",
                "거래활발도": f"{r['vol_z20']:.1f}" if r["vol_z20"] is not None else "-",
                "5일평균": f"{r['ma5']:,.0f}원" if r["ma5"] is not None else "-",
                "20일평균": f"{r['ma20']:,.0f}원" if r["ma20"] is not None else "-",
                "일평균거래액": f"{int(r['avg_trd_val20']/100_000_000)}억원",
            }
            for i, r in enumerate(results)
        ])

        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # 상세 차트
        st.markdown("---")
        st.subheader("📈 상세 차트 보기")
        
        options = [f"{r['name']} ({r['code']})" for r in results[:top_n]]
        selected = st.selectbox("📊 차트를 볼 종목을 선택하세요", options=options)
        
        if selected:
            code = selected.split("(")[-1].split(")")[0]
            r = next((x for x in results if x["code"] == code), None)
            
            if r is not None:
                # 주요 지표
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("💰 현재가", f"{r['price']:,}원")
                with c2:
                    st.metric("📈 5일 평균선", f"{r['ma5']:,.0f}원" if r["ma5"] else "-")
                with c3:
                    st.metric("📉 20일 평균선", f"{r['ma20']:,.0f}원" if r["ma20"] else "-")
                with c4:
                    rsi_status = "적정" if (r['rsi'] or 0) < 70 else "고평가"
                    st.metric("🌡️ 과열도", 
                             f"{r['rsi']:.0f}" if r["rsi"] else "-",
                             delta=rsi_status,
                             delta_color="normal")
                
                # 차트
                fig = plot_stock_chart(r["df"], r["name"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 매수 신호 상세
                if r["signals"]:
                    st.success(f"**✅ 이 종목의 매수 신호**")
                    for signal in r["signals"]:
                        st.markdown(f"- {signal}")
                else:
                    st.info("현재 특별한 매수 신호가 없습니다.")
                
                # 도움말
                with st.expander("💡 차트 보는 법"):
                    st.markdown("""
                    - **캔들차트**: 빨간색은 상승, 파란색은 하락
                    - **주황색 선 (5일 평균선)**: 최근 5일 평균 가격
                    - **파란색 선 (20일 평균선)**: 최근 20일 평균 가격
                    - **5일선이 20일선 위에 있으면**: 상승 추세 🔥
                    - **거래량**: 막대가 클수록 거래가 활발함
                    """)


if __name__ == "__main__":
    main()