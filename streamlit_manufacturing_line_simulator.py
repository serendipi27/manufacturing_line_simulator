import os
from dataclasses import dataclass
from typing import List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ==============================
# 제조공정 병목 시뮬레이터 (교육용)
# 실행 방법: streamlit run streamlit_manufacturing_line_simulator.py
# ==============================

st.set_page_config(page_title="제조공정 병목 시뮬레이터", layout="wide")


# ------------------------------
# 한글 폰트 설정
# ------------------------------
def set_korean_font():
    """리눅스 배포 환경 및 로컬 환경을 모두 고려한 한글 폰트 설정"""
    linux_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font_names = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]

    if os.path.exists(linux_font_path):
        font_prop = fm.FontProperties(fname=linux_font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        for font_name in font_names:
            try:
                fm.findfont(font_name, rebuild_if_missing=True)
                plt.rcParams["font.family"] = font_name
                break
            except Exception:
                continue

    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


@dataclass
class StageResult:
    name: str
    process_time: float
    utilization: float
    avg_wait: float
    max_queue: int
    completed: int


# ------------------------------
# 세션 상태 관리
# ------------------------------
def initialize_session_state():
    if "current_simulation_result" not in st.session_state:
        st.session_state.current_simulation_result = None
    if "previous_simulation_result" not in st.session_state:
        st.session_state.previous_simulation_result = None


def build_kpis(summary: dict) -> dict:
    return {
        "총 완료시간(Makespan)": round(summary["makespan"], 2),
        "평균 리드타임": round(summary["avg_lead_time"], 2),
        "Throughput": round(summary["throughput"], 3),
        "예상 병목 공정": summary["bottleneck"],
    }



def save_simulation_to_session(df: pd.DataFrame, lead_df: pd.DataFrame, result_df: pd.DataFrame, summary: dict):
    """
    버튼을 눌러 새 시뮬레이션을 실행한 순간에만:
    1) 기존 current 결과를 previous로 이동
    2) 새 결과를 current에 저장
    """
    new_result = {
        "df": df.copy(),
        "lead_df": lead_df.copy(),
        "result_df": result_df.copy(),
        "summary": summary.copy(),
        "kpis": build_kpis(summary),
    }

    if st.session_state.current_simulation_result is not None:
        current_result = st.session_state.current_simulation_result
        st.session_state.previous_simulation_result = {
            "df": current_result["df"].copy(),
            "lead_df": current_result["lead_df"].copy(),
            "result_df": current_result["result_df"].copy(),
            "summary": current_result["summary"].copy(),
            "kpis": current_result["kpis"].copy(),
        }

    st.session_state.current_simulation_result = new_result


# ------------------------------
# 시뮬레이션 로직
# ------------------------------
def simulate_line(process_times: List[float], arrival_interval: float, num_jobs: int):
    """
    단순 직렬 생산라인 시뮬레이션
    - 각 작업은 순서대로 모든 공정을 통과
    - 각 공정은 동시에 1개 작업만 처리 가능
    - 작업은 일정 간격으로 투입
    """
    n_stages = len(process_times)
    stage_names = [f"공정 {chr(65 + i)}" for i in range(n_stages)]
    stage_available = [0.0] * n_stages

    records = []
    queue_events = {name: [] for name in stage_names}

    for job_id in range(1, num_jobs + 1):
        release_time = (job_id - 1) * arrival_interval
        prev_finish = release_time

        for s_idx, p_time in enumerate(process_times):
            stage_name = stage_names[s_idx]
            start_time = max(prev_finish, stage_available[s_idx])
            finish_time = start_time + p_time
            wait_time = start_time - prev_finish

            queue_events[stage_name].append((prev_finish, +1))
            queue_events[stage_name].append((start_time, -1))

            records.append(
                {
                    "job_id": job_id,
                    "stage": stage_name,
                    "release_time": release_time,
                    "arrival_time": prev_finish,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "process_time": p_time,
                    "wait_time": wait_time,
                }
            )

            stage_available[s_idx] = finish_time
            prev_finish = finish_time

    df = pd.DataFrame(records)

    total_finish_time = df.groupby("job_id")["finish_time"].max().max()
    total_start_time = df.groupby("job_id")["release_time"].min().min()
    makespan = total_finish_time - total_start_time

    stage_results = []
    for s_idx, stage_name in enumerate(stage_names):
        sdf = df[df["stage"] == stage_name].copy()
        busy_time = sdf["process_time"].sum()
        utilization = busy_time / makespan if makespan > 0 else 0
        avg_wait = sdf["wait_time"].mean()

        events = sorted(queue_events[stage_name], key=lambda x: (x[0], -x[1]))
        q = 0
        max_q = 0
        for _, delta in events:
            q += delta
            max_q = max(max_q, q)

        stage_results.append(
            StageResult(
                name=stage_name,
                process_time=process_times[s_idx],
                utilization=utilization,
                avg_wait=avg_wait,
                max_queue=max_q,
                completed=sdf["job_id"].nunique(),
            )
        )

    lead_df = (
        df.groupby("job_id")
        .agg(release_time=("release_time", "min"), finish_time=("finish_time", "max"))
        .reset_index()
    )
    lead_df["lead_time"] = lead_df["finish_time"] - lead_df["release_time"]

    throughput = num_jobs / makespan if makespan > 0 else 0

    summary = {
        "makespan": makespan,
        "throughput": throughput,
        "avg_lead_time": lead_df["lead_time"].mean(),
        "max_lead_time": lead_df["lead_time"].max(),
        "bottleneck": max(stage_results, key=lambda x: x.process_time).name,
    }

    result_df = pd.DataFrame([vars(r) for r in stage_results])
    return df, lead_df, result_df, summary


# ------------------------------
# 시각화 함수
# ------------------------------
def draw_gantt(df: pd.DataFrame):
    jobs = sorted(df["job_id"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in df.iterrows():
        y = jobs.index(row["job_id"])
        ax.barh(
            y=y,
            width=row["finish_time"] - row["start_time"],
            left=row["start_time"],
            height=0.7,
            edgecolor="black",
        )
        ax.text(
            row["start_time"] + (row["finish_time"] - row["start_time"]) / 2,
            y,
            row["stage"],
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([f"Job {j}" for j in jobs])
    ax.set_xlabel("시간")
    ax.set_ylabel("작업")
    ax.set_title("작업별 공정 진행 간트 차트")
    plt.tight_layout()
    return fig



def draw_stage_metrics(result_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(result_df["name"], result_df["utilization"])
    axes[0].set_title("공정별 가동률")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("비율")

    axes[1].bar(result_df["name"], result_df["avg_wait"])
    axes[1].set_title("공정별 평균 대기시간")
    axes[1].set_ylabel("시간")

    axes[2].bar(result_df["name"], result_df["max_queue"])
    axes[2].set_title("공정별 최대 큐 길이")
    axes[2].set_ylabel("개수")

    plt.tight_layout()
    return fig


# ------------------------------
# UI 렌더링 함수
# ------------------------------
def build_stage_editor(defaults: List[float]):
    st.sidebar.markdown("## 공정 파라미터 설정")
    p_times = []
    for i, d in enumerate(defaults):
        value = st.sidebar.slider(
            f"공정 {chr(65 + i)} 처리시간",
            min_value=1.0,
            max_value=15.0,
            value=float(d),
            step=0.5,
        )
        p_times.append(value)
    return p_times



def render_previous_kpis_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 이전 시뮬레이션 결과")

    previous_result = st.session_state.previous_simulation_result
    if previous_result is None:
        st.sidebar.info("아직 이전 실행 결과가 없습니다. 두 번째 실행부터 표시됩니다.")
        return

    prev_df = pd.DataFrame(
        {
            "KPI": list(previous_result["kpis"].keys()),
            "값": list(previous_result["kpis"].values()),
        }
    )
    st.sidebar.dataframe(prev_df, use_container_width=True, hide_index=True)



def render_main_result(result: dict):
    df = result["df"]
    lead_df = result["lead_df"]
    result_df = result["result_df"]
    summary = result["summary"]

    st.markdown("---")
    st.subheader("1. 핵심 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "총 완료시간(Makespan)",
        f"{summary['makespan']:.1f}",
        help="첫 번째 작업 투입 시점부터 마지막 작업 완료 시점까지 걸린 전체 시간입니다. 값이 작을수록 전체 라인의 처리 효율이 좋습니다.",
    )
    c2.metric(
        "평균 리드타임",
        f"{summary['avg_lead_time']:.1f}",
        help="작업 하나가 투입된 뒤 모든 공정을 마치고 완료될 때까지 걸리는 평균 시간입니다. 대기가 길어질수록 값이 커집니다.",
    )
    c3.metric(
        "Throughput",
        f"{summary['throughput']:.2f}",
        help="단위 시간당 완료된 작업 수입니다. 값이 클수록 같은 시간 안에 더 많은 작업을 처리했다는 뜻입니다.",
    )
    c4.metric(
        "예상 병목 공정",
        summary["bottleneck"],
        help="현재 설정에서 전체 흐름을 가장 제한하는 공정입니다. 일반적으로 처리시간이 가장 길고 대기와 가동률에 큰 영향을 줍니다.",
    )

    st.info(
        "메인 화면은 방금 또는 마지막으로 '시뮬레이션 실행' 버튼을 눌러 확정된 결과입니다. 사이드바에는 그 직전 실행 결과가 표시됩니다."
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("2. 공정별 성능 대시보드")
        st.dataframe(
            result_df.rename(
                columns={
                    "name": "공정명",
                    "process_time": "처리시간",
                    "utilization": "가동률",
                    "avg_wait": "평균대기시간",
                    "max_queue": "최대큐길이",
                    "completed": "완료작업수",
                }
            ),
            use_container_width=True,
        )
        st.pyplot(draw_stage_metrics(result_df))

    with right:
        st.subheader("3. 작업 흐름 간트 차트")
        st.pyplot(draw_gantt(df))

    st.subheader("4. 작업별 리드타임")
    st.line_chart(lead_df.set_index("job_id")["lead_time"])

    st.subheader("5. 상세 데이터")
    with st.expander("작업-공정 상세 로그 보기"):
        st.dataframe(df, use_container_width=True)

    st.subheader("6. 학습 가이드")
    st.markdown(
        """
### 이 결과를 보면서 생각해보세요
- 가장 오래 걸리는 공정이 실제로 대기와 가동률에 어떤 영향을 주는가?
- 병목 공정을 줄였을 때 Throughput은 얼마나 개선되는가?
- 비병목 공정을 줄였을 때는 왜 개선 폭이 작을까?
- 작업 투입 간격을 늘리거나 줄이면 큐와 리드타임은 어떻게 바뀌는가?

### 추천 실습 순서
1. **기본 라인**으로 먼저 실행합니다.
2. 병목으로 보이는 공정을 하나 선택해 처리시간을 줄여봅니다.
3. 이번에는 병목이 아닌 공정의 처리시간을 줄여봅니다.
4. 두 결과를 비교해서 "어디를 개선해야 하는가"를 정리합니다.
"""
    )


# ------------------------------
# 앱 본문
# ------------------------------
initialize_session_state()

st.title("제조공정 병목 시뮬레이터")
st.markdown(
    """
이 웹앱은 **공정 흐름, 병목, 대기, 가동률** 개념을 직접 조작하며 학습하기 위한 교육용 도구입니다.

### 사용 방법
1. 왼쪽 사이드바에서 공정별 처리시간과 작업 투입 간격을 조절합니다.
2. **시뮬레이션 실행** 버튼을 누릅니다.
3. 메인 화면에는 방금 실행한 결과가 표시됩니다.
4. 사이드바의 **이전 시뮬레이션 결과**와 비교하면서 병목 개선 효과를 확인합니다.
"""
)

with st.sidebar:
    st.markdown("## 실습 시나리오")
    scenario = st.selectbox(
        "실습 모드 선택",
        [
            "기본 라인",
            "병목 공정 개선",
            "비병목 공정 개선",
            "전체 속도 향상",
        ],
    )

if scenario == "기본 라인":
    defaults = [3.0, 5.0, 2.0, 3.0]
elif scenario == "병목 공정 개선":
    defaults = [3.0, 3.0, 2.0, 3.0]
elif scenario == "비병목 공정 개선":
    defaults = [2.0, 5.0, 2.0, 2.0]
else:
    defaults = [2.5, 4.0, 1.5, 2.5]

process_times = build_stage_editor(defaults)

with st.sidebar:
    arrival_interval = st.slider("작업 투입 간격", 1.0, 10.0, 2.0, 0.5)
    num_jobs = st.slider("총 작업 수", 5, 50, 20, 1)
    run_button = st.button("시뮬레이션 실행", use_container_width=True)

# 중요: 버튼 클릭 시 먼저 세션 상태를 갱신한 뒤,
# 그 다음 사이드바와 메인 화면을 렌더링해야 현재/이전 결과가 올바르게 보인다.
if run_button:
    df, lead_df, result_df, summary = simulate_line(process_times, arrival_interval, num_jobs)
    save_simulation_to_session(df, lead_df, result_df, summary)

# 사이드바에는 직전 실행 결과 표시
render_previous_kpis_sidebar()

# 메인 화면에는 현재(가장 마지막 실행) 결과 표시
current_result = st.session_state.current_simulation_result
if current_result is not None:
    render_main_result(current_result)
else:
    st.markdown("### 왼쪽에서 파라미터를 조절한 뒤 **시뮬레이션 실행** 버튼을 눌러주세요.")
