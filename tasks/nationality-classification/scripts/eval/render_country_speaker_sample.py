"""Render auditable country prediction distributions from a completed sample run."""
import argparse
import base64
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_validation_per_country import COUNTRY_KO


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - radius, center + radius


def render(out, dialect):
    report = json.loads((out / "report.json").read_text())
    metrics = report["metrics"]
    countries = ("China", "Uzbekistan")
    modes = ("single", "three")
    mode_names = {"single": "녹음 1개", "three": "녹음 3개 연결"}
    labels_by_country = {}
    for country in countries:
        labels = set().union(*(set(metrics[country][mode]["distribution"]) for mode in modes))
        labels_by_country[country] = sorted(labels, key=lambda label: (
            label != country, -sum(metrics[country][mode]["distribution"].get(label, 0) for mode in modes), label))
    rows = max(len(v) for v in labels_by_country.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .65 * rows + 2.4)))
    colors = ("#8796ba", "#087e8b")
    for ax, country in zip(axes, countries):
        labels = labels_by_country[country]
        y = np.arange(len(labels))
        n = metrics[country]["single"]["n_speakers"]
        for shift, mode, color, name in zip((-.18, .18), modes, colors, ("1 recording", "3 concatenated recordings")):
            counts = [metrics[country][mode]["distribution"].get(label, 0) for label in labels]
            bars = ax.barh(y + shift, [100 * k / n for k in counts], height=.32, color=color, label=name)
            for bar, count in zip(bars, counts):
                if count:
                    ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                            "{} ({:.0f}%)".format(count, 100 * count / n), va="center", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([label + ("  [correct]" if label == country else "") for label in labels])
        ax.invert_yaxis()
        ax.set_xlim(0, 123)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Share of speakers (%)")
        ax.set_title("True label: {} | n = {} speakers".format(country, n), loc="left", fontweight="bold", pad=15)
        ax.grid(axis="x", color="#e4e8ef", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Nationality validation: where each speaker was classified", fontsize=17, fontweight="bold", x=.04, ha="left")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
    fig.text(.04, .015, "Same speakers in both modes | deployed 41-country NPU model | Uzbekistan has only 10 validation speakers", fontsize=10, color="#4a5263")
    fig.tight_layout(rect=(0, .11, 1, .94), w_pad=3)
    fig.savefig(out / "distribution.png", dpi=170, facecolor="white")
    fig.savefig(out / "distribution.svg", facecolor="white")
    plt.close(fig)
    table_rows, distribution_rows, parts_rows = [], [], []
    for country in countries:
        for mode in modes:
            m = metrics[country][mode]
            n, h1, h5 = m["n_speakers"], m["correct_top1"], m["correct_top5"]
            lo, hi = wilson(h1, n)
            parts_rows.append("<li>{} / {}: 평균 {:.1f}초, 실제 처리 구간 수 {}</li>".format(
                COUNTRY_KO.get(country, country), mode_names[mode], m["mean_recording_duration_sec"],
                ", ".join("{}구간 {}명".format(k, v) for k, v in sorted(m["actual_parts"].items()))))
            table_rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}/{} ({:.1f}%)</td><td>{:.1f}–{:.1f}%</td><td>{}/{} ({:.1f}%)</td><td>{}</td></tr>".format(
                COUNTRY_KO.get(country, country), mode_names[mode], n, h1, n, 100*h1/n,
                100*lo, 100*hi, h5, n, 100*h5/n, m["rejections_or_errors"]))
        for label in labels_by_country[country]:
            vals = []
            for mode in modes:
                n = metrics[country][mode]["n_speakers"]
                count = metrics[country][mode]["distribution"].get(label, 0)
                vals.append("{}명 ({:.1f}%)".format(count, 100*count/n))
            distribution_rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                COUNTRY_KO.get(country, country), COUNTRY_KO.get(label, label), *vals))
    dialect_html = ""
    if dialect:
        d = json.loads(dialect.read_text())
        dialect_html = """<h2>방언 실험 결과</h2><p>2026-10-14 18:19 종료. 지역 방언 5종 + 표준어 후보의 6클래스 비교이며 북한말은 제외했습니다.</p>
        <table><tr><th>지표</th><th>기존 모델</th><th>이번 후보</th></tr>
        <tr><td>검증 정확도 (3,600개)</td><td>{:.2f}%</td><td>{:.2f}%</td></tr>
        <tr><td>테스트 정확도 (6,000개)</td><td>{:.2f}%</td><td>{:.2f}%</td></tr>
        <tr><td>테스트 Macro F1</td><td>{:.4f}</td><td>{:.4f}</td></tr></table>
        <p>테스트에서 {:.2f}%p 하락하여 개선을 확인하지 못했습니다. 기존 모델을 교체하지 않았습니다.
        기존에 개발 과정에서 확인한 테스트셋이며, 실제 부스 마이크 성능은 별도 검증이 필요합니다.
        표준어는 데이터의 Dialect 01 코드에 따른 잠정 라벨입니다.</p>""".format(
            100*d["baseline_valid"]["top1"], 100*d["selected"]["top1"],
            100*d["baseline_test"]["top1"], 100*d["selected_test"]["top1"],
            d["baseline_test"]["macro_f1"], d["selected_test"]["macro_f1"], -d["improvement_pp"])
    image_data = base64.b64encode((out / "distribution.png").read_bytes()).decode()
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>중국·우즈베키스탄 국적 분류 검증 결과</title><style>
    body{{font:16px/1.7 system-ui,sans-serif;color:#233044;max-width:1100px;margin:40px auto;padding:0 22px;background:#f8fafc}}
    h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;margin:20px 0;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce3ed}}th{{background:#eaf0f7}}
    img{{width:100%;background:white}}code{{word-break:break-all}}.note{{padding:14px 18px;border-left:4px solid #087e8b;background:#edf7f7}}.muted{{color:#546176;font-size:14px}}
    </style><h1>중국·우즈베키스탄 국적 분류 검증</h1>
    <p class="note"><b>중국 50명 · 우즈베키스탄 10명</b>을 실제 운영 NPU 모델로 재평가했습니다.
    우즈베키스탄은 validation 전체가 10명이라 요청한 50명을 확보할 수 없습니다. 학습 화자를 추가하거나 같은 사람을 중복 집계하지 않았습니다.</p>
    <p>모델: {model}. 각 사람의 한국어 녹음을 입력해 41개 국가 중 1순위 예측을 평가했습니다. 두 나라만 고르는 이진 분류가 아닙니다.</p>
    <table><tr><th>정답 국적</th><th>입력</th><th>화자 수</th><th>1순위 정확도</th><th>95% Wilson 구간</th><th>상위 5개 포함률</th><th>거절/오류</th></tr>{scores}</table>
    <p class="muted">각 모드에서 화자당 예측 1건입니다. 우즈베키스탄은 한 명의 정오답으로 정확도가 10%p 바뀝니다. 신뢰구간은 이 표본의 이항 비율에 대한 근사이며 데이터셋 편향을 포함하지 않습니다.</p>
    <h2>1순위 예측 국가 분포</h2><img alt="중국 및 우즈베키스탄 화자의 예측 국가별 인원과 비율" src="data:image/png;base64,{image}">
    <table><tr><th>정답 국적</th><th>예측 국적</th><th>녹음 1개</th><th>녹음 3개 연결</th></tr>{distributions}</table>
    <h2>평가 방식</h2><ul>
    <li>기존 validation 명단: 중국 68명/477개 녹음, 우즈베키스탄 10명/74개 녹음. 학습 명단과 화자 ID 중복 없음.</li>
    <li>단어 읽기(RW)를 제외한 3~20초 녹음이 3개 이상인 화자: 중국 65명, 우즈베키스탄 10명. 해시 순서(seed 42)로 중국 50명과 우즈베키스탄 10명을 고정했습니다.</li>
    <li>이전 검증 요청에 보관된 음성 180개를 복원했습니다. 원본 manifest, 화자·국적·경로 연결과 녹음 길이를 검사했으며 과거 정오답으로 선별하지 않았습니다.</li>
    <li>1개 모드는 선택한 첫 녹음, 3개 모드는 그 녹음을 포함해 같은 사람의 3개 녹음을 이어 붙입니다. 실제 연속 발화 실험은 아닙니다.</li>
    <li>서비스의 /predict_long에 parts=1 또는 3을 요청했습니다. 실제 서비스는 길이에 따라 구간 수를 조정하며, 요청 수보다 많거나 적을 수 있습니다. 시간 기준 분할이므로 실제 문장 경계와 다를 수 있습니다.</li>
    <li>예측 거절과 오류는 오답에 포함하며 평가 후 표본을 바꾸지 않았습니다. 현재 검증셋 기준 수치이며 새로운 부스 녹음에 대한 성능은 아닙니다.</li></ul>
    <p>실제 입력 길이와 처리 구간:</p><ul>{parts}</ul>
    <p class="muted">완료: {completed}<br>Manifest SHA256: <code>{sha}</code></p>{dialect}
    </html>""".format(model=escape(report["model_health"]["model"]), scores="".join(table_rows),
        image=image_data, distributions="".join(distribution_rows), completed=escape(report["completed_at"]),
        sha=report["protocol"]["manifest_sha256"], parts="".join(parts_rows), dialect=dialect_html)
    (out / "report.html").write_text(document, encoding="utf-8")
    print(out / "report.html")
    print(out / "distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--dialect", type=Path)
    args = parser.parse_args()
    render(args.out, args.dialect)
