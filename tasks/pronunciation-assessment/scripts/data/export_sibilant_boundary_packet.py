"""Export blinded PCM and EMPTY Praat point tiers for actual expert annotation.

Empty tiers remain empty until an annotator supplies observations. Existing
TAG hints are not copied into the acoustic boundary tiers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import wave

LANDMARKS = ("frication_start", "frication_to_diffuse_noise", "first_vowel_voicing",
             "first_regular_run_start", "first_regular_run_end")


def template(duration):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '', 'xmin = 0',
             'xmax = %.9f' % duration, 'tiers? <exists>', 'size = %d' % len(LANDMARKS), 'item []:']
    for i, name in enumerate(LANDMARKS, 1):
        lines.extend(['    item [%d]:' % i, '        class = "TextTier"', '        name = "%s"' % name,
                      '        xmin = 0', '        xmax = %.9f' % duration, '        points: size = 0'])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-context", action="store_true",
                    help="export union of analysis and word-context intervals")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite an annotation packet")
    rows = [json.loads(s) for s in (a.packet / "items.private.jsonl").read_text(encoding="utf-8").splitlines()]
    a.out.mkdir(parents=True)
    (a.out / "audio").mkdir()
    mapping = []
    for who in ("expert_A", "expert_B"):
        (a.out / who).mkdir()
    for row in rows:
        interval = list(row["analysis_interval"])
        if a.include_context:
            interval = [min(interval[0], row["context_interval"][0]),
                        max(interval[1], row["context_interval"][1])]
        with wave.open(row["source_wav"], "rb") as f:
            sr = f.getframerate()
            if (sr, f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
                raise ValueError("expected original mono PCM16/16k")
            start = max(0, int(round(interval[0] * sr)))
            end = min(f.getnframes(), int(round(interval[1] * sr)))
            f.setpos(start)
            data = f.readframes(end - start)
        duration = (end - start) / sr
        with wave.open(str(a.out / "audio" / (row["id"] + ".wav")), "wb") as f:
            f.setparams((1, 2, sr, 0, "NONE", "not compressed"))
            f.writeframes(data)
        for who in ("expert_A", "expert_B"):
            (a.out / who / (row["id"] + ".TextGrid")).write_text(template(duration), encoding="utf-8")
        audio = a.out / "audio" / (row["id"] + ".wav")
        mapping.append({"id": row["id"], "source_offset_seconds": start / sr,
                        "sample_rate": sr, "start_sample": start, "end_sample": end,
                        "source_sha256": row["source_sha256"],
                        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                        "empty_template_sha256": hashlib.sha256(template(duration).encode("utf-8")).hexdigest()})
    # This manifest belongs to the analyst, not the blinded participant packet.
    (a.out / "source_offsets.private.json").write_text(json.dumps(mapping, indent=2) + "\n")
    protocol = {
        "language": "ko", "status": "templates_only_no_expert_annotations",
        "instructions": [
            "audio의 WAV와 본인 expert 폴더의 같은 ID TextGrid를 Praat에서 함께 엽니다. 다른 평가자의 파일은 보지 않습니다.",
            "고주파 마찰의 시작과, 집중된 고주파 마찰이 분산된 잡음으로 전환되는 지점을 구별합니다. 단순히 마지막 고주파 에너지가 사라진 시점과 같지 않습니다.",
            "잡음이 약해진 구간은 기식·무음·잔여 마찰 중 무엇인지 파형과 스펙트럼으로 확인합니다. 명확하지 않으면 uncertain이라고 기록합니다.",
            "first_vowel_voicing은 뒤 모음의 첫 성대 주기, regular_run_start/end는 처음 확인되는 연속 3개 규칙적 주기의 시작·끝입니다. 두 정의를 섞지 않습니다.",
            "확인할 수 있는 경계만 해당 tier에 점을 찍습니다. 음향 사건이 없거나 모호하면 임의의 시점을 채우지 않고, 별도 notes.jsonl에 absent/uncertain과 가능한 시간 범위를 기록합니다.",
            "경계가 겹치는 유성 마찰이나 초기 불규칙 발성도 허용합니다. 반드시 마찰·기식·모음 세 구간이 모두 존재한다고 가정하지 않습니다.",
            "대상 소리가 어느 위치인지 확실하지 않거나 이 파일에 없으면 target_association을 uncertain 또는 outside_clip으로 기록합니다. 다른 자음이나 다음 모음의 경계를 대신 찍지 않습니다.",
            "추천 관찰 설정은 5ms wideband 스펙트로그램과 원래 파형입니다. 6dB/oct pre-emphasis 표시도 비교할 수 있지만 원본 음성은 변경하지 않습니다.",
            "저장한 시각은 현재 WAV의 상대시간입니다. 원본 음성의 시각으로 변환할 때는 분석 담당자가 source offset을 더합니다."
        ],
        "reference": "https://sites.bu.edu/pamlab/files/2021/10/Chang2013_KL.pdf#page=10",
        "notes_schema": {"id": "opaque_id", "annotator": "expert_A_or_B", "actor_type": "human_expert",
                         "transition_kind": "aspiration|silence|mixed|overlap|none|uncertain",
                         "target_association": "confirmed|uncertain|outside_clip",
                         "landmark_status": {k: "observed|absent|uncertain|unreviewed" for k in LANDMARKS},
                         "uncertain_ranges_relative_seconds": {}, "note": ""},
        "items": len(rows), "include_word_context": a.include_context,
        "source_packet_sha256": hashlib.sha256((a.packet / "items.private.jsonl").read_bytes()).hexdigest()
    }
    (a.out / "annotation_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "expert_templates": 2 * len(rows), "completed_annotations": 0}))


if __name__ == "__main__":
    main()
