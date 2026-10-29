# -*- coding: utf-8 -*-
"""국적 학습 목록이 가리키는 음성 중 지워진 것을 AIHub 에서 다시 받아 16 kHz 로 복원한다.

2026-10-11 정리 때 국적 전용 음성(131 대부분 · 040~043 자유발화 · 040~043 발음 원본 48 kHz)이 지워졌다.
매니페스트와 라벨은 남아 있고, 040~043 발음 녹음은 mdd_gold_wav16k 에 16 kHz 사본이 있다.
`run_zeroth` 학습 목록(lab131full_zeroth) 기준으로 원래 경로에 남은 음성은 13% 뿐이다.

  plan      매니페스트 각 행이 지금 읽히는지 나눈다 — 원래 경로 / mdd_gold 사본 / 복원 필요 / 복원 불가.
            복원 필요 행을 AIHub zip 별로 묶어 <out>/_plan.json · <out>/_needed/<파일키>.txt 에 쓴다.
  run       zip 을 큰 것부터 하나씩 받아 필요한 wav 만 꺼내 16 kHz mono int16 으로 쓰고 tar 를 지운다.
            tar 안의 분할 zip(.partN)은 합치지 않고 tar 위에서 이어 붙인 것처럼 읽는다 — aihubshell 은
            tar · 조각 · 합친 zip 이 한때 같이 있어 파일 크기의 3배가 필요하지만, 여기서는 tar 하나뿐이다.
  manifest  경로를 지금 읽히는 파일로 바꾼 새 매니페스트 폴더(<src>_r16k)를 쓴다.
            kor001(AIHub 001 자유대화 — 승인 만료로 못 받는다)은 train 에서 같은 KsponSpeech 학습 묶음의
            안 쓴 발화로 같은 자리에서 바꾸고, valid·test 에서는 뺀다. 실제 사용자 녹음(user_laptop)은 넣지 않는다.
            그 밖에 음성이 없는 행이 --max_missing_frac(기본 0.1%)를 넘으면 아무것도 쓰지 않고 실패한다.
  probe     남은 zip 이 있는 데이터셋마다 승인이 살아 있는지 본다(받기 요청의 첫 512바이트만 읽고 끊는다).
            AIHub 승인은 데이터셋마다 유효기간이 있고, 만료되면 받기 요청에 tar 대신 안내문이 온다.
  status    진행 상황 (끝난 zip · 받는 중인 tar 크기 · 복원한 파일 수).

출력: $SPEECH_AI_DATA/nationality_wav16k/<데이터셋>/<stem 앞 2자>/<stem>.wav
      _index.tsv (stem · 데이터셋 · 파일키 · 원본 sr · 채널 · 16k 샘플 수) · _done/<파일키>.json
변환식은 resample_gold_16k.py 와 같다(wave → 채널 평균 → resample_poly → clip×32767 → int16).
원본이 이미 16 kHz mono int16 이면(131) 바이트를 그대로 쓴다 — run_zeroth 가 학습 때 읽은 것과 같다.
API 키는 ~/.aihub_apikey 또는 AIHUB_APIKEY 에서 읽고, curl 에는 표준입력 설정으로 넘긴다(프로세스 목록에 안 보인다).

    $PY tasks/nationality-classification/scripts/data/restore_nationality_audio.py plan
    $PY tasks/nationality-classification/scripts/data/restore_nationality_audio.py run --only 57685   # 1 GB 로 시험
    $PY tasks/nationality-classification/scripts/data/restore_nationality_audio.py run                 # 전부 — tmux 에서
    $PY tasks/nationality-classification/scripts/data/restore_nationality_audio.py manifest
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS

import argparse, bisect, collections, csv, io, json, os, random, re, shutil, subprocess, tarfile, threading, time, wave, zipfile
from math import gcd
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import numpy as np

SR = 16000
API = "https://api.aihub.or.kr/down/0.6/{ds}.do?fileSn={fk}"
MANIFESTS = ("train.csv", "valid.csv", "test.csv", "test_koraug_zeroth.csv")
GiB = 2 ** 30

# 데이터셋 번호 → AIHub 데이터셋키, 그리고 음성 zip (파일키, zip 이름, GB) — aihubshell -mode l (2026-10-18)
DATASET_KEY = {"131": 505, "040": 71469, "041": 71489, "042": 71490, "043": 71479}
ZIPS = {
    "131": [(57666, "TS_1. 베트남어.zip", 54), (57667, "TS_2. 영어.zip", 9), (66279, "TS_3. 일본어_add.zip", 53),
            (66280, "TS_4. 중국어_add.zip", 75), (66281, "TS_5. 태국어_add.zip", 38), (66282, "TS_6. 기타_add.zip", 74),
            (57684, "VS_1. 베트남어.zip", 7), (57685, "VS_2. 영어.zip", 1), (57686, "VS_3. 일본어.zip", 7),
            (57687, "VS_4. 중국어.zip", 9), (66283, "VS_5. 태국어_add.zip", 5), (57683, "VS_6. 기타.zip", 9)],
    "040": [(475404, "TS_Speech_sound_1.zip", 68), (475405, "TS_Speech_sound_2.zip", 39), (475411, "VS_Speech_sound.zip", 13),
            (475406, "TS_pronunciation_sound_1.zip", 60), (475407, "TS_pronunciation_sound_2.zip", 6),
            (475412, "VS_pronunciation_sound.zip", 8)],
    "041": [(475416, "TS_Speech_sound_1.zip", 70), (475417, "TS_Speech_sound_2.zip", 70), (475418, "TS_Speech_sound_3.zip", 19),
            (475424, "VS_Speech_sound.zip", 20), (475419, "TS_pronunciation_sound_1.zip", 61),
            (475420, "TS_pronunciation_sound_2.zip", 44), (475425, "VS_pronunciation_sound.zip", 13)],
    "042": [(475429, "TS_Speech_sound_1.zip", 70), (475430, "TS_Speech_sound_2.zip", 45), (475436, "VS_Speech_sound.zip", 14),
            (475431, "TS_pronunciation_sound_1.zip", 60), (475432, "TS_pronunciation_sound_2.zip", 0.64),
            (475437, "VS_pronunciation_sound.zip", 8)],
    "043": [(475441, "TS_Speech_sound_1.zip", 70), (475442, "TS_Speech_sound_2.zip", 70), (475443, "TS_Speech_sound_3.zip", 20),
            (475449, "VS_Speech_sound.zip", 20), (475444, "TS_pronunciation_sound_1.zip", 62),
            (475445, "TS_pronunciation_sound_2.zip", 43), (475450, "VS_pronunciation_sound.zip", 13)],
}
# 매니페스트 경로에는 zip 이름의 공백이 _ 로 바뀐 폴더명이 나온다 ("TS_6. 기타_add.zip" → "TS_6._기타_add")
FOLDER = {(ds, name[:-4].replace(" ", "_")): (DATASET_KEY[ds], fk, name, gb) for ds, zs in ZIPS.items() for fk, name, gb in zs}


def source_of(path):
    """매니페스트 audio_path → (데이터셋 번호, zip 폴더명). 모르면 None."""
    parts = path.split("/")
    ds = next((p[:3] for p in parts if p[:3] in DATASET_KEY and p[3:4] == "."), None)
    folder = next((p for p in parts if p[:3] in ("TS_", "VS_")), None)
    return ds, folder


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def restored_path(out, ds, stem):
    return os.path.join(out, ds, stem[:2], stem + ".wav")


def gold_path(gold_dir, stem):
    return os.path.join(gold_dir, stem[:2], stem + ".wav")


def load_gold(gold_dir):
    stems = set()
    with open(os.path.join(gold_dir, "_index.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            stems.add(line.split("\t", 1)[0])
    return stems


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


# ── plan ────────────────────────────────────────────────────────────────

def cmd_plan(a):
    gold = load_gold(a.gold)
    out = Path(a.out)
    need = collections.defaultdict(dict)                 # 파일키 → {stem: 예상 초}
    summary, lost = {}, collections.Counter()
    for name in MANIFESTS:
        p = Path(a.src) / name
        if not p.is_file():
            continue
        _, rows = read_csv(p)
        c = collections.Counter()
        for row in rows:
            path, dom = row["audio_path"], row["domain"]
            stem = stem_of(path)
            if os.path.isfile(path):
                c["원래 경로"] += 1
            elif dom.startswith("web") and stem in gold:
                c["mdd_gold 사본"] += 1
            else:
                key = source_of(path)
                if key in FOLDER:
                    fk = FOLDER[key][1]
                    if os.path.isfile(restored_path(a.out, key[0], stem)):
                        c["복원됨"] += 1
                    else:
                        c["복원 필요"] += 1
                    need[fk][stem] = float(row.get("duration_sec") or 0)
                else:
                    c[f"복원 불가({dom})"] += 1
                    lost[dom] += 1
        summary[name] = dict(c)
        print(f"[{name}] {len(rows):,}행 · " + " · ".join(f"{k} {v:,}" for k, v in c.most_common()), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "_needed").mkdir(exist_ok=True)
    zips = []
    for (ds, folder), (dkey, fk, zname, gb) in FOLDER.items():
        if fk not in need:
            continue
        stems = sorted(need[fk])
        (out / "_needed" / f"{fk}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
        est = sum(need[fk].values()) * SR * 2
        zips.append({"ds": ds, "dataset_key": dkey, "file_key": fk, "zip": zname, "folder": folder, "gb": gb,
                     "needed": len(stems), "est_out_gb": round(est / 1e9, 2)})
    zips.sort(key=lambda z: -z["gb"])
    plan = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "src": str(a.src), "manifests": summary,
            "zips": zips, "download_gb": round(sum(z["gb"] for z in zips), 1),
            "est_out_gb": round(sum(z["est_out_gb"] for z in zips), 1), "unrecoverable": dict(lost)}
    (out / "_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[plan] zip {len(zips)}개 · 받을 양 {plan['download_gb']} GB · 남길 양 약 {plan['est_out_gb']} GB · "
          f"복원 불가 {dict(lost)}", flush=True)
    for z in zips:
        print(f"  {z['ds']} {z['zip']:30s} {z['gb']:>6} GB · 필요 {z['needed']:>7,}개 · 약 {z['est_out_gb']} GB", flush=True)


# ── run ─────────────────────────────────────────────────────────────────

class Concat(io.RawIOBase):
    """tar 안에 연속으로 놓인 조각들(오프셋·크기)을 하나의 파일처럼 읽는다. 복사하지 않는다."""

    def __init__(self, path, segs):
        self.f = open(path, "rb")
        self.segs = segs
        self.starts = [0]
        for _, n in segs:
            self.starts.append(self.starts[-1] + n)
        self.size, self.pos = self.starts[-1], 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            off, seg = self.segs[i]
            within = self.pos - self.starts[i]
            self.f.seek(off + within)
            chunk = self.f.read(min(n - done, seg - within))
            if not chunk:
                break
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done

    def close(self):
        self.f.close()
        super().close()


def zip_views(tar_path):
    """tar → [(zip 이름, 조각 [(오프셋, 크기)])]. 조각은 .partN 번호순, 조각이 없으면 zip 하나.
    같은 조각이 두 번 들어 있으면(first_repeat 참고) 처음 것만 쓴다 — 둘 다 넣으면 zip 이 깨진다."""
    groups = collections.defaultdict(dict)
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            mm = re.match(r"(.*\.zip)\.part(\d+)$", m.name)
            if mm:
                groups[mm.group(1)].setdefault(int(mm.group(2)), (m.offset_data, m.size))
            elif m.name.lower().endswith(".zip"):
                groups[m.name].setdefault(0, (m.offset_data, m.size))
    return [(k, [v[i] for i in sorted(v)]) for k, v in groups.items()]


def first_repeat(tar_path):
    """받는 중인 tar 에서 앞서 나온 (이름, 크기)가 다시 시작되는 머리 위치. 없으면 None.
    AIHub 가 한 tar 에 같은 zip 조각들을 두 번 이어 보낸 적이 있다(2026-10-19 041 TS_Speech_sound_1 — 두 사본이 같았다)."""
    seen = set()
    try:
        tf = tarfile.open(tar_path, "r:")
    except (tarfile.ReadError, OSError):            # 아직 첫 머리도 다 안 왔거나 tar 가 아니다
        return None
    with tf:
        while True:
            try:
                m = tf.next()
            except tarfile.ReadError:                # 받는 중인 마지막 멤버
                return None
            if m is None:
                return None
            if (m.name, m.size) in seen:
                return m.offset
            seen.add((m.name, m.size))


def close_tar_at(tar_path, cut):
    """cut 뒤를 버리고 tar 끝 표시(0 블록 2개)를 붙인다."""
    with open(tar_path, "r+b") as f:
        f.truncate(cut)
        f.seek(cut)
        f.write(b"\0" * 1024)


def ok_mark(tar_path):
    """다 받은 tar 표시 — 있으면 다시 받지 않는다(이어받기 요청이 두 번째 사본을 다시 받아 올 수 있다)."""
    return tar_path.parent / (tar_path.name + ".ok")


def member_name(info):
    """AIHub zip 은 CP949 이름에 UTF-8 플래그가 없다 (extract_aihub_zips.py 와 같은 처리)."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def convert(job):
    """(stem, wav 바이트, 출력 경로) → (stem, 원본 sr, 채널, 16k 샘플 수, 상태). resample_gold_16k.to16k 와 같은 식."""
    stem, data, dst = job
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        if sr == SR and ch == 1 and sw == 2:
            with open(tmp, "wb") as f:
                f.write(data)
            frames = n
        else:
            from scipy.signal import resample_poly
            x = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
            x = (x - 128) / 128 if sw == 1 else x / {2: 32768.0, 4: 2147483648.0}[sw]
            if ch > 1:
                x = x.reshape(-1, ch).mean(1)
            if sr != SR:
                g = gcd(int(sr), SR)
                x = resample_poly(x, SR // g, sr // g)
            pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
            with wave.open(tmp, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
            frames = pcm.size
        os.replace(tmp, dst)
        return stem, sr, ch, frames, "ok"
    except Exception as e:
        return stem, "", "", 0, f"error {type(e).__name__}: {e}"[:200]


def api_key():
    k = os.environ.get("AIHUB_APIKEY", "").strip()
    p = Path.home() / ".aihub_apikey"
    if not k and p.is_file():
        k = p.read_text().strip()
    if not k:
        raise SystemExit("AIHub API 키가 없다 — AIHUB_APIKEY 또는 ~/.aihub_apikey")
    return k


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def download(dkey, fk, tar_path, key, tries=6):
    """curl 로 tar 를 받는다(끊기면 이어받기). 같은 멤버가 다시 오기 시작하면 거기서 끊고 tar 를 닫는다. → True/False"""
    stop = threading.Event()

    def watch():
        last, t_last = 0, time.time()
        while not stop.wait(300):
            n = tar_path.stat().st_size if tar_path.exists() else 0
            now = time.time()
            log(f"  받는 중 {n / GiB:.1f} GiB ({(n - last) / (now - t_last) / 2**20:.1f} MiB/s)")
            last, t_last = n, now

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        for t in range(1, tries + 1):
            # 10분 동안 1 KB/s 아래로 멈추면 끊고 이어받는다
            cmd = ["curl", "-L", "-sS", "-C", "-", "--speed-limit", "1024", "--speed-time", "600",
                   "-o", str(tar_path), "-w", "%{http_code}", "-K", "-", API.format(ds=dkey, fk=fk)]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                p.stdin.write(f'header = "apikey: {key}"\n')
                p.stdin.close()
            except BrokenPipeError:
                pass
            cut = None
            try:
                while p.poll() is None:
                    time.sleep(20)
                    cut = first_repeat(tar_path)
                    if cut is not None:
                        break
            finally:
                if p.poll() is None:                  # 같은 조각이 다시 오기 시작했거나 여기서 예외가 났다
                    p.terminate()
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            # communicate() 는 못 쓴다 — 위에서 닫은 stdin 을 다시 flush 하다 ValueError 로 죽는다(3.8, 2026-10-19 실제로 그랬다)
            out, err = p.stdout.read(), p.stderr.read()
            p.stdout.close()
            p.stderr.close()
            code = out.strip()[-3:]
            ok = (p.returncode == 0 and code in ("200", "206")) or (code == "416" and tar_path.exists())  # 416 = 이미 다 받음
            if cut is None and ok and tarfile.is_tarfile(str(tar_path)):
                cut = first_repeat(tar_path)          # 마지막 20초 사이에 두 번째 사본이 시작됐을 수도 있다
                if cut is None:
                    ok_mark(tar_path).touch()
                    return True
            if cut is not None:
                close_tar_at(tar_path, cut)
                log(f"  같은 zip 조각이 다시 오기 시작 — {cut / GiB:.1f} GiB 에서 끊고 tar 를 닫음")
                ok_mark(tar_path).touch()
                return True
            log(f"  curl 실패 (시도 {t}/{tries}) rc={p.returncode} http={code} {err.strip()[:200]}")
            if tar_path.exists() and tar_path.stat().st_size < 2 ** 20 and not tarfile.is_tarfile(str(tar_path)):
                msg = tar_path.read_bytes()[:600].decode("utf-8", "replace").strip()
                log(f"  서버 응답: {msg}")   # 승인 만료 등 — tar 가 아니다
                tar_path.unlink()
                return False
            if p.returncode == 33 and tar_path.exists():       # 서버가 이어받기를 안 받아 준다 → 처음부터
                tar_path.unlink()
            time.sleep(60)
        return False
    finally:
        stop.set()


def probe(dkey, fk, key, timeout=90):
    """승인이 살아 있는지 — 받기 요청을 보내 첫 512바이트만 읽고 끊는다. → (ok, 서버 안내문)"""
    cmd = ["curl", "-L", "-sS", "--max-time", str(timeout), "-K", "-", API.format(ds=dkey, fk=fk)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        p.stdin.write(f'header = "apikey: {key}"\n'.encode())
        p.stdin.close()
        head = p.stdout.read(512)
    finally:
        p.kill()
        p.wait()
    if head[257:262] == b"ustar":                        # tar 머리 — 받기가 시작됐다
        return True, ""
    return False, head.decode("utf-8", "replace").strip()[:200] or "응답 없음"


def cmd_probe(a):
    """끝나지 않은 zip 이 있는 데이터셋마다 가장 작은 zip 으로 승인을 확인한다. 전부 살아 있으면 0."""
    out = Path(a.out)
    plan = json.loads((out / "_plan.json").read_text(encoding="utf-8"))
    pending = {}
    for z in plan["zips"]:
        if (out / "_done" / f"{z['file_key']}.json").exists():
            continue
        if z["ds"] not in pending or z["gb"] < pending[z["ds"]]["gb"]:
            pending[z["ds"]] = z
    if not pending:
        log("[probe] 남은 zip 이 없다")
        return 0
    key = api_key()
    bad = 0
    for ds, z in sorted(pending.items()):
        ok, msg = probe(z["dataset_key"], z["file_key"], key)
        bad += not ok
        log(f"[probe] {ds} (데이터셋키 {z['dataset_key']}) " + ("승인 유효" if ok else f"받을 수 없음 — {msg}"))
    return 1 if bad else 0


def cmd_run(a):
    out = Path(a.out)
    plan = json.loads((out / "_plan.json").read_text(encoding="utf-8"))
    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    (out / "_done").mkdir(exist_ok=True)
    key = api_key()
    index = out / "_index.tsv"
    if not index.exists():
        index.write_text("stem\tds\tfile_key\tsrc_sr\tsrc_ch\tframes16k\n", encoding="utf-8")
    only = {int(x) for x in a.only.split(",")} if a.only else None
    fails = []
    for z in plan["zips"]:
        fk, ds = z["file_key"], z["ds"]
        if only and fk not in only:
            continue
        done_file = out / "_done" / f"{fk}.json"
        if done_file.exists():
            log(f"[skip] {ds} {z['zip']} — 이미 끝남")
            continue
        need = set((out / "_needed" / f"{fk}.txt").read_text(encoding="utf-8").split())
        todo = {s for s in need if not os.path.isfile(restored_path(a.out, ds, s))}
        if not todo:
            done_file.write_text(json.dumps({"zip": z["zip"], "needed": len(need), "found": len(need), "written": 0,
                                             "note": "이미 전부 복원됨"}, ensure_ascii=False))
            continue
        tar_path = stage / f"{fk}.tar"
        have = tar_path.stat().st_size if tar_path.exists() else 0
        free = shutil.disk_usage(stage).free
        want = z["gb"] * 1e9 * 1.05 - have + z["est_out_gb"] * 1e9 + a.reserve_gb * 1e9
        if free < want:
            log(f"[중단] 디스크 부족 — {z['zip']} 에 약 {want / 1e9:.0f} GB 필요, 여유 {free / 1e9:.0f} GB")
            fails.append(fk)
            break
        t0 = time.time()
        if tar_path.exists() and ok_mark(tar_path).exists():
            log(f"[have] {ds} {z['zip']} — 이미 다 받은 tar {have / GiB:.1f} GiB · 필요 {len(todo):,}/{len(need):,}개")
        else:
            log(f"[get ] {ds} {z['zip']} ({z['gb']} GB, 키 {fk}) · 필요 {len(todo):,}/{len(need):,}개 · 여유 {free / 1e9:.0f} GB")
            if not download(z["dataset_key"], fk, tar_path, key):
                log(f"[FAIL] {z['zip']} 받기 실패 — 승인·키·네트워크 확인")
                fails.append(fk)
                continue
            log(f"[got ] {tar_path.stat().st_size / GiB:.1f} GiB · {(time.time() - t0) / 60:.0f}분")

        t1 = time.time()
        found, written, dup, errors, frames_total = set(), 0, 0, [], 0
        views = zip_views(tar_path)
        with open(index, "a", encoding="utf-8") as idx, Pool(a.workers) as pool:
            for zname, segs in views:
                # 넘겨준 파일 객체는 ZipFile 이 닫지 않는다 — 직접 닫아야 tar 를 지웠을 때 공간이 실제로 풀린다
                raw = io.BufferedReader(Concat(tar_path, segs), 1 << 20)
                with raw, zipfile.ZipFile(raw) as zf:
                    batch = []

                    def flush():
                        nonlocal written, frames_total
                        for stem, sr, ch, n, st in pool.imap_unordered(convert, batch, chunksize=8):
                            if st == "ok":
                                written += 1; frames_total += n
                                idx.write(f"{stem}\t{ds}\t{fk}\t{sr}\t{ch}\t{n}\n")
                            else:
                                errors.append((stem, st))
                        idx.flush()
                        batch.clear()

                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        base = PurePosixPath(member_name(info).replace("\\", "/")).name
                        if not base.lower().endswith(".wav"):
                            continue
                        stem = base[:-4]
                        if stem not in need:
                            continue
                        if stem in found:
                            dup += 1
                            continue
                        found.add(stem)
                        if stem not in todo:
                            continue
                        batch.append((stem, zf.read(info), restored_path(a.out, ds, stem)))   # CRC 검사
                        if len(batch) >= 256:
                            flush()
                    if batch:
                        flush()
        missing = sorted(need - found)
        rec = {"zip": z["zip"], "file_key": fk, "ds": ds, "needed": len(need), "found": len(found), "written": written,
               "missing": len(missing), "missing_sample": missing[:20], "dup_in_zip": dup, "errors": len(errors),
               "error_sample": errors[:10], "hours16k": round(frames_total / SR / 3600, 2),
               "download_min": round((t1 - t0) / 60, 1), "extract_min": round((time.time() - t1) / 60, 1),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        log(f"[done] {z['zip']} 찾음 {len(found):,}/{len(need):,} · 씀 {written:,} · 없음 {len(missing):,} · "
            f"오류 {len(errors)} · {rec['extract_min']}분")
        if errors:
            fails.append(fk)               # 오류가 있으면 끝남 표시를 안 남긴다 → 다시 돌리면 그 파일만 다시 쓴다
            (out / "_done" / f"{fk}.errors.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            done_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        if not a.keep_tar:
            tar_path.unlink()
            ok_mark(tar_path).unlink(missing_ok=True)
            log(f"[rm  ] {tar_path.name}")
    log(f"[end ] 실패 {fails}" if fails else "[end ] 전부 끝")
    return 1 if fails else 0


# ── manifest ────────────────────────────────────────────────────────────

def kspon_pool(src_rows, n_needed, seed):
    """kor001 대체용 KsponSpeech 발화 — train 의 Kspon 묶음(폴더=화자 단위) 안에서, 어느 매니페스트에도 안 쓴 것."""
    used = set()
    bundles = collections.OrderedDict()
    for name, rows in src_rows.items():
        for row in rows:
            used.add(re.sub(r"_aug\d+$", "", stem_of(row["audio_path"])))
            if name == "train.csv" and row["domain"] == "kspon":
                bundles.setdefault(row["speaker"], os.path.dirname(row["audio_path"]))
    cands = {}
    rng = random.Random(seed)
    for spk, d in sorted(bundles.items()):
        fs = sorted(f for f in os.listdir(d) if f.endswith(".wav") and f[:-4] not in used)
        rng.shuffle(fs)
        cands[spk] = [(os.path.join(d, f), spk) for f in fs]
    picked, it = [], {k: iter(v) for k, v in cands.items()}
    while len(picked) < n_needed and it:
        for spk in list(it):
            for path, _ in it[spk]:
                with wave.open(path) as w:
                    sec = w.getnframes() / w.getframerate()
                if sec >= 1.0:                                   # prep 의 min_sec 과 같다
                    picked.append({"audio_path": path, "label_en": "Korea", "speaker": spk, "domain": "kspon",
                                   "task": "KOR", "duration_sec": round(sec, 2)})
                    break
            else:
                del it[spk]
            if len(picked) >= n_needed:
                break
    return picked


def cmd_manifest(a):
    src, dst = Path(a.src), Path(a.dst or (str(a.src).rstrip("/") + "_r16k"))
    gold = load_gold(a.gold)
    src_rows, fields = {}, {}
    for name in MANIFESTS:
        if (src / name).is_file():
            fields[name], src_rows[name] = read_csv(src / name)
    n_kor001 = sum(r["domain"] == "kor001" for r in src_rows.get("train.csv", []))
    repl = iter(kspon_pool(src_rows, n_kor001, a.seed)) if n_kor001 else iter(())
    meta = {"src": str(src), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}, "replaced_train_rows": []}
    built, missing = {}, 0                                # missing = 일부러 뺀 것(사용자 녹음 · kor001 valid·test) 말고 음성이 없는 행
    for name, rows in src_rows.items():
        c, out_rows = collections.Counter(), []
        for i, row in enumerate(rows):
            path, dom = row["audio_path"], row["domain"]
            stem = stem_of(path)
            ds, _ = source_of(path)
            if dom == "user_laptop":                         # 실제 사용자 녹음 — 개인정보라 새 시험지에 넣지 않는다
                c["뺌: 사용자 녹음"] += 1
                continue
            if os.path.isfile(path):
                c["원래 경로"] += 1
            elif dom.startswith("web") and stem in gold:
                row = dict(row, audio_path=gold_path(a.gold, stem)); c["mdd_gold 사본"] += 1
            elif ds and os.path.isfile(restored_path(a.out, ds, stem)):
                row = dict(row, audio_path=restored_path(a.out, ds, stem)); c["복원본"] += 1
            elif dom == "kor001" and name == "train.csv":
                new = next(repl, None)
                if new is None:
                    c["뺌: kor001 대체 부족"] += 1
                    missing += 1
                    continue
                row = new; c["kor001 → Kspon 대체"] += 1
                meta["replaced_train_rows"].append(i)
            elif dom == "kor001":                            # valid·test 는 대체하지 않는다 — 시험지가 달라진다
                c["뺌: kor001 (valid·test)"] += 1
                continue
            else:
                c[f"뺌: 없음({dom})"] += 1
                missing += 1
                continue
            out_rows.append(row)
        built[name] = out_rows
        meta["files"][name] = {"src_rows": len(rows), "rows": len(out_rows), **dict(c)}
        print(f"[{name}] {len(rows):,} → {len(out_rows):,}행 · " + " · ".join(f"{k} {v:,}" for k, v in c.most_common()), flush=True)
    total = sum(len(rows) for rows in src_rows.values())
    meta["missing_rows"] = missing
    if missing > a.max_missing_frac * total:
        raise SystemExit(f"[manifest] 음성이 없는 행 {missing:,}개({missing / total:.2%}) — 허용치 {a.max_missing_frac:.2%} 를 넘어 "
                         f"쓰지 않았다. 복원(run)이 다 끝났는지 확인하거나, 알고서 쓰려면 --max_missing_frac 을 올린다")
    dst.mkdir(parents=True, exist_ok=True)
    for name, out_rows in built.items():
        with open(dst / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields[name])
            w.writeheader(); w.writerows(out_rows)
    for f in ("speaker_split.json", "prep_meta.json"):
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    meta["note"] = ("행 순서는 원본과 같다(대체 행은 같은 자리). teacher_train.npz 는 복사하지 않았다 — "
                    "대체 행과 빠진 행 때문에 행 번호 정렬이 맞지 않는다.")
    (dst / "restore_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[manifest] → {dst}", flush=True)


# ── status ──────────────────────────────────────────────────────────────

def cmd_status(a):
    out = Path(a.out)
    plan = json.loads((out / "_plan.json").read_text(encoding="utf-8"))
    tot_need = tot_found = 0
    for z in plan["zips"]:
        d = out / "_done" / f"{z['file_key']}.json"
        tar = Path(a.stage) / f"{z['file_key']}.tar"
        if d.exists():
            r = json.loads(d.read_text(encoding="utf-8"))
            st = f"끝 — 찾음 {r.get('found', 0):,}/{r.get('needed', 0):,}"
            tot_found += r.get("found", 0)
        elif tar.exists():
            st = f"받는 중 {tar.stat().st_size / GiB:.1f}/{z['gb']} GB"
        else:
            st = "대기"
        tot_need += z["needed"]
        print(f"  {z['ds']} {z['zip']:30s} {z['gb']:>6} GB · {st}")
    print(f"복원 {tot_found:,}/{tot_need:,}개 · 디스크 여유 {shutil.disk_usage(out).free / 1e9:.0f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "manifest", "probe", "status"])
    ap.add_argument("--src", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth", help="원본 매니페스트 폴더")
    ap.add_argument("--dst", default="", help="manifest: 새 매니페스트 폴더 (기본 <src>_r16k)")
    ap.add_argument("--out", default=f"{DATA_ROOT}/nationality_wav16k", help="복원한 16 kHz wav 폴더")
    ap.add_argument("--gold", default=f"{DATA_ROOT}/mdd_gold_wav16k", help="040~043 발음 녹음 16 kHz 사본")
    ap.add_argument("--stage", default=f"{DATA_ROOT}/aihub_staging_nat", help="tar 를 잠시 받는 곳 (/data 쪽)")
    ap.add_argument("--only", default="", help="run: 이 파일키만 (쉼표로 여러 개)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reserve_gb", type=float, default=40, help="run: 받기 전에 남겨 둘 여유")
    ap.add_argument("--keep_tar", action="store_true", help="run: 끝난 tar 를 지우지 않는다")
    ap.add_argument("--seed", type=int, default=42, help="manifest: kor001 대체 발화 고르기")
    ap.add_argument("--max_missing_frac", type=float, default=0.001,
                    help="manifest: 음성이 없는 행이 전체의 이 비율을 넘으면 쓰지 않고 실패")
    a = ap.parse_args()
    cmds = {"plan": cmd_plan, "run": cmd_run, "manifest": cmd_manifest, "probe": cmd_probe, "status": cmd_status}
    return cmds[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
