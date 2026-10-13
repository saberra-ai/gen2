"""Fetch a bounded, revision-pinned FLEURS test subset (CC-BY-4.0).

Requires numpy and soundfile; use the locked Bark reference environment.
Archives are streamed and closed after the selected recordings, not retained.
"""
import concurrent.futures
import hashlib
import io
import json
import pathlib
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
import soxr

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
ROOT = pathlib.Path("target/whistle/corpus")
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"
LANGUAGES = {"en": "en_us", "de": "de_de", "fr": "fr_fr", "es": "es_419",
             "it": "it_it", "nl": "nl_nl", "pl": "pl_pl"}


def fetch(item):
    language, config = item
    dest = ROOT / language
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / "corpus.json"
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{BASE}/{config}/test.tsv", timeout=120) as response:
        rows = {r.split("\t")[1]: r.split("\t") for r in response.read().decode().splitlines()}
    records = []
    with urllib.request.urlopen(f"{BASE}/{config}/audio/test.tar.gz", timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                name = pathlib.PurePosixPath(member.name).name
                if not member.isfile() or name not in rows:
                    continue
                raw = archive.extractfile(member).read()
                audio, rate = sf.read(io.BytesIO(raw), dtype="float32")
                if rate != 16000 or audio.ndim != 1:
                    raise ValueError(f"unexpected audio format: {name}")
                if not 640 <= len(audio) <= 960000:
                    continue
                identifier = f"{language}-{name[:-4]}"
                pcm = np.asarray(audio, dtype="<f4").tobytes()
                (ROOT / f"{identifier}.f32").write_bytes(pcm)
                records.append({"id": identifier, "pcm": f"{identifier}.f32",
                    "language": language, "human_text": rows[name][2],
                    "source_audio_sha256": hashlib.sha256(raw).hexdigest(),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "samples": len(audio),
                    "slice": "parity" if len(records) < 10 else "held-out-quality"})
                if len(records) == 20:
                    break
    if len(records) != 20:
        raise ValueError(f"insufficient recordings for {language}")
    receipt.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(language, len(records), flush=True)
    return records


def extend_coverage(records):
    critical = {
        "en-1003119935936341070": ["numbers"],
        "en-10167324587744183095": ["names"],
        "en-10233995782544396174": ["negation"],
        "en-1038214857203833067": ["numbers", "names"],
        "en-10405958125670257178": ["negation"],
    }
    for record in records:
        record["critical_content"] = critical.get(record["id"], [])
    extra = []
    for i, language in enumerate(LANGUAGES):
        record = next(r for r in records if r["language"] == language and r["slice"] == "held-out-quality")
        pcm = np.fromfile(ROOT / record["pcm"], dtype="<f4")
        rng = np.random.default_rng(1300 + i)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        for label, audio in [("quiet-40dB", pcm * np.float32(.01)),
                             ("noisy-10dB", pcm + rng.normal(0, rms / np.sqrt(10), len(pcm)).astype(np.float32))]:
            name = f'{record["id"]}-{label}'
            raw = audio.astype("<f4").tobytes()
            (ROOT / f"{name}.f32").write_bytes(raw)
            extra.append({**record, "id":name, "pcm":f"{name}.f32", "pcm_sha256":hashlib.sha256(raw).hexdigest(),
                "slice":label, "derived_from":record["id"], "transformation":label, "seed":1300+i})
    for record in [records[0], next(r for r in records if r["language"] == "es")]:
        pcm = np.fromfile(ROOT / record["pcm"], dtype="<f4")
        for rate in [8000, 16000, 44100, 48000]:
            mono = pcm if rate == 16000 else soxr.resample(pcm,16000,rate,quality="HQ")
            for channels in [1,2,8]:
                for subtype in ["PCM_16","FLOAT"]:
                    name = f'{record["id"]}-wav-{rate}-{channels}-{subtype}'
                    wave = np.repeat(mono[:,None],channels,axis=1)
                    wav = ROOT / f"{name}.wav"
                    sf.write(wav,wave,rate,subtype=subtype)
                    decoded,_ = sf.read(wav,dtype="float32",always_2d=True)
                    normalized = decoded.mean(axis=1)
                    if rate != 16000: normalized = soxr.resample(normalized,rate,16000,quality="HQ")
                    raw = normalized.astype("<f4").tobytes()
                    (ROOT / f"{name}.f32").write_bytes(raw)
                    extra.append({**record, "id":name,"pcm":f"{name}.f32","pcm_sha256":hashlib.sha256(raw).hexdigest(),
                        "samples":len(normalized),"slice":"wav-conformance","wav":wav.name,"rate":rate,"channels":channels,
                        "encoding":subtype,"wav_sha256":hashlib.sha256(wav.read_bytes()).hexdigest(),"derived_from":record["id"]})
    return records + extra


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records = [r for subset in executor.map(fetch, LANGUAGES.items()) for r in subset]
    records = extend_coverage(records)
    (ROOT / "corpus.json").write_text(json.dumps({"dataset": "google/fleurs",
        "revision": REVISION, "split": "test", "license": "CC-BY-4.0",
        "selection": "first 20 supported recordings in each pinned test archive",
        "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
