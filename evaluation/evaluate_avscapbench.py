import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


JUDGE_SYSTEM_PROMPT = """
# Role
You are an Adaptive Evaluator for Omni-modal Video Captions.

# Task
Your goal is to verify if the **Candidate Caption** successfully recalls a checklist of **Ground Truth (GT) Events**.

# Input Understanding: Adaptive Format Support
The Candidate Caption may follow one of two formats. You must evaluate based on the content, regardless of the format:
1.  **Structured Format**: Uses inline tags like `(SFX: ...)` or `(Speech: ...)`.
2.  **Natural Narrative Format**: Uses descriptive sentences (e.g., "A loud crash is heard," "The music starts," "He says 'Hello'").

# Evaluation Rules (By Category)

## 1. Visual Events Evaluation
*   **Target**: The entire caption text.
*   **Criteria**: Semantic Match.
    *   Does the candidate describe the core visual action, object, or scene change mentioned in the GT?
    *   *Note*: Ignore whether the text is inside or outside parentheses. If the visual information is present anywhere, it is a **Hit (1)**.

## 2. Pure Audio Events Evaluation
*   **Target**: Look for **explicit mentions of auditory perception**.
*   **Criteria**: The candidate must acknowledge the *sound* itself, not just the visual source.
*   **Acceptable Evidence**:
    *   **Explicit Tags**: `(SFX: ...)`, `(Speech: ...)`, `(Music: ...)`.
    *   **Auditory Verbs/Nouns**: "heard", "sound", "noise", "voice", "music", "audio", "scream", "thud", "click".
    *   **Speech Transcription**: If the candidate quotes dialogue (e.g., "He says 'Stop!'"), this counts as capturing the Audio Event (Speech).
    *   **Adjectives of Sound**: "Loud", "Quiet", "High-pitched", "Rhythmic" (when applied to an event).
*   **Differentiation**:
    *   "A dog barks" (Acceptable - implies sound).
    *   "A dog opens its mouth" (Miss - purely visual).
    *   "An explosion" (Borderline - Miss unless "loud" or "sound" is mentioned).
    *   "A loud explosion" (Hit - auditory attribute).

## 3. Synergistic Events Evaluation (CRITICAL)
*   **Definition**: These events represent the **synchronization** or **causal link** between a Visual Trigger and an Audio Response.
*   **Criteria: Semantic Linkage**.
    *   Does the text explicitly connect the sound to the visual event?
*   **Acceptable Connections**:
    1.  **Syntax (Structured)**: The Audio Tag follows immediately after the Visual Trigger sentence.
        *   *Ex:* "The car hits the wall (SFX: Crash)." -> **Hit**
    2.  **Narrative (Natural)**: The text uses connectors to show simultaneity or cause.
        *   *Ex:* "The car hits the wall **with a** loud crash." -> **Hit**
        *   *Ex:* "**As** he falls, he screams." -> **Hit**
        *   *Ex:* "The music starts **when** the scene changes." -> **Hit**
*   **Failure Cases (Miss)**:
    *   "The car hits the wall. Later, a crash is heard." (Wrong timing).
    *   "There is a car. There is a crash sound." (No linkage described).

# Output Format (JSON Only)
Return a JSON object with three arrays of 1s and 0s. The length of each array MUST match the number of input GT events for that category.

{
  "visual_hits": [1, 0, ...],
  "audio_hits": [1, 1, ...],
  "synergy_hits": [0, 0, ...]
}
"""


def load_gt_events(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["video_id"]).replace(".mp4", ""): item["event"] for item in data}


def load_captions(path: Path) -> dict[str, str]:
    captions = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            captions[str(obj["video_id"]).replace(".mp4", "")] = obj["output"]
    return captions


def safe_mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def get_video_duration(video_id: str, videos_dir: Path | None) -> float:
    if videos_dir is None:
        return 0.0
    video_path = videos_dir / f"{video_id}.mp4"
    if not video_path.exists():
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def build_client():
    from google import genai
    from google.genai import types

    api_key = os.environ.get("JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("Set JUDGE_API_KEY before running --run-evals.")

    base_url = os.environ.get("JUDGE_BASE_URL")
    http_options = None
    if base_url:
        http_options = types.HttpOptions(base_url=base_url, api_version="v1", timeout=120000)
    return genai.Client(api_key=api_key, http_options=http_options), types


def evaluate_single(client, types, judge_model: str, video_id: str, caption: str, gt_event: dict, duration: float) -> dict | None:
    visual_events = gt_event.get("visual_events", [])
    audio_events = gt_event.get("audio_events", {"speech": [], "music": [], "sfx": []})
    if isinstance(audio_events, list):
        audio_events = {"speech": audio_events, "music": [], "sfx": []}

    speech_events = audio_events.get("speech", [])
    music_events = audio_events.get("music", [])
    sfx_events = audio_events.get("sfx", [])
    flat_audio = speech_events + music_events + sfx_events

    prompt = f"""--- CANDIDATE CAPTION ---
{caption}

--- GROUND TRUTH CHECKLIST ---
Visual Events: {json.dumps(visual_events, ensure_ascii=False)}
Pure Audio Events: {json.dumps(flat_audio, ensure_ascii=False)}
Synergistic Events: {json.dumps(gt_event.get("synergistic_events", []), ensure_ascii=False)}

Please evaluate item by item and return the hit arrays.
"""
    response = client.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    result = json.loads(response.text)

    def pad(values, total):
        return (values + [0] * total)[:total]

    visual_hits = pad(result.get("visual_hits", []), len(visual_events))
    audio_hits = pad(result.get("audio_hits", []), len(flat_audio))
    synergy_hits = pad(result.get("synergy_hits", []), len(gt_event.get("synergistic_events", [])))

    speech_n = len(speech_events)
    music_n = len(music_events)
    speech_hits = audio_hits[:speech_n]
    music_hits = audio_hits[speech_n:speech_n + music_n]
    sfx_hits = audio_hits[speech_n + music_n:]

    total_hits = sum(visual_hits) + sum(audio_hits) + sum(synergy_hits)
    total_events = len(visual_hits) + len(audio_hits) + len(synergy_hits)
    return {
        "video_id": video_id,
        "duration": duration,
        "visual_recall": safe_mean(visual_hits),
        "audio_speech_recall": safe_mean(speech_hits),
        "audio_music_recall": safe_mean(music_hits),
        "audio_sfx_recall": safe_mean(sfx_hits),
        "audio_recall": safe_mean(audio_hits),
        "synergy_recall": safe_mean(synergy_hits),
        "total_recall": total_hits / total_events if total_events else 0.0,
        "hits_detail": {
            "visual_hits": visual_hits,
            "audio_speech_hits": speech_hits,
            "audio_music_hits": music_hits,
            "audio_sfx_hits": sfx_hits,
            "synergy_hits": synergy_hits,
        },
    }


def summarize(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.jsonl")):
        results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not results:
            continue
        n = len(results)
        avg = lambda key: sum(r.get(key, 0.0) for r in results) / n
        rows.append({
            "model": path.stem,
            "videos": n,
            "visual": avg("visual_recall"),
            "audio": avg("audio_recall"),
            "speech": avg("audio_speech_recall"),
            "music": avg("audio_music_recall"),
            "sfx": avg("audio_sfx_recall"),
            "synergy": avg("synergy_recall"),
            "total": avg("total_recall"),
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8") as f:
        f.write("model,videos,visual,audio,speech,music,sfx,synergy,total\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['videos']},{r['visual']:.4f},{r['audio']:.4f},"
                f"{r['speech']:.4f},{r['music']:.4f},{r['sfx']:.4f},{r['synergy']:.4f},{r['total']:.4f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captions on AVSCapBench.")
    parser.add_argument("--gt", required=True, type=Path, help="Path to OmniCaption.json")
    parser.add_argument("--captions-dir", default=Path("model_captions"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/eval"), type=Path)
    parser.add_argument("--videos-dir", default=None, type=Path)
    parser.add_argument("--models", default="all", help='Comma-separated model names, or "all".')
    parser.add_argument("--run-evals", action="store_true", help="Call the judge model for missing videos.")
    parser.add_argument("--max-workers", default=8, type=int)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gemini-3.1-pro"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_map = load_gt_events(args.gt)
    model_names = sorted(p.stem for p in args.captions_dir.glob("*.jsonl")) if args.models == "all" else [m.strip() for m in args.models.split(",")]

    client = types = None
    if args.run_evals:
        client, types = build_client()

    for model_name in model_names:
        captions_path = args.captions_dir / f"{model_name}.jsonl"
        if not captions_path.exists():
            print(f"skip missing captions: {captions_path}")
            continue

        output_path = args.output_dir / f"{model_name}.jsonl"
        existing = {}
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    existing[str(obj["video_id"]).replace(".mp4", "")] = obj
        elif not args.run_evals:
            print(f"{model_name}: no released eval file found; use --run-evals to create one")
            continue

        captions = load_captions(captions_path)
        missing_ids = [vid for vid in sorted(captions, key=lambda x: int(x) if x.isdigit() else x) if vid in gt_map and vid not in existing]

        if args.run_evals and missing_ids:
            def task(vid):
                return evaluate_single(
                    client,
                    types,
                    args.judge_model,
                    vid,
                    captions[vid],
                    gt_map[vid],
                    get_video_duration(vid, args.videos_dir),
                )

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(task, vid) for vid in missing_ids]
                for future in tqdm(futures, desc=model_name, unit="video"):
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"eval failed: {exc}")
                        continue
                    if result:
                        existing[str(result["video_id"]).replace(".mp4", "")] = result

        results = sorted(existing.values(), key=lambda x: int(str(x["video_id"]).replace(".mp4", "")) if str(x["video_id"]).replace(".mp4", "").isdigit() else str(x["video_id"]))
        with output_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{model_name}: {len(results)} evaluated videos")

    summarize(args.output_dir)


if __name__ == "__main__":
    main()
