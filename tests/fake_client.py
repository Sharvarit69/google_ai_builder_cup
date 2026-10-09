"""A stand-in for the Gemini client so tests run the real code with no API key.

Fake Veo returns the prompt text as the "video". The fake critic reads that text
to decide whether the box is open in the clip, compares that with what the critic
prompt expects, and reports a wardrobe mismatch if the clip mentions "red kurta".
"""
import json
import re
from types import SimpleNamespace as NS


class FakeClient:
    def __init__(self):
        self.models = self
        self.operations = self
        self.files = self
        self.veo_prompts = []
        self.veo_errors = []      # exceptions to raise on the next generate_videos calls
        self.refuse_next = 0      # number of upcoming generations to refuse

    # --- text and video understanding ---
    def generate_content(self, model, contents, config=None):
        prompt = contents[-1]
        media = contents[:-1]
        if "Split this short video script" in prompt:
            lines = re.findall(r"^\d+\. (.+)$", prompt, re.M)
            beats = [{"beat_id": f"B{i+1}", "script_lines": [i + 1], "action": line,
                      "caption": line, "props_needed": ["lunch box"],
                      "needs_second_person": "colleague stands" in line}
                     for i, line in enumerate(lines)]
            return NS(text=json.dumps({"beats": beats}))
        if "fixed visual details" in prompt:
            n = len(re.findall(r"^Shot \d+:", prompt, re.M))
            return NS(text=json.dumps({
                "name": "Priya", "appearance": "black hair in a low ponytail",
                "wardrobe": "plain mustard-yellow kurta", "accessories": [],
                "props": [{"name": "steel lunch box", "description": "round",
                           "states": ["closed"] * (n - 1) + ["open and empty"]}],
                "location": "office desk", "time_of_day": "daytime"}))
        if "Plan one video shot" in prompt:
            n = len(re.findall(r"^Beat \d+:", prompt, re.M))
            return NS(text=json.dumps({"shots": [
                {"shot_type": "medium shot", "camera_motion": "static",
                 "action": f"action {i+1}", "second_person_hands": False} for i in range(n)]}))
        if "Turn their" in prompt:      # learning a rule from a creator's note
            note = prompt.split("THE CREATOR'S NOTE:")[1].split("Return only JSON")[0].strip()
            general = any(k in note.lower() for k in ("appear", "pops", "vanish", "always"))
            text = ("No object appears or vanishes without someone causing it."
                    if "appear" in note.lower() or "pops" in note.lower() else note[0].upper() + note[1:])
            return NS(text=json.dumps({"text": text, "applies_to_all_shots": general}))
        if "continuity checker" in prompt:
            clip = media[-1].data
            clip_open = any(k in clip for k in (b"lid off", b"it is open", b"it is: open"))
            bad_prop = clip_open != ("must be: OPEN" in prompt)
            bad_wardrobe = b"red kurta" in clip
            has_prop = "no specific prop required" not in prompt
            bad_look = has_prop and b"square" in clip and "square" not in prompt.split("First describe")[0]
            check = lambda bad, seen: {"verdict": "mismatch" if bad else "match", "observed": seen}
            violations = []
            if bad_prop:
                violations.append({"type": "PROP_STATE", "timestamp_seconds": 2.5,
                                   "expected": "closed", "observed": "open with the lid off"})
            if bad_wardrobe:
                violations.append({"type": "WARDROBE", "timestamp_seconds": 0,
                                   "expected": "yellow kurta", "observed": "red kurta"})
            rules = []
            if "RULES THE CREATOR ADDED" in prompt:
                block = prompt.split("RULES THE CREATOR ADDED")[1].split("For every mismatch")[0]
                for number, text in re.findall(r"^(\d+)\. (.+)$", block, re.M):
                    broken = b"glitch" in clip and "appears" in text
                    rules.append({"number": int(number), "verdict": "mismatch" if broken else "match",
                                  "observed": "a note popped into view" if broken else "holds"})
            observations = ([
                {"what": "The desk lamp flickers once", "timestamp_seconds": 1.0, "severity": "minor",
                 "suggested_fix": "The lamp gives steady light."},
                {"what": "A sixth finger appears on her left hand", "timestamp_seconds": 3.0,
                 "severity": "major", "suggested_fix": "Both hands have five fingers throughout."},
            ] if b"extra finger" in clip else [])
            return NS(text=json.dumps({
                "rules": rules, "observations": observations,
                "seen": "a woman at a desk",
                "wardrobe": check(bad_wardrobe, "red kurta" if bad_wardrobe else "yellow kurta"),
                "prop_look": check(bad_look, "square box" if bad_look else "as described") if has_prop else None,
                "prop_state": check(bad_prop, "open" if bad_prop else "as expected"),
                "scene": check(False, "one person, desk, day"),
                "cross_shot": check(False, "same person") if len(media) == 2 else None,
                "violations": violations, "decision": "ACCEPT"}))
        raise AssertionError("unexpected prompt: " + prompt[:80])

    # --- Veo ---
    def generate_videos(self, model, prompt, config=None):
        if self.veo_errors:
            raise self.veo_errors.pop(0)
        self.veo_prompts.append((prompt, bool(getattr(config, "reference_images", None))))
        refused = self.refuse_next > 0
        self.refuse_next -= refused
        return NS(done=False, error=None, _prompt=prompt, _refused=refused)

    def get(self, operation=None, name=None):
        if operation is None:  # files.get(name=...)
            return NS(state=NS(name="ACTIVE"), name=name)
        videos = [] if operation._refused else [NS(video=NS(video_bytes=None, _p=operation._prompt))]
        return NS(done=True, error=None,
                  response=NS(generated_videos=videos, rai_media_filtered_reasons=["safety"]))

    def download(self, file):
        return file._p.encode("utf-8")

    def upload(self, file):
        with open(file, "rb") as f:
            return NS(state=NS(name="ACTIVE"), name=file, data=f.read())
