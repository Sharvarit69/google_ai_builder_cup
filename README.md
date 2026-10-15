# DramaGraph

A continuity checker for AI-generated short videos, built for the Google AI Builder Cup
(Media, Content & Digital Experiences track).

DramaGraph turns a short script into a shot plan, generates each shot with Veo, has Gemini
check every clip against the story's character and prop details, and regenerates only the
shots that fail. It can also check clips made elsewhere.

## Status

Working: script to story beats, character and prop details, shot plan, Veo generation with a spending cap, continuity checks, automatic repair of failed shots, and a check-only mode for clips made elsewhere. Accepted shots are stitched into one vertical video with a title screen and captions.

## How the critic decides

Three layers, from most to least fixed:

1. **Fixed checks** for every clip: wardrobe, prop appearance, prop state, scene, and a
   comparison with the previous shot. A mismatch regenerates the shot.
2. **Open observations**: the critic also watches like a script supervisor and reports
   anything else a viewer would find odd. These are advisory and never spend video budget.
3. **Rules learned from the creator**: when a person spots a problem the critic missed, or
   promotes an observation, their note becomes a rule the critic enforces from then on.
   Rules stay in their own episode. A rule that applies to every shot can be kept on
   purpose (☆ Keep); kept rules are offered to new episodes but never added without a click.
   Rules about small details (a clock time, writing, a number) get a **close look**: sharp
   still frames are taken from the clip, Gemini finds the object, the app enlarges it, and
   Gemini reads it step by step. An unsure reading is "unclear", never a guess, and a wrong
   reading goes to the creator for review instead of an automatic remake, because video
   models often cannot draw such details exactly.

## Models

| Job | Model | Setting |
|---|---|---|
| Critic (watches each clip) | Gemini `gemini-3.8-flash` | `VIDEO_UNDERSTANDING_MODEL` |
| Planning (beats, details, shots, rules) | same as the critic unless set | `TEXT_MODEL` |
| Video generation | Veo `veo-3.1-fast-generate-preview`, 8-second 9:16 clips at 720p | `VEO_MODEL` |

Other settings are listed in `.env.example`. `CRITIC_FPS` sets how many frames a second the
critic looks at (Gemini's default is about 1).

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your own values
pytest                  # tests need no API key; Gemini and Veo are replaced by a fake
streamlit run app.py
```

Without `GCS_BUCKET`, episodes are saved in `local_data/`. Without `GEN_PASSCODE`, video
generation stays switched off; check-only mode still works.

## Deploy to Cloud Run

The API key and the passcode live in Secret Manager, never in the repository.

```bash
gcloud secrets create gemini-key --data-file=-      # paste the key, then Ctrl+D
gcloud secrets create gen-passcode --data-file=-    # a made-up passcode, not the key
gcloud storage buckets create gs://YOUR_BUCKET --location=asia-southeast1

gcloud run deploy dramagraph --source . --region asia-southeast1 \
  --set-secrets=GEMINI_API_KEY=gemini-key:latest,GEN_PASSCODE=gen-passcode:latest \
  --set-env-vars=GCS_BUCKET=YOUR_BUCKET,VIDEO_UNDERSTANDING_MODEL=gemini-3.8-flash,VEO_MODEL=veo-3.1-fast-generate-preview
```

The Cloud Run service account needs *Secret Manager Secret Accessor* and access to the
bucket. Later code changes only need `gcloud run deploy dramagraph --source . --region asia-southeast1`;
settings can be changed without a rebuild with `gcloud run services update dramagraph
--region asia-southeast1 --update-env-vars=NAME=value`.

## Measure the critic

```bash
python -m eval.run_eval eval/labels.csv path/to/clips 3
```

`labels.csv` holds one hand-labelled row per clip (see `eval/run_eval.py` for the columns).
The last number is how many times each clip is checked, to measure consistency.

## Future scope

- A larger evaluation set (25–30 clips, including subtle wardrobe changes and clips from
  other video tools), scored by the second each error appears, with "caught", "sent to
  review" and "missed" reported separately.
- Product continuity for ads: the same product, logo and packaging across every clip.
- Rule libraries per creator or team, once the app has sign-in.

## Google technology used

Gemini (planning and video understanding), Veo (video generation), Cloud Run, Cloud Storage.

Never commit `.env` or any API key.
