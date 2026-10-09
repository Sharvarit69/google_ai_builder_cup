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
   General rules carry over to new episodes.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your own values
pytest                  # tests need no API key
streamlit run app.py
```

## Google technology used

Gemini (planning and video understanding), Veo (video generation), Cloud Run, Cloud Storage.

Never commit `.env` or any API key.
