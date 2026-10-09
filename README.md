# DramaGraph

A continuity checker for AI-generated short videos, built for the Google AI Builder Cup
(Media, Content & Digital Experiences track).

DramaGraph turns a short script into a shot plan, generates each shot with Veo, has Gemini
check every clip against the story's character and prop details, and regenerates only the
shots that fail. It can also check clips made elsewhere.

## Status

Check-only mode works: upload up to four clips, describe what should be true, and get a continuity report. Script planning, Veo generation and repair are next.

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
