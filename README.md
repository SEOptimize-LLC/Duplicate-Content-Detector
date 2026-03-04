# 🔎 Duplicate Content Detector

A Streamlit-based SEO tool that identifies and prioritizes duplicate and cannibalized content across your website using three complementary detection layers.

## What It Does

### Layer 1 — Semantic Similarity (Screaming Frog Embeddings)
Detects pages that "sound the same" even when exact wording differs. Upload your Screaming Frog embeddings CSV and the app computes pairwise cosine similarity across all your pages, grouping them into duplicate clusters.

**Similarity Score Guide:**
| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| > 0.85 | Likely duplicate | Consolidate or redirect |
| 0.60 – 0.85 | Overlapping intent/topic | Review and differentiate |
| < 0.60 | Distinct content | Usually fine |

### Layer 2 — Keyword Cannibalization (Google Search Console)
Connects to your GSC account and finds queries where multiple pages are competing for clicks and impressions. Ranked by a composite **Impact Score** = `(total_clicks + log(impressions)) × competing_url_count`.

### Layer 3 — AI Recommendations (OpenRouter)
For your highest-risk URL pairs, get specific remediation advice from GPT-5.1, Claude Sonnet 4.6, or Gemini Flash — including which page to keep, which to redirect, and how to differentiate content you're keeping both of.

---

## Prerequisites

1. **Python 3.9+**
2. **Screaming Frog SEO Spider v20+** (for embeddings export)
3. **Google Cloud Project** with Search Console API enabled (for GSC integration)
4. **OpenRouter account** with API key (for AI recommendations)

---

## Installation & Local Setup

### 1. Clone / download this project

```bash
cd "Duplicate Content Detector"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

Copy `.streamlit/secrets.toml` and fill in your credentials (see **Configuration** below).

### 4. Run locally

```bash
streamlit run app.py
```

---

## Configuration

Edit `.streamlit/secrets.toml`:

```toml
[gsc]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
refresh_token = "YOUR_REFRESH_TOKEN"   # Optional but recommended

OPENROUTER_API_KEY = "sk-or-YOUR_KEY"
```

---

## Google Search Console Setup

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project**
3. Name it (e.g., "Duplicate Content Detector") and click **Create**

### Step 2: Enable the Search Console API

1. In your project, go to **APIs & Services → Library**
2. Search for **"Google Search Console API"**
3. Click on it → **Enable**

### Step 3: Create OAuth 2.0 Credentials

1. Go to **APIs & Services → Credentials**
2. Click **+ Create Credentials → OAuth client ID**
3. If prompted, configure the OAuth consent screen:
   - User type: **External**
   - App name: anything (e.g., "Duplicate Content Detector")
   - Add your email as a test user
4. Back in Create OAuth client ID:
   - Application type: **Web application** (for Streamlit Cloud)
   - Name: anything
   - Authorized redirect URIs: add your Streamlit Cloud app URL + `/Data_Input`
     - For local dev: `http://localhost:8501/Data_Input`
     - For Streamlit Cloud: `https://your-app-name.streamlit.app/Data_Input`
5. Click **Create** → copy your **Client ID** and **Client Secret**

### Step 4: Get Your Refresh Token (Recommended for Streamlit Cloud)

Run this helper script locally once to get a persistent refresh token:

```bash
# Set your credentials
set GSC_CLIENT_ID=your_client_id
set GSC_CLIENT_SECRET=your_client_secret

# Run the helper
python scripts/get_refresh_token.py
```

A browser window will open for Google authentication. After completing the flow, the script prints your `refresh_token` — copy it into `secrets.toml`.

With a stored refresh token, the app authenticates automatically without user interaction on each Streamlit Cloud session.

---

## Screaming Frog Embeddings Export

### How to export embeddings:

1. Open Screaming Frog SEO Spider (v20 or later required)
2. Run a crawl of your website
3. After the crawl completes, go to:
   **Bulk Export → Content → Embeddings**
4. Save the CSV file to your computer

### What the export contains:

- **Column 1:** `Address` — the page URL
- **Columns 2–385:** Numeric values — the 384-dimensional embedding vector for each page

This file can be large (one row per URL, 385 columns). The app handles files with 500+ URLs on Streamlit free tier.

> **Note:** Screaming Frog generates embeddings using the **all-MiniLM-L6-v2** model by default, which produces 384-dimensional vectors. The app auto-detects the number of dimensions from your file.

---

## Deploying to Streamlit Community Cloud (Free)

### 1. Push to GitHub

Create a new private GitHub repository and push this project:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/duplicate-content-detector
git push -u origin main
```

**Important:** The `.streamlit/secrets.toml` file contains credentials. Add it to `.gitignore`:

```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **New app**
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Click **Advanced settings → Secrets**
6. Paste your `secrets.toml` contents into the secrets field
7. Click **Deploy**

### 3. Update OAuth Redirect URI

After deployment, note your app URL (e.g., `https://your-app.streamlit.app`).

Go back to Google Cloud Console → Credentials → your OAuth client → add:
```
https://your-app.streamlit.app/Data_Input
```
as an authorized redirect URI.

---

## Project Structure

```
Duplicate Content Detector/
├── app.py                          # Home page + navigation
├── pages/
│   ├── 1_Data_Input.py             # SF upload + GSC auth + URL filter
│   ├── 2_Semantic_Similarity.py    # Cosine similarity + clusters + heatmap
│   ├── 3_GSC_Cannibalization.py    # Query overlap analysis
│   ├── 4_Combined_Risk.py          # Cross-referenced risk dashboard
│   └── 5_AI_Recommendations.py    # OpenRouter AI analysis
├── utils/
│   ├── gsc_handler.py              # GSC OAuth + Search Analytics API
│   ├── embeddings_handler.py       # SF CSV parser + cosine similarity
│   ├── clustering.py               # URL cluster detection
│   └── openrouter_handler.py       # AI model integration
├── scripts/
│   └── get_refresh_token.py        # One-time OAuth token helper
├── .streamlit/
│   └── secrets.toml                # Credentials (never commit this)
├── requirements.txt
└── README.md
```

---

## Similarity Thresholds Explained

The app uses **cosine similarity** on Screaming Frog's embedding vectors. This measures how "directionally similar" two pieces of content are in semantic space:

- **1.0** = identical meaning
- **0.85–1.0** = likely duplicates (same topic, angle, and depth)
- **0.60–0.85** = overlapping intent (similar topic, possibly different angles)
- **0.40–0.60** = borderline (shared terminology, different focus)
- **< 0.40** = clearly distinct content

These thresholds are configurable in the app interface.

---

## Troubleshooting

### "Could not detect URL column" on SF upload
Ensure your Screaming Frog CSV has a column named `Address`, `URL`, or `Page`. The bulk embeddings export uses `Address` by default.

### GSC OAuth "redirect_uri_mismatch" error
The redirect URI in your Google Cloud credentials must exactly match the one your app uses. Update it in Google Cloud Console → Credentials → your OAuth client.

### "No data returned" from GSC
- Verify the property URL exactly matches what's in GSC (including `https://` vs `http://` and trailing slashes)
- Try a shorter date range first
- Ensure your Google account has access to that property

### App crashes with large URL sets
The pairwise similarity matrix for N URLs requires O(N²) memory. For > 300 URLs, consider using the URL filter (Data Input page, Section C) to analyze a subset.

---

## OpenRouter Models

| Model | Use Case | Speed | Cost |
|-------|----------|-------|------|
| `openai/gpt-5.1` | Best reasoning, complex analysis | Medium | Higher |
| `anthropic/claude-sonnet-4-6` | Excellent SEO advice, nuanced | Medium | Higher |
| `google/gemini-3.1-flash-lite-preview` | Fast, cost-effective, bulk use | Fast | Lower |

Get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

---

## License

MIT — free to use, modify, and deploy for personal or commercial SEO work.
