"""
Helper script: Run locally to get your Google OAuth refresh token.

Usage:
  1. Set your client_id and client_secret below (or in environment variables)
  2. Run: python scripts/get_refresh_token.py
  3. Complete the Google auth flow in your browser
  4. Copy the printed refresh_token into .streamlit/secrets.toml

This only needs to be run once. The refresh token does not expire unless
you explicitly revoke it from your Google account settings.
"""

import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

# ── Configuration ─────────────────────────────────────────────────────────────
# Either set these values directly or use environment variables:
#   GSC_CLIENT_ID, GSC_CLIENT_SECRET

CLIENT_ID = os.environ.get("GSC_CLIENT_ID", "YOUR_CLIENT_ID.apps.googleusercontent.com")
CLIENT_SECRET = os.environ.get("GSC_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

# ─────────────────────────────────────────────────────────────────────────────

if "YOUR_CLIENT" in CLIENT_ID or "YOUR_CLIENT" in CLIENT_SECRET:
    print("ERROR: Please set your CLIENT_ID and CLIENT_SECRET before running this script.")
    print("Edit scripts/get_refresh_token.py or set environment variables:")
    print("  set GSC_CLIENT_ID=your_client_id")
    print("  set GSC_CLIENT_SECRET=your_client_secret")
    exit(1)

client_config = {
    "installed": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
    }
}

flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
creds = flow.run_local_server(port=0)

print("\n" + "=" * 60)
print("SUCCESS! Copy these values into your .streamlit/secrets.toml:")
print("=" * 60)
print(f"\n[gsc]")
print(f'client_id = "{CLIENT_ID}"')
print(f'client_secret = "{CLIENT_SECRET}"')
print(f'refresh_token = "{creds.refresh_token}"')
print("\n" + "=" * 60)
print("\nFull token info (for reference):")
print(json.dumps({
    "access_token": creds.token,
    "refresh_token": creds.refresh_token,
    "token_uri": creds.token_uri,
    "scopes": list(creds.scopes) if creds.scopes else SCOPES,
}, indent=2))
