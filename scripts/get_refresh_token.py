"""
One-time helper: get your Google OAuth refresh token.

Run this locally once to generate the refresh_token you need for Streamlit secrets.

Usage:
    python scripts/get_refresh_token.py

The script will:
  1. Ask for your Client ID and Client Secret (from Google Cloud)
  2. Open a browser tab for you to sign in with your Google account
  3. Print the refresh_token to paste into Streamlit secrets

The refresh_token does not expire unless you explicitly revoke it.
You only need to run this script once.

IMPORTANT — credential type:
  Use "Desktop App" in Google Cloud Console (not "Web application").
  If you used "Web application", add http://localhost:8085/ to its
  Authorized Redirect URIs in Google Cloud → APIs & Services → Credentials.
"""

import os

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
REDIRECT_PORT = 8085


def main():
    print("=" * 60)
    print("  Google Search Console — Refresh Token Generator")
    print("=" * 60)
    print()
    print("You need your OAuth 2.0 Client ID and Client Secret from")
    print("Google Cloud Console (APIs & Services → Credentials).")
    print()
    print("Credential type should be: Desktop App")
    print("If you used 'Web application', add this redirect URI in")
    print("Google Cloud Console → Authorized Redirect URIs:")
    print(f"  http://localhost:{REDIRECT_PORT}/")
    print()

    # Read from env vars first, fall back to interactive prompt
    client_id = os.environ.get("GSC_CLIENT_ID", "").strip()
    if not client_id or "YOUR_CLIENT" in client_id:
        client_id = input("Client ID: ").strip()

    client_secret = os.environ.get("GSC_CLIENT_SECRET", "").strip()
    if not client_secret or "YOUR_CLIENT" in client_secret:
        client_secret = input("Client Secret: ").strip()

    if not client_id or not client_secret:
        print("\nERROR: Client ID and Client Secret are required.")
        return

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                f"http://localhost:{REDIRECT_PORT}/",
                "http://localhost",
            ],
        }
    }

    print()
    print("Opening browser for Google sign-in…")
    print("Sign in with the Google account that has Search Console access.")
    print()

    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    creds = flow.run_local_server(port=REDIRECT_PORT)

    print()
    print("=" * 60)
    print("  SUCCESS!  Paste the block below into Streamlit Secrets")
    print("=" * 60)
    print()
    print("Go to: Streamlit Cloud → your app → Settings → Secrets")
    print("Replace everything with the block below (fill in your OpenRouter key):")
    print()
    print("─" * 60)
    print('OPENROUTER_API_KEY = "sk-or-YOUR_OPENROUTER_KEY"')
    print()
    print("[gsc]")
    print(f'client_id = "{client_id}"')
    print(f'client_secret = "{client_secret}"')
    print(f'refresh_token = "{creds.refresh_token}"')
    print("─" * 60)
    print()
    print("NOTE: OPENROUTER_API_KEY must be ABOVE [gsc] in the secrets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
