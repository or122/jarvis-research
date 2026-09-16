"""Is the petrol engine fuelled?

Checks that a key is present and that it actually works, without ever printing
it. The key is shown only as a masked fingerprint - enough to tell two keys
apart, not enough to use.

The live test is one tiny request. It costs a fraction of a cent, and it is
worth it: a key that is present but wrong fails at the worst moment otherwise.

Run:  .venv/bin/python model/check_key.py
"""
import os
import sys

from hybrid import ENV_FILE, PETROL_MODEL, petrol_available


def mask(key):
    """Show enough to identify a key, never enough to use it."""
    if len(key) < 14:
        return "(too short to be real)"
    return f"{key[:11]}…{key[-4:]}"


def main():
    print("=== PETROL ENGINE CHECK ===\n")

    print(f"looking for .env at: {ENV_FILE}")
    print(f"  file exists:       {'yes' if os.path.exists(ENV_FILE) else 'NO'}")

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

    if key:
        print(f"  ANTHROPIC_API_KEY: {mask(key)}")
    elif token:
        print("  ANTHROPIC_AUTH_TOKEN is set")
    else:
        print("  ANTHROPIC_API_KEY: NOT SET")

    if not petrol_available():
        print("\nFAIL  no key found — the petrol engine cannot run.")
        print("\nTo fix:")
        print("  cd ~/flow")
        print("  cp .env.example .env")
        print("  open -e .env        # paste your key, save, close")
        print("\nGet a key at https://console.anthropic.com/settings/keys")
        sys.exit(1)

    if key and not key.startswith(("sk-ant-", "sk-")):
        print("\nWARN  that does not look like an Anthropic key "
              "(they start with sk-ant-).")

    print(f"\ntesting {PETROL_MODEL} with one tiny request…")
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=PETROL_MODEL,
            max_tokens=32,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": "Reply with exactly: ready"}],
        )
        said = "".join(b.text for b in response.content if b.type == "text").strip()
        used = response.usage
        print(f"  replied: {said!r}")
        print(f"  tokens:  {used.input_tokens} in, {used.output_tokens} out")
        print("\nPASS  the petrol engine is fuelled and working.")
        print("      Restart the server and code questions will go to Claude.")
        return 0

    except anthropic.AuthenticationError:
        print("\nFAIL  the key was rejected. Check it is copied in full.")
    except anthropic.PermissionDeniedError:
        print(f"\nFAIL  this key cannot use {PETROL_MODEL}.")
        print("      Try a different model in .env, e.g. claude-sonnet-5")
    except anthropic.NotFoundError:
        print(f"\nFAIL  no model called {PETROL_MODEL}.")
    except anthropic.RateLimitError:
        print("\nFAIL  rate limited — the key works, but it is busy or out of credit.")
    except anthropic.APIConnectionError:
        print("\nFAIL  could not reach the API. Is the wifi on?")
    except Exception as e:
        print(f"\nFAIL  {type(e).__name__}: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
