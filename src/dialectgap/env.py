"""Hugging Face token setup for Kaggle/Colab notebooks."""
import os

# Kaggle secret labels to try, in order. The secret's VALUE must be a Hugging Face token (starts with "hf_").
SECRET_NAMES = ("Kaggle_Token", "HF_TOKEN")


def load_hf_token(check_repos=("google/gemma-3-4b-it",)) -> str | None:
    """Read the HF token from Kaggle Secrets (or the environment), export it, and verify it.

    Returns the HF username, or None if no token was found. Raises if a token is found but invalid,
    or if it cannot access one of `check_repos` (e.g. gated-model terms not accepted).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
        except ImportError:
            client = None
        for name in SECRET_NAMES if client else ():
            try:
                token = client.get_secret(name)
                print(f"Loaded Kaggle secret '{name}'")
                break
            except Exception:
                continue
    if not token:
        print(f"No HF token found (tried Kaggle secrets {SECRET_NAMES} and $HF_TOKEN). Gated models will fail.")
        return None
    token = token.strip()
    if not token.startswith("hf_"):
        raise ValueError("The secret does not look like a Hugging Face token (expected it to start with 'hf_').")
    os.environ["HF_TOKEN"] = token

    from huggingface_hub import model_info, whoami
    user = whoami(token=token)["name"]
    print(f"HF token OK: logged in as {user}")
    for repo in check_repos:
        model_info(repo, token=token)  # raises GatedRepoError / 401 if terms not accepted
        print(f"  access OK: {repo}")
    return user
