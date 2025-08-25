import yaml, os

def load_role_profile(role_name: str, base_path: str) -> dict:
    # expects a file like data/role_profiles/{role_name}.yml
    fname = os.path.join(base_path, "data", "role_profiles", f"{role_name}.yml")
    with open(fname, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
