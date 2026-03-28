"""Fix torchrl import renames across OmniDrones codebase."""
import os
import re

rename_map = {
    "BoundedTensorSpec": ("Bounded", "BoundedTensorSpec"),
    "UnboundedContinuousTensorSpec": ("Unbounded", "UnboundedContinuousTensorSpec"),
    "CompositeSpec": ("Composite", "CompositeSpec"),
    "DiscreteTensorSpec": ("Categorical", "DiscreteTensorSpec"),
    "MultiDiscreteTensorSpec": ("MultiCategorical", "MultiDiscreteTensorSpec"),
}

def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()

    original = content
    lines = content.split("\n")
    new_lines = []
    in_import_block = False

    for line in lines:
        stripped = line.strip()

        if "from torchrl" in line and "import" in line:
            in_import_block = True

        if in_import_block:
            for old_name, (new_name, alias) in rename_map.items():
                if f"{new_name} as " in line:
                    continue
                if old_name in line:
                    # Handle "UnboundedContinuousTensorSpec as UnboundedTensorSpec" pattern
                    line = line.replace(f"{old_name} as UnboundedTensorSpec", f"{new_name} as UnboundedTensorSpec")
                    # Handle bare old_name
                    if old_name in line and f"as {old_name}" not in line:
                        line = line.replace(old_name, f"{new_name} as {old_name}")

            if ")" in stripped:
                in_import_block = False
            elif not stripped.endswith(",") and not stripped.endswith("(") and "import" not in stripped:
                if stripped and not stripped.startswith("#"):
                    in_import_block = False

        new_lines.append(line)

    content = "\n".join(new_lines)
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False

# Walk all Python files
count = 0
for root, dirs, files in os.walk("omni_drones"):
    for fname in files:
        if not fname.endswith(".py"):
            continue
        path = os.path.join(root, fname)
        with open(path) as f:
            text = f.read()
        # Only process files with old-style torchrl imports
        has_old = any(old in text and f"as {old}" not in text for old in rename_map)
        if has_old:
            if fix_file(path):
                print(f"Fixed: {path}")
                count += 1

print(f"\nTotal files fixed: {count}")
