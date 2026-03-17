#!/usr/bin/env python3
"""
Migration script: Jupyter Book v1 → v2
- Converts {code-block} python / :class: thebe → {code-cell} python
- Adds kernelspec frontmatter to files with code-cells but no frontmatter
"""
import os
import re

KERNELSPEC_FRONTMATTER = """---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

"""

BOOK_DIR = "/home/roderickperez/DataScienceProjects/EAGE_Python_course/EAGE_PythonRenewableEnergyCourse"

def has_frontmatter(content):
    return content.startswith("---")

def has_kernelspec(content):
    return "kernelspec" in content[:500]

def convert_thebe_blocks(content):
    """
    Convert:
      ```{code-block} python
      :class: thebe
      <code>
      ```
    to:
      ```{code-cell} python
      <code>
      ```
    Also handles ipython3 variant.
    """
    # Pattern: ```{code-block} python/ipython3 followed by :class: thebe
    pattern = re.compile(
        r'```\{code-block\}\s+(python|ipython3)\n:class: thebe\n',
        re.MULTILINE
    )
    result = pattern.sub('```{code-cell} python\n', content)

    # Also handle ```{code-block} python\n:class: thebe (with possible spaces)
    pattern2 = re.compile(
        r'```\{code-block\}\s+(python|ipython3)\s*\n\s*:class:\s*thebe\s*\n',
        re.MULTILINE
    )
    result = pattern2.sub('```{code-cell} python\n', result)

    return result

def convert_ipython3_cells(content):
    """Convert {code-cell} ipython3 → {code-cell} python for JupyterLite compat."""
    return content.replace('```{code-cell} ipython3', '```{code-cell} python')

def add_kernelspec_if_needed(content, filepath):
    """Add kernelspec frontmatter if the file has code-cells but no kernelspec."""
    has_cells = '{code-cell}' in content
    if not has_cells:
        return content  # No executable cells, no changes needed

    if has_frontmatter(content):
        if has_kernelspec(content):
            return content  # Already has kernelspec, do nothing
        else:
            # Has frontmatter but no kernelspec — inject kernelspec into existing frontmatter
            # Find the closing ---
            end = content.find('\n---\n', 3)
            if end == -1:
                end = content.find('\n---', 3)
            if end != -1:
                frontmatter = content[:end]
                rest = content[end:]
                kernelspec_block = """
kernelspec:
  name: python3
  display_name: Python 3
  language: python"""
                return frontmatter + kernelspec_block + rest
            return content
    else:
        # No frontmatter at all — prepend full kernelspec block
        return KERNELSPEC_FRONTMATTER + content

def migrate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        original = f.read()

    content = original

    # Step 1: Convert thebe code-blocks to code-cells
    content = convert_thebe_blocks(content)

    # Step 2: Normalize ipython3 → python for JupyterLite
    content = convert_ipython3_cells(content)

    # Step 3: Add kernelspec frontmatter if needed
    content = add_kernelspec_if_needed(content, filepath)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    changed = []
    skipped = []

    for root, dirs, files in os.walk(BOOK_DIR):
        # Skip hidden dirs and _build
        dirs[:] = [d for d in dirs if not d.startswith('_') and not d.startswith('.')]
        for fname in files:
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(root, fname)
            try:
                modified = migrate_file(fpath)
                if modified:
                    rel = fpath.replace(BOOK_DIR + '/', '')
                    changed.append(rel)
                else:
                    skipped.append(fpath.replace(BOOK_DIR + '/', ''))
            except Exception as e:
                print(f"ERROR processing {fpath}: {e}")

    print(f"\n=== Migration complete ===")
    print(f"Modified ({len(changed)} files):")
    for f in sorted(changed):
        print(f"  ✓ {f}")
    print(f"\nUnchanged ({len(skipped)} files):")
    for f in sorted(skipped):
        print(f"  - {f}")

if __name__ == '__main__':
    main()
