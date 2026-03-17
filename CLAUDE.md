# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ML Zero To Hero** — a Hugo static site deployed to GitHub Pages at `https://taaaaaaw.github.io/`. Uses the [Stack theme](https://github.com/CaiJimmy/hugo-theme-stack) as a git submodule.

## Commands

```bash
# Local development server
hugo server

# Build for production (also run by CI)
hugo --minify

# Create a new post
hugo new content/posts/<category>/<slug>.md

# Update theme submodule
git submodule update --init --recursive
```

## Deployment

Pushing to `main` automatically triggers GitHub Actions (`.github/workflows/deploy.yml`), which builds with `hugo --minify` and deploys `./public` to GitHub Pages. No manual deploy step needed.

## Content Structure

Posts live in `content/posts/<category>/`. Front matter uses YAML and should include:

```yaml
---
title: "Post Title"
date: YYYY-MM-DD
description: ""
categories:
    - Category-Name
tags:
    - tag
draft: false
---
```

- `disablePathToLower = true` is set — category/tag names in URLs preserve their original casing.
- Math rendering (`math = true`), table of contents (`toc = true`), and heading anchors are enabled globally via `hugo.toml`.
- The Stack theme (`themes/stack/`) is a submodule — do not edit files inside it directly.
