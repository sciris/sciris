# Sciris Claude Code Plugin

This folder contains a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills plugin that helps Claude work more effectively with [Sciris](https://sciris.org).

## What it does

The plugin provides Claude with domain-specific knowledge about Sciris features, organized into focused skills. When you ask Claude to help with Sciris code, these skills give it detailed guidance on the right functions, patterns, and idioms to use.

The plugin also configures MCP servers ([Context7](https://context7.com) and [GitMCP](https://gitmcp.io)) so Claude can look up current Sciris documentation on the fly.

## Installation

In Claude Code, add the Sciris repository as a marketplace:

```
/install-plugin https://github.com/sciris/sciris
```

Or manually add `https://github.com/sciris/sciris` as a marketplace in your Claude Code settings.

## Available skills

| Skill | Description |
|-------|-------------|
| **sciris-intro** | Basic Sciris features and quick reference |
| **sciris-arrays** | Array operations (`sc.findinds`, `sc.findnearest`, `sc.toarray`, etc.) |
| **sciris-dicts** | Containers (`sc.odict`, `sc.objdict`, `sc.counter`, etc.) |
| **sciris-files** | File I/O (`sc.save`, `sc.load`, `sc.savejson`, etc.) |
| **sciris-plotting** | Matplotlib extensions (`sc.boxoff`, `sc.dateformatter`, etc.) |
| **sciris-parallel** | Parallelization (`sc.parallelize`, `sc.loadbalancer`) |
| **sciris-dates** | Date/time handling (`sc.date`, `sc.daterange`, `sc.timer`) |
| **sciris-printing** | Printing and formatting (`sc.heading`, `sc.sigfig`, `sc.progressbar`) |
| **sciris-utils** | Utilities (`sc.mergedicts`, `sc.tryexcept`, `sc.checkmem`, etc.) |
| **sciris-advanced** | Advanced features (`sc.odict` subclassing, nested operations, versioning) |

## Organization

```
claude_plugin/
├── README.md                          # This file
├── .claude-plugin/
│   └── plugin.json                    # Plugin manifest (name, MCP servers, metadata)
└── skills/
    ├── sciris-intro/SKILL.md
    ├── sciris-arrays/SKILL.md
    ├── sciris-dicts/SKILL.md
    ├── sciris-files/SKILL.md
    ├── sciris-plotting/SKILL.md
    ├── sciris-parallel/SKILL.md
    ├── sciris-dates/SKILL.md
    ├── sciris-printing/SKILL.md
    ├── sciris-utils/SKILL.md
    └── sciris-advanced/SKILL.md
```

Each skill is a single `SKILL.md` file with YAML frontmatter (name and description for triggering) followed by markdown content with usage examples and API guidance.

The marketplace registration file at `.claude-plugin/marketplace.json` (in the repo root) points to this folder.
