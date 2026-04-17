"""
DAAP Agent Skills Manager — loads and registers AgentScope skills.

Skills are optional and configured via JSON:
  - Default: ~/.daap/skills.json
  - Override: DAAP_SKILLS_CONFIG_PATH

Supported config formats:
1) Object with "skills":
   {
     "skills": [
       {"dir": "/abs/path/to/skill", "targets": ["master", "subagent"]},
       {"dir": "/abs/path/to/another-skill", "targets": "master"}
     ]
   }

2) Bare list:
   [
     "/abs/path/to/skill-a",
     {"dir": "/abs/path/to/skill-b", "targets": ["subagent"]}
   ]

Targets:
  - "master": register only on master-agent toolkit
  - "subagent": register only on node/sub-agent toolkits
  - "all": register on both (default)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".daap" / "skills.json"
CONFIG_ENV_VAR = "DAAP_SKILLS_CONFIG_PATH"
_VALID_TARGETS = {"master", "subagent", "all"}


class SkillValidationError(Exception):
    """Raised when a skill directory fails validation."""


@dataclass(frozen=True)
class AgentSkillSpec:
    """Normalized skill spec."""

    directory: str
    targets: frozenset[str]


class SkillManager:
    """Loads configured skills and registers them onto AgentScope toolkits."""

    _DISCOVERY_PATHS: list[str] = [
        str(Path.home() / ".daap" / "skills"),
        "./skills",
    ]

    def __init__(self, config_path: str | None = None):
        cfg = config_path or os.environ.get(CONFIG_ENV_VAR, "")
        self._config_path = Path(cfg).expanduser() if cfg else DEFAULT_CONFIG_PATH
        config_specs = self._load_specs(self._config_path)
        discovered_specs = self._discover_skills(self._DISCOVERY_PATHS)
        config_dirs = {spec.directory for spec in config_specs}
        unique_discovered = [
            spec for spec in discovered_specs if spec.directory not in config_dirs
        ]
        self._specs: list[AgentSkillSpec] = config_specs + unique_discovered
        self._toolkit: Any = None
        self._toolkit_target: str = "master"

    def list_skill_dirs(self, target: str) -> list[str]:
        """Return configured skill directories for the given target."""
        normalized_target = (target or "").strip().lower()
        if normalized_target not in {"master", "subagent"}:
            return []

        out: list[str] = []
        for spec in self._specs:
            if "all" in spec.targets or normalized_target in spec.targets:
                out.append(spec.directory)
        return out

    def register_toolkit_skills(self, toolkit: Any, target: str) -> list[str]:
        """
        Register configured skills onto the provided toolkit.

        Returns list of successfully registered skill directories.
        """
        register = getattr(toolkit, "register_agent_skill", None)
        if not callable(register):
            logger.warning(
                "Toolkit does not support register_agent_skill(); skipping configured skills."
            )
            return []

        loaded: list[str] = []
        for skill_dir in self.list_skill_dirs(target):
            try:
                self._validate_skill_dir(skill_dir)
                register(skill_dir)
                loaded.append(skill_dir)
            except SkillValidationError as exc:
                logger.warning("Skipping invalid skill '%s': %s", skill_dir, exc)
            except Exception as exc:
                logger.warning("Failed registering skill '%s': %s", skill_dir, exc)
        return loaded

    @classmethod
    def _load_specs(cls, path: Path) -> list[AgentSkillSpec]:
        if not path.exists():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading skills config '%s': %s", path, exc)
            return []

        items: list[Any]
        if isinstance(raw, dict):
            if isinstance(raw.get("skills"), list):
                items = raw["skills"]
            else:
                logger.warning(
                    "Skills config '%s' has no 'skills' list. Expected {'skills': [...]} or [...].",
                    path,
                )
                return []
        elif isinstance(raw, list):
            items = raw
        else:
            logger.warning("Skills config '%s' must be an object or array.", path)
            return []

        specs: list[AgentSkillSpec] = []
        for item in items:
            spec = cls._normalize_item(item)
            if spec is not None:
                try:
                    cls._validate_skill_dir(spec.directory)
                    specs.append(spec)
                except SkillValidationError as exc:
                    logger.warning("Skipping invalid configured skill '%s': %s", spec.directory, exc)
        return specs

    @staticmethod
    def _normalize_item(item: Any) -> AgentSkillSpec | None:
        if isinstance(item, str):
            raw_dir = item
            targets = frozenset({"all"})
        elif isinstance(item, dict):
            raw_dir = item.get("dir") or item.get("path")
            raw_targets = item.get("targets", "all")

            if isinstance(raw_targets, str):
                target_items = [raw_targets]
            elif isinstance(raw_targets, list):
                target_items = raw_targets
            else:
                target_items = ["all"]

            normalized_targets = {
                str(t).strip().lower()
                for t in target_items
                if str(t).strip().lower() in _VALID_TARGETS
            }
            targets = frozenset(normalized_targets or {"all"})
        else:
            return None

        directory = str(raw_dir or "").strip()
        if not directory:
            return None

        normalized_dir = SkillManager._normalize_directory(directory)
        return AgentSkillSpec(directory=normalized_dir, targets=targets)

    @staticmethod
    def _normalize_directory(directory: str) -> str:
        expanded = os.path.expandvars(directory)
        return str(Path(expanded).expanduser())

    @staticmethod
    def _normalize_targets(targets: Any) -> frozenset[str]:
        if isinstance(targets, str):
            target_items = [targets]
        elif isinstance(targets, (list, tuple, set, frozenset)):
            target_items = list(targets)
        else:
            target_items = ["all"]

        normalized = {
            str(item).strip().lower()
            for item in target_items
            if str(item).strip().lower() in _VALID_TARGETS
        }
        return frozenset(normalized or {"all"})

    @staticmethod
    def _parse_skill_meta(directory: str) -> dict[str, str]:
        """
        Extract name and description from SKILL.md.

        Supports:
        1. YAML-like frontmatter between --- markers: name / description keys
        2. Bare markdown heading: # name, first non-heading line as description
        """
        content = (Path(directory) / "SKILL.md").read_text(encoding="utf-8")
        meta: dict[str, str] = {}

        if content.startswith("---"):
            lines = content.split("\n")
            end_idx = None
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    end_idx = i
                    break

            if end_idx is not None:
                for line in lines[1:end_idx]:
                    if ":" not in line:
                        continue
                    key, _, val = line.partition(":")
                    meta[key.strip().lower()] = val.strip()
                return meta

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("# "):
                meta["name"] = line[2:].strip()
                for next_line in lines[i + 1:]:
                    stripped = next_line.strip()
                    if stripped and not stripped.startswith("#"):
                        meta["description"] = stripped
                        break
                break

        return meta

    @staticmethod
    def _validate_skill_dir(directory: str) -> None:
        """
        Validate a skill directory and raise SkillValidationError on failure.

        Checks:
        1. Directory exists
        2. SKILL.md exists
        3. SKILL.md is non-empty
        4. Parsed metadata has required `name` and `description` fields
        """
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise SkillValidationError(f"Skill '{directory}': directory not found")

        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            raise SkillValidationError(f"Skill '{directory}': missing SKILL.md")

        content = skill_md.read_text(encoding="utf-8").strip()
        if not content:
            raise SkillValidationError(f"Skill '{directory}': SKILL.md is empty")

        meta = SkillManager._parse_skill_meta(str(path))
        if not meta.get("name"):
            raise SkillValidationError(
                f"Skill '{directory}': SKILL.md missing required field 'name'"
            )
        if not meta.get("description"):
            raise SkillValidationError(
                f"Skill '{directory}': SKILL.md missing required field 'description'"
            )

    @classmethod
    def _discover_skills(cls, search_paths: list[str]) -> list[AgentSkillSpec]:
        """
        Search each path for skill subdirectories containing valid SKILL.md files.

        Discovered skills default to target="all".
        """
        found: list[AgentSkillSpec] = []
        for search_path in search_paths:
            root = Path(search_path).expanduser()
            if not root.is_dir():
                continue

            for candidate in sorted(root.iterdir()):
                if not candidate.is_dir():
                    continue
                try:
                    cls._validate_skill_dir(str(candidate))
                    found.append(
                        AgentSkillSpec(
                            directory=str(candidate),
                            targets=frozenset({"all"}),
                        )
                    )
                except SkillValidationError as exc:
                    logger.debug("Auto-discovery skipped '%s': %s", candidate, exc)
        return found

    def bind_toolkit(self, toolkit: Any, target: str = "master") -> None:
        """Bind the active toolkit so add/remove can hot-reload skills."""
        self._toolkit = toolkit
        self._toolkit_target = (target or "master").strip().lower()

    def add_skill(
        self,
        directory: str,
        targets: str | list[str] | tuple[str, ...] | set[str] | frozenset[str] = "all",
        persist: bool = True,
    ) -> tuple[str, bool]:
        """
        Validate, register, and optionally persist a skill.

        Returns (skill_name, was_newly_added).
        """
        normalized_dir = self._normalize_directory(directory)
        self._validate_skill_dir(normalized_dir)

        existing = next((spec for spec in self._specs if spec.directory == normalized_dir), None)
        if existing is not None:
            meta = self._parse_skill_meta(normalized_dir)
            return meta.get("name", Path(normalized_dir).name), False

        normalized_targets = self._normalize_targets(targets)
        should_hot_register = (
            self._toolkit is not None
            and (
                "all" in normalized_targets
                or self._toolkit_target in normalized_targets
            )
        )
        if should_hot_register:
            register = getattr(self._toolkit, "register_agent_skill", None)
            if callable(register):
                register(normalized_dir)

        self._specs.append(
            AgentSkillSpec(directory=normalized_dir, targets=normalized_targets)
        )
        if persist:
            self._write_config()

        meta = self._parse_skill_meta(normalized_dir)
        return meta.get("name", Path(normalized_dir).name), True

    def remove_skill(self, directory: str, persist: bool = True) -> str:
        """
        Remove a skill from manager state and optionally persist config updates.

        Returns removed skill name. Raises KeyError when not found.
        """
        normalized_dir = self._normalize_directory(directory)
        matching = [spec for spec in self._specs if spec.directory == normalized_dir]
        if not matching:
            raise KeyError(f"Skill not registered: '{normalized_dir}'")

        try:
            meta = self._parse_skill_meta(normalized_dir)
            skill_name = meta.get("name", Path(normalized_dir).name)
        except Exception:
            skill_name = Path(normalized_dir).name

        self._specs = [spec for spec in self._specs if spec.directory != normalized_dir]

        should_hot_unregister = (
            self._toolkit is not None
            and any(
                "all" in spec.targets or self._toolkit_target in spec.targets
                for spec in matching
            )
        )
        if should_hot_unregister:
            unregister = getattr(self._toolkit, "unregister_agent_skill", None)
            if callable(unregister):
                unregister(normalized_dir)

        if persist:
            self._write_config()

        return skill_name

    def _write_config(self) -> None:
        """Atomically write current skills to the configured JSON file."""
        entries: list[dict[str, Any]] = []
        for spec in self._specs:
            entries.append(
                {
                    "dir": spec.directory,
                    "targets": sorted(spec.targets),
                }
            )

        payload = json.dumps({"skills": entries}, indent=2, ensure_ascii=False)
        tmp_path = self._config_path.with_suffix(f"{self._config_path.suffix}.tmp")
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(str(tmp_path), str(self._config_path))


_SKILL_MANAGER: SkillManager | None = None


def get_skill_manager(config_path: str | None = None) -> SkillManager:
    """Return singleton SkillManager."""
    global _SKILL_MANAGER

    if _SKILL_MANAGER is None:
        _SKILL_MANAGER = SkillManager(config_path=config_path)
    return _SKILL_MANAGER


def apply_configured_skills(toolkit: Any, target: str) -> list[str]:
    """
    Best-effort skill registration helper (never raises).

    target: "master" or "subagent"
    """
    try:
        manager = get_skill_manager()
        return manager.register_toolkit_skills(toolkit, target=target)
    except Exception as exc:
        logger.warning("Skill registration disabled (non-fatal): %s", exc)
        return []
