"""Configuration Panda3D avant ``Ursina()`` / ShowBase — l'ordre d'appel compte."""

from __future__ import annotations

VALID_GL_PROFILES = frozenset({"default", "core32", "core33", "legacy21"})


def apply_panda_gl_profile(profile: str) -> None:
    """
    Appliquer un profil OpenGL via loadPrcFileData **avant** tout import Ursina/ShowBase.

    - default : aucun prc (recommandé en premier sur macOS pour éviter l'écran noir GLSL).
    - core32 / core33 : contexte Core explicite (peut casser des shaders 130/140).
    - legacy21 : expérimental (OpenGL 2.1).
    """
    p = (profile or "default").lower().strip()
    if p not in VALID_GL_PROFILES:
        raise ValueError(f"--gl-profile doit être parmi {sorted(VALID_GL_PROFILES)}, reçu: {profile!r}")
    if p == "default":
        return

    from panda3d.core import loadPrcFileData

    if p == "core32":
        loadPrcFileData("", "gl-version 3 2")
    elif p == "core33":
        loadPrcFileData("", "gl-version 3 3")
    elif p == "legacy21":
        loadPrcFileData("", "gl-version 2 1")
