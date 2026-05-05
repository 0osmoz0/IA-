"""Shaders unlit pour le labyrinthe : essais 150 puis 120 (OpenGL 2.1 / macOS ancien)."""

from __future__ import annotations

from typing import Any, Optional

from ursina.shader import Shader
from ursina.vec2 import Vec2

_maze_unlit: Any = None  # Shader | False si tout échoue

VERT_150 = """#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;
out vec2 uvs;
out vec4 vertex_color;
uniform vec2 texture_scale;
uniform vec2 texture_offset;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = p3d_MultiTexCoord0 * texture_scale + texture_offset;
    vertex_color = p3d_Color;
}
"""

FRAG_150 = """#version 150

uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
in vec2 uvs;
in vec4 vertex_color;
out vec4 color;

void main() {
    color = texture(p3d_Texture0, uvs) * p3d_ColorScale * vertex_color;
}
"""

VERT_120 = """#version 120

uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
attribute vec4 p3d_Color;
varying vec2 uvs;
varying vec4 vertex_color;
uniform vec2 texture_scale;
uniform vec2 texture_offset;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = p3d_MultiTexCoord0 * texture_scale + texture_offset;
    vertex_color = p3d_Color;
}
"""

FRAG_120 = """#version 120

uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
varying vec2 uvs;
varying vec4 vertex_color;

void main() {
    gl_FragColor = texture2D(p3d_Texture0, uvs) * p3d_ColorScale * vertex_color;
}
"""

_INPUTS = {
    "texture_scale": Vec2(1, 1),
    "texture_offset": Vec2(0, 0),
}


def _try_compile(name: str, vert: str, frag: str) -> Optional[Shader]:
    sh = Shader(name=name, vertex=vert, fragment=frag, default_input=dict(_INPUTS))
    sh.compile(shader_includes=False)
    return sh


def get_maze_unlit_shader() -> Optional[Shader]:
    """Retourne un shader unique compilé (150 puis 120), ou None."""
    global _maze_unlit
    if _maze_unlit is False:
        return None
    if _maze_unlit is not None:
        return _maze_unlit

    for name, v, f in (
        ("maze_glsl150_unlit", VERT_150, FRAG_150),
        ("maze_glsl120_unlit", VERT_120, FRAG_120),
    ):
        try:
            _maze_unlit = _try_compile(name, v, f)
            return _maze_unlit
        except Exception:
            continue

    _maze_unlit = False
    return None
