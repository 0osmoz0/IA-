"""Rendu 3D Ursina synchronisé sur l'état du labyrinthe."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from panda3d.core import OrthographicLens, Point3

from ursina import (
    AmbientLight,
    DirectionalLight,
    Entity,
    Text,
    Ursina,
    Vec3,
    camera,
    color,
    destroy,
    window,
)


def _cell_world(y: int, x: int, h: int, w: int) -> Vec3:
    cx = x - (w - 1) / 2.0
    cz = -y + (h - 1) / 2.0
    return Vec3(cx, 0.0, cz)


def _find_main_3d_display_region(app: Ursina):
    """Région d'affichage branchée sur la caméra 3D par défaut (ShowBase.cam)."""
    win = app.win
    main = getattr(app, "cam", None)
    n = win.getNumDisplayRegions()
    for i in range(n):
        dr = win.getDisplayRegion(i)
        c = dr.getCamera()
        if c is None or c.isEmpty():
            continue
        if main is not None and not main.isEmpty() and c == main:
            return dr
    return win.getDisplayRegion(0)


ViewMode = Literal["carte", "split", "suive"]


class MazeRenderer:
    def __init__(
        self,
        app: Ursina,
        *,
        view_mode: ViewMode = "carte",
        simple_shading: bool = False,
    ):
        self.app = app
        self._view_mode: ViewMode = view_mode
        self._single_view = view_mode != "split"
        self._simple_shading = simple_shading
        from src.maze_glsl import get_maze_unlit_shader

        self._flat_shader = get_maze_unlit_shader()
        self._static_entities: List[Entity] = []
        self._dynamic_entities: List[Entity] = []
        self._hud: Optional[Text] = None
        self._last_grid_shape: Optional[tuple[int, int]] = None
        self._split_done = False
        self._top_cam_np: Any = None
        self._top_ortho: Optional[OrthographicLens] = None
        DirectionalLight(type="directional", rotation=(50, -40, 0))
        AmbientLight(color=color.rgba(0.55, 0.55, 0.58))
        self._setup_split_view()

    def _ent(self, **kwargs: Any) -> Entity:
        kwargs.pop("unlit", None)
        if self._flat_shader is not None:
            kwargs["shader"] = self._flat_shader
        elif self._simple_shading:
            kwargs.setdefault("unlit", True)
            try:
                return Entity(**kwargs)
            except TypeError:
                kwargs.pop("unlit", None)
        return Entity(**kwargs)

    def _setup_split_view(self) -> None:
        if self._split_done or self._single_view:
            self._split_done = True
            return
        win = self.app.win
        dr_main = _find_main_3d_display_region(self.app)
        dr_main.setDimensions(0.0, 0.5, 0.0, 1.0)
        sort = dr_main.getSort()
        self._top_cam_np = self.app.makeCamera(
            win,
            displayRegion=(0.5, 1.0, 0.0, 1.0),
            sort=sort + 1,
            camName="maze_topdown",
        )
        self._top_ortho = OrthographicLens()
        self._top_cam_np.node().setLens(self._top_ortho)
        self._split_done = True

    def _update_top_camera(self, h: int, w: int) -> None:
        if self._single_view or self._top_cam_np is None or self._top_ortho is None:
            return
        pad = 2.0
        base_fw = float(w) + pad
        base_fh = float(h) + pad
        xsize = max(1, self.app.win.getXSize())
        ysize = max(1, self.app.win.getYSize())
        panel_aspect = (0.5 * xsize) / ysize
        ar = base_fw / max(base_fh, 1e-6)
        if ar < panel_aspect:
            film_w = base_fh * panel_aspect
            film_h = base_fh
        else:
            film_w = base_fw
            film_h = base_fw / max(panel_aspect, 1e-6)
        self._top_ortho.setFilmSize(film_w, film_h)
        alt = max(18.0, max(h, w) * 1.15)
        self._top_cam_np.setPos(0, alt, 0)
        self._top_cam_np.lookAt(Point3(0, 0, 0))

    def clear_static(self) -> None:
        for e in self._static_entities:
            destroy(e)
        self._static_entities.clear()

    def clear_dynamic(self) -> None:
        for e in self._dynamic_entities:
            destroy(e)
        self._dynamic_entities.clear()

    def rebuild_if_needed(self, snap: Dict[str, Any]) -> None:
        grid = snap["grid"]
        shape = tuple(grid.shape)
        if shape != self._last_grid_shape:
            self._last_grid_shape = shape
            self._full_rebuild(snap)

    def _full_rebuild(self, snap: Dict[str, Any]) -> None:
        self.clear_static()
        grid = snap["grid"]
        h, w = grid.shape
        floor = self._ent(
            model="plane",
            scale=(max(w, h) + 2, 1, max(w, h) + 2),
            color=color.rgba(0.3, 0.34, 0.4, 1),
            position=Vec3(0, -0.02, 0),
        )
        self._static_entities.append(floor)

        for y in range(h):
            for x in range(w):
                if int(grid[y, x]) != 1:
                    continue
                p = _cell_world(y, x, h, w)
                wall = self._ent(
                    model="cube",
                    scale=(0.92, 1.25, 0.92),
                    position=Vec3(p.x, 0.62, p.z),
                    color=color.rgb(0.72, 0.74, 0.8),
                )
                self._static_entities.append(wall)

        ey, ex = snap["exit"]
        ep = _cell_world(ey, ex, h, w)
        goal = self._ent(
            model="cube",
            scale=(0.55, 0.12, 0.55),
            position=Vec3(ep.x, 0.08, ep.z),
            color=color.rgba(0.15, 0.95, 0.35, 1),
        )
        self._static_entities.append(goal)

    def sync(self, snap: Dict[str, Any]) -> None:
        self.rebuild_if_needed(snap)
        grid = snap["grid"]
        h, w = grid.shape
        self.clear_dynamic()

        ay, ax = snap["agent"]
        ap = _cell_world(ay, ax, h, w)
        agent_cube = self._ent(
            model="cube",
            scale=(0.5, 0.55, 0.5),
            position=Vec3(ap.x, 0.38, ap.z),
            color=color.rgba(0.15, 0.55, 0.95, 1),
        )
        self._dynamic_entities.append(agent_cube)

        if snap.get("key_pos") and not snap.get("key_taken"):
            ky, kx = snap["key_pos"]
            kp = _cell_world(ky, kx, h, w)
            self._dynamic_entities.append(
                self._ent(
                    model="cube",
                    scale=(0.28, 0.15, 0.28),
                    position=Vec3(kp.x, 0.2, kp.z),
                    color=color.gold,
                )
            )

        if snap.get("door_pos"):
            dy, dx = snap["door_pos"]
            dp = _cell_world(dy, dx, h, w)
            door_color = color.rgba(0.55, 0.35, 0.15, 0.95) if not snap["has_key"] else color.rgba(0.25, 0.6, 0.25, 0.5)
            self._dynamic_entities.append(
                self._ent(
                    model="cube",
                    scale=(0.55, 0.35, 0.55),
                    position=Vec3(dp.x, 0.4, dp.z),
                    color=door_color,
                )
            )

        for ty, tx in snap.get("traps", set()):
            tp = _cell_world(ty, tx, h, w)
            self._dynamic_entities.append(
                self._ent(
                    model="cube",
                    scale=(0.35, 0.05, 0.35),
                    position=Vec3(tp.x, 0.04, tp.z),
                    color=color.rgba(0.9, 0.15, 0.15, 0.85),
                )
            )

        pa, pb = snap.get("portal_a"), snap.get("portal_b")
        if pa is not None and pb is not None:
            for (yy, xx), col in ((pa, color.violet), (pb, color.magenta)):
                pp = _cell_world(yy, xx, h, w)
                self._dynamic_entities.append(
                    self._ent(
                        model="cube",
                        scale=(0.32, 0.12, 0.32),
                        position=Vec3(pp.x, 0.1, pp.z),
                        color=col,
                    )
                )

        mw = snap.get("mobile_wall")
        if mw is not None:
            my, mx = mw
            mp = _cell_world(my, mx, h, w)
            blocking = snap.get("mobile_blocking", False)
            col = color.rgba(1, 0.45, 0.1, 1) if blocking else color.rgba(1, 0.7, 0.2, 0.35)
            self._dynamic_entities.append(
                self._ent(
                    model="cube",
                    scale=(0.88, 0.88, 0.88) if blocking else (0.55, 0.2, 0.55),
                    position=Vec3(mp.x, 0.55 if blocking else 0.18, mp.z),
                    color=col,
                )
            )

        blocking = snap.get("mobile_blocking", False)
        if self._view_mode == "carte":
            layout = "Vue CARTE (dessus, lisible) | "
        elif self._view_mode == "suive":
            layout = "Vue SUIVI 3D (caméra agent) | "
        else:
            layout = "Gauche: 3D | Droite: dessus | "
        txt = (
            f"{layout}"
            f"Niv {snap['level'] + 1}/5 | Clé: {'oui' if snap['has_key'] else 'non'} | "
            f"Mur mob.: {'bloque' if blocking else 'ouvert'}"
        )
        if self._hud is None:
            self._hud = Text(
                text=txt,
                position=(-0.88, 0.47),
                origin=(-0.5, 0.5),
                scale=1.05,
                color=color.rgb(1, 1, 1),
            )
        else:
            self._hud.text = txt

        self._apply_camera(h, w, ap)
        self._update_top_camera(h, w)

    def _apply_camera(self, h: int, w: int, ap: Vec3) -> None:
        """Réglage caméra selon le mode (carte = plateau ; suive / split = perspective)."""
        if self._view_mode == "carte":
            camera.orthographic = True
            pad = 2.8
            fw = float(w) + pad
            fh = float(h) + pad
            ar = max(0.01, float(window.aspect_ratio))
            if fw / fh < ar:
                fw = fh * ar
            else:
                fh = fw / ar
            camera.orthographic_lens.set_film_size(fw, fh)
            alt = max(22.0, max(h, w) * 2.2)
            camera.position = Vec3(0, alt, 0)
            camera.look_at(Vec3(0, 0, 0))
            return

        camera.orthographic = False
        if self._view_mode == "suive":
            d = max(9.0, max(h, w) * 0.55)
            camera.position = Vec3(ap.x + d * 0.65, d * 0.85, ap.z + d * 0.65)
            camera.look_at(Vec3(ap.x, 0.35, ap.z))
            return

        camera.position = Vec3(ap.x, max(8.0, max(h, w) * 0.65), ap.z + max(8.0, max(h, w) * 0.65))
        camera.look_at(Vec3(ap.x, 0.25, ap.z))
