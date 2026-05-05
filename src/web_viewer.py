"""Interface web 2D (Flask) pour voir le labyrinthe sans OpenGL."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>IA labyrinthe — vue web</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0f1218; color: #e8eaed; margin: 0; padding: 16px; }
    h1 { font-size: 1.15rem; font-weight: 600; margin: 0 0 10px; letter-spacing: 0.02em; }
    #meta { font-size: 0.88rem; line-height: 1.55; margin-bottom: 10px; color: #bdc1c6; }
    #meta .hl { color: #e8eaed; font-weight: 600; }
    #meta .ok { color: #6ee7b7; }
    #meta .blocked { color: #fca5a5; }
    #meta .rneg { color: #fbbf24; }
    #wrap {
      display: inline-block;
      border: 3px solid #3b4354;
      border-radius: 12px;
      overflow: hidden;
      background: #080a0d;
      box-shadow: 0 12px 40px rgba(0,0,0,.45);
    }
    canvas { display: block; }
    #hint { margin-top: 14px; font-size: 0.82rem; color: #80868b; max-width: 720px; }
    .lag { color: #f9ab00; }
  </style>
</head>
<body>
  <h1>Labyrinthe — vue live</h1>
  <div id="meta">Chargement…</div>
  <div id="wrap"><canvas id="c" width="720" height="720"></canvas></div>
  <p id="hint">Légende : <b style="color:#22d3ee">agent</b> (sillage = derniers pas) · <b style="color:#4ade80">sortie</b> · gris = mur · <b style="color:#fbbf24">clé</b> · <b style="color:#b45309">porte</b> · <b style="color:#f87171">piège</b> · <b style="color:#c084fc">portail</b> · <b style="color:#fb923c">mur mobile</b>. Si le compteur global monte mais l’agent ne bouge pas, il tente souvent d’aller dans un mur (voir « mur » en rouge).</p>
  <script>
    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    const meta = document.getElementById('meta');

    const ACT = ['haut', 'droite', 'bas', 'gauche'];
    const ARROW = ['↑', '→', '↓', '←'];

    const COL = {
      wall: '#4b5568',
      floor: '#1e2430',
      exit: '#059669',
      key: '#fbbf24',
      door: '#b45309',
      doorOpen: '#15803d',
      trap: '#b91c1c',
      mobile: '#ea580c',
      mobileOpen: 'rgba(234,88,12,0.28)',
    };

    function draw(data) {
      if (!data || !data.grid || !data.grid.length) {
        meta.textContent = 'En attente des données…';
        return;
      }
      const g = data.grid;
      const rows = g.length, cols = g[0].length;
      const maxPx = Math.min(720, Math.floor(window.innerWidth) - 48);
      const cell = Math.max(8, Math.floor(Math.min(maxPx / cols, maxPx / rows)));

      canvas.width = cols * cell;
      canvas.height = rows * cell;

      ctx.fillStyle = '#080a0d';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      function cellCenter(y, x) {
        return [x * cell + cell / 2, y * cell + cell / 2];
      }

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const v = g[y][x];
          ctx.fillStyle = v === 1 ? COL.wall : COL.floor;
          ctx.fillRect(x * cell, y * cell, cell, cell);
          if (v !== 1 && ((x + y) & 1) === 1) {
            ctx.fillStyle = 'rgba(255,255,255,0.03)';
            ctx.fillRect(x * cell, y * cell, cell, cell);
          }
        }
      }

      if (data.exit) {
        const [ey, ex] = data.exit;
        ctx.fillStyle = COL.exit;
        ctx.fillRect(ex * cell + 1, ey * cell + 1, cell - 2, cell - 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        ctx.strokeRect(ex * cell + 0.5, ey * cell + 0.5, cell - 1, cell - 1);
      }
      if (data.key_pos && !data.key_taken) {
        const [ky, kx] = data.key_pos;
        ctx.fillStyle = COL.key;
        ctx.beginPath();
        const [cx, cy] = cellCenter(ky, kx);
        ctx.arc(cx, cy, Math.max(2, cell * 0.32), 0, 6.28);
        ctx.fill();
      }
      if (data.door_pos) {
        const [dy, dx] = data.door_pos;
        ctx.fillStyle = data.has_key ? COL.doorOpen : COL.door;
        ctx.fillRect(dx * cell + 2, dy * cell + 2, cell - 4, cell - 4);
      }
      if (data.traps) {
        data.traps.forEach(([ty, tx]) => {
          ctx.fillStyle = COL.trap;
          ctx.fillRect(tx * cell + 3, ty * cell + 3, cell - 6, cell - 6);
        });
      }
      if (data.portal_a && data.portal_b) {
        [[data.portal_a, '#a855f7'], [data.portal_b, '#d946ef']].forEach(([p, col]) => {
          const [py, px] = p;
          ctx.strokeStyle = col;
          ctx.lineWidth = 2;
          ctx.strokeRect(px * cell + 1, py * cell + 1, cell - 2, cell - 2);
        });
      }
      if (data.mobile_wall) {
        const [my, mx] = data.mobile_wall;
        ctx.fillStyle = data.mobile_blocking ? COL.mobile : COL.mobileOpen;
        ctx.fillRect(mx * cell, my * cell, cell, cell);
      }

      /* Traînée récente (dernier en plus visible) */
      if (data.trail && data.trail.length > 1) {
        const tr = data.trail;
        for (let i = 0; i < tr.length - 1; i++) {
          const [ty, tx] = tr[i];
          const age = i / Math.max(1, tr.length - 1);
          const alpha = 0.08 + 0.38 * age;
          ctx.fillStyle = 'rgba(34,211,238,' + alpha + ')';
          const [cx, cy] = cellCenter(ty, tx);
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(2, cell * 0.22), 0, 6.28);
          ctx.fill();
        }
      }

      let actionHtml = '';
      const la = data.last_action;
      if (la != null && la >= 0 && la < 4) {
        const [ay, ax] = data.agent || [0, 0];
        const [cx, cy] = cellCenter(ay, ax);
        const dirs = [[0,-1],[1,0],[0,1],[-1,0]];
        const [dx, dy] = dirs[la];
        ctx.strokeStyle = data.last_moved ? 'rgba(110,231,183,0.9)' : 'rgba(248,113,113,0.95)';
        ctx.lineWidth = Math.max(2, cell * 0.1);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + dx * cell * 0.38, cy + dy * cell * 0.38);
        ctx.stroke();
        actionHtml = ARROW[la] + ' <span class="hl">' + ACT[la] + '</span>';
      } else {
        actionHtml = '<span class="hl">—</span>';
      }

      if (data.agent) {
        const [ay, ax] = data.agent;
        const [cx, cy] = cellCenter(ay, ax);
        const rad = Math.max(4, cell * 0.44);
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath();
        ctx.arc(cx, cy, rad, 0, 6.28);
        ctx.fill();
        ctx.strokeStyle = '#f0f9ff';
        ctx.lineWidth = Math.max(2, cell * 0.08);
        ctx.stroke();
        ctx.fillStyle = 'rgba(15,23,42,0.85)';
        ctx.beginPath();
        ctx.arc(cx, cy, rad * 0.35, 0, 6.28);
        ctx.fill();
      }

      const mb = data.mobile_blocking ? 'bloqué' : 'ouvert';
      const movedCls = data.last_moved ? 'ok' : 'blocked';
      const movedTxt = data.last_moved ? 'oui' : 'non (dans un mur)';
      const lr = (data.last_reward != null) ? Number(data.last_reward).toFixed(2) : '—';
      const ep = (data.episode_steps != null) ? data.episode_steps : '—';
      meta.innerHTML =
        'Action : ' + actionHtml +
        ' · déplacement : <span class="' + movedCls + '"><b>' + movedTxt + '</b></span>' +
        ' · pas (épisode) : <b class="hl">' + ep + '</b>' +
        ' · pas (total) : <b class="hl">' + (data.step_count ?? 0) + '</b>' +
        ' · R : <span class="' + ((data.last_reward != null && data.last_reward < 0) ? 'rneg' : 'hl') + '">' + lr + '</span>' +
        '<br/>Niveau <b class="hl">' + ((data.level ?? 0) + 1) + '</b>/5 · clé : <b class="hl">' + (data.has_key ? 'oui' : 'non') +
        '</b> · mur mobile : <b class="hl">' + mb + '</b>';
    }

    async function poll() {
      try {
        const r = await fetch('/api/snapshot', { cache: 'no-store' });
        const j = await r.json();
        draw(j);
      } catch (e) {
        meta.innerHTML = '<span class="lag">Erreur réseau (serveur lancé ?)</span>';
      }
      setTimeout(poll, 16);
    }
    poll();
  </script>
</body>
</html>
"""


def snapshot_to_json(snap: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in snap.items():
        if k == "grid" and hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif k == "traps":
            out[k] = [list(t) for t in v]
        elif k == "trail":
            out[k] = [list(p) for p in v]
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, set):
            out[k] = [list(t) for t in v]
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
    return out


class WebViewerBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._json = "{}"
        self._thread: Optional[threading.Thread] = None

    def publish(self, snap: Dict[str, Any]) -> None:
        payload = snapshot_to_json(snap)
        with self._lock:
            self._json = json.dumps(payload)

    def _make_app(self):
        from flask import Flask, Response

        app = Flask(__name__)
        bridge = self

        @app.route("/")
        def index() -> Response:
            return Response(HTML_PAGE, mimetype="text/html; charset=utf-8")

        @app.route("/api/snapshot")
        def api_snap() -> Response:
            with bridge._lock:
                data = bridge._json
            return Response(
                data,
                mimetype="application/json",
                headers={"Cache-Control": "no-store"},
            )

        return app

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        app = self._make_app()

        def run() -> None:
            app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False,
                debug=False,
            )

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        print(f"\n>>> Vue web : ouvrez http://{self.host}:{self.port}/ dans votre navigateur\n")
        import time

        time.sleep(0.35)


class LiveWebCallback(BaseCallback):
    def __init__(self, bridge: WebViewerBridge, every: int = 1):
        super().__init__(verbose=0)
        self.bridge = bridge
        self.every = max(1, every)

    def _on_step(self) -> bool:
        if self.n_calls % self.every != 0:
            return True
        snaps = self.training_env.env_method("snapshot_for_render")
        if snaps:
            self.bridge.publish(snaps[0])
        return True
