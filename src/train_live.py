"""Entraînement PPO avec rendu live (Ursina) ou headless."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.curriculum import CurriculumManager
from src.maze_env import MazeEnv
from src.panda_gl_config import VALID_GL_PROFILES
from src.web_viewer import LiveWebCallback, WebViewerBridge


class CurriculumWrapper(gym.Wrapper):
    """Enregistre succès/échec et promeut le curriculum à la fin d'épisode."""

    def snapshot_for_render(self):
        return self.env.unwrapped.snapshot_for_render()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            base = self.env.unwrapped
            success = bool(info.get("success", False))
            base.curriculum.register_episode(success)
            base.curriculum.maybe_advance()
        return obs, reward, terminated, truncated, info


class LiveRenderCallback(BaseCallback):
    def __init__(self, renderer: Any, every: int = 1):
        super().__init__(verbose=0)
        self.renderer = renderer
        self.every = max(1, every)

    def _on_step(self) -> bool:
        if self.renderer is None:
            return True
        if self.n_calls % self.every != 0:
            return True
        snaps = self.training_env.env_method("snapshot_for_render")
        if snaps:
            snap = snaps[0]
            self.renderer.rebuild_if_needed(snap)
            self.renderer.sync(snap)
        return True


def build_env(curriculum: CurriculumManager, *, sticky_maze: bool = True) -> gym.Env:
    return CurriculumWrapper(
        MazeEnv(seed=None, curriculum=curriculum, sticky_maze=sticky_maze)
    )


def _make_ppo(env: gym.Env, n_steps: int, load_path: str | None) -> PPO:
    if load_path and os.path.isfile(load_path):
        return PPO.load(load_path, env=env)
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        learning_rate=3e-4,
        gamma=0.99,
    )


def run_web(
    timesteps: int,
    chunk: int,
    save_path: str,
    load_path: str | None,
    eval_only: bool,
    fast: bool,
    seed: int | None,
    web_host: str,
    web_port: int,
    sticky_maze: bool,
    web_step_delay: float,
) -> None:
    import time

    curriculum = CurriculumManager()
    env = build_env(curriculum, sticky_maze=sticky_maze)
    n_steps = max(64, chunk)
    model = _make_ppo(env, n_steps, load_path)

    base = env.unwrapped
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    bridge = WebViewerBridge(host=web_host, port=web_port)
    bridge.publish(base.snapshot_for_render())
    bridge.start_background()

    poll_every = 4 if fast else 1

    if eval_only:
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(action))
                bridge.publish(base.snapshot_for_render())
                if term or trunc:
                    obs, _ = env.reset()
                    bridge.publish(base.snapshot_for_render())
                if web_step_delay > 0:
                    time.sleep(web_step_delay)
        except KeyboardInterrupt:
            print("Arrêt (Ctrl+C).")
        return

    train_cb = LiveWebCallback(bridge, every=poll_every)
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=train_cb,
    )
    model.save(save_path)
    print(f"Modèle sauvegardé: {save_path}")


def run_headless(
    timesteps: int, save_path: str, load_path: str | None, seed: int | None, sticky_maze: bool
) -> None:
    curriculum = CurriculumManager()
    env = build_env(curriculum, sticky_maze=sticky_maze)
    n_steps = 128
    model = _make_ppo(env, n_steps, load_path)
    if seed is not None:
        env.reset(seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(save_path)
    print(f"Modèle sauvegardé: {save_path}")


def run_visual(
    chunk: int,
    save_path: str,
    load_path: str | None,
    eval_only: bool,
    fast: bool,
    seed: int | None,
    gl_profile: str,
    view_mode: str,
    simple_shading: bool,
    sticky_maze: bool,
) -> None:
    from src.panda_gl_config import apply_panda_gl_profile

    apply_panda_gl_profile(gl_profile)

    from src.render_ursina import MazeRenderer

    from ursina import Entity, Ursina

    curriculum = CurriculumManager()
    env = build_env(curriculum, sticky_maze=sticky_maze)
    n_steps = max(64, chunk)
    model = _make_ppo(env, n_steps, load_path)

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    app = Ursina(title="IA labyrinthe — évasion 3D", vsync=True)
    renderer = MazeRenderer(
        app,
        view_mode=view_mode,
        simple_shading=simple_shading,
    )
    render_every = 4 if fast else 1
    train_cb = LiveRenderCallback(renderer, every=render_every)

    base = env.unwrapped
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    snap0 = base.snapshot_for_render()
    renderer.rebuild_if_needed(snap0)
    renderer.sync(snap0)

    if eval_only:

        class PlayLoop(Entity):
            def __init__(self):
                super().__init__(ignore=True)
                self.obs = obs

            def update(self):
                action, _ = model.predict(self.obs, deterministic=True)
                self.obs, _r, term, trunc, _info = env.step(int(action))
                renderer.sync(env.unwrapped.snapshot_for_render())
                if term or trunc:
                    self.obs, _ = env.reset()
                    renderer.rebuild_if_needed(env.unwrapped.snapshot_for_render())
                    renderer.sync(env.unwrapped.snapshot_for_render())

        PlayLoop()
    else:

        class TrainLoop(Entity):
            def update(self):
                model.learn(
                    total_timesteps=n_steps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                    callback=train_cb,
                )
                renderer.sync(env.unwrapped.snapshot_for_render())

        TrainLoop()

    app.run()

    if not eval_only:
        model.save(save_path)
        print(f"Modèle sauvegardé: {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="RL maze 3D — entraînement / visualisation")
    p.add_argument("--headless-train", action="store_true", help="Entraînement sans fenêtre")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--chunk", type=int, default=128, help="Horizon PPO par frame (taille rollout)")
    p.add_argument("--eval-only", action="store_true", help="Jouer sans mise à jour des poids")
    p.add_argument("--fast", action="store_true", help="Rafraîchissement rendu moins fréquent")
    p.add_argument("--model", type=str, default="ppo_maze.zip")
    p.add_argument("--load", type=str, default="", help="Charger un modèle existant")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--gl-profile",
        type=str,
        default="default",
        choices=sorted(VALID_GL_PROFILES),
        help="Profil OpenGL Panda3D avant Ursina (défaut: default — souvent le meilleur sur macOS)",
    )
    p.add_argument(
        "--view",
        type=str,
        default="carte",
        choices=("carte", "split", "suive"),
        help="carte=voyant du dessus (défaut), split=3D+dessus, suive=caméra derrière l'agent",
    )
    p.add_argument(
        "--single-view",
        action="store_true",
        help="Obsolète : équivalent à --view suive",
    )
    p.add_argument(
        "--simple-shading",
        action="store_true",
        help="Tente un rendu unlit (moins de shaders) si l'écran reste noir",
    )
    p.add_argument(
        "--web",
        action="store_true",
        help="Interface 2D dans le navigateur (Flask), sans Ursina / OpenGL",
    )
    p.add_argument("--web-host", type=str, default="127.0.0.1")
    p.add_argument("--web-port", type=int, default=8765)
    p.add_argument(
        "--web-step-delay",
        type=float,
        default=0.0,
        help="Pause en secondes entre chaque pas en mode --web --eval-only (0 = rapide max ; ex. 0.03 pour ralentir)",
    )
    p.add_argument(
        "--new-maze-each-episode",
        action="store_true",
        help="Régénérer le labyrinthe à chaque fin d'épisode (comportement ancien ; défaut : même labyrinthe jusqu'au succès)",
    )
    args = p.parse_args()

    load_path = args.load.strip() or None
    save_path = os.path.join(_ROOT, args.model)

    view_mode = args.view
    if args.single_view:
        view_mode = "suive"

    if args.headless_train:
        run_headless(
            args.timesteps,
            save_path,
            load_path,
            args.seed,
            sticky_maze=not args.new_maze_each_episode,
        )
    elif args.web:
        run_web(
            timesteps=args.timesteps,
            chunk=args.chunk,
            save_path=save_path,
            load_path=load_path,
            eval_only=args.eval_only,
            fast=args.fast,
            seed=args.seed,
            web_host=args.web_host,
            web_port=args.web_port,
            sticky_maze=not args.new_maze_each_episode,
            web_step_delay=max(0.0, args.web_step_delay),
        )
    else:
        run_visual(
            chunk=args.chunk,
            save_path=save_path,
            load_path=load_path,
            eval_only=args.eval_only,
            fast=args.fast,
            seed=args.seed,
            gl_profile=args.gl_profile,
            view_mode=view_mode,
            simple_shading=args.simple_shading,
            sticky_maze=not args.new_maze_each_episode,
        )


if __name__ == "__main__":
    main()
