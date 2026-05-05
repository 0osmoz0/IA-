# IA labyrinthe 3D (RL + Ursina)

Entraînement **PPO** (Stable-Baselines3) pour qu’un agent sorte de labyrinthes générés au hasard, avec **curriculum** (niveaux A→E : taille, mur mobile, clé/porte, pièges, portail) et rendu **3D live** via **Ursina**. Tant que l’agent n’atteint pas la sortie, **le même labyrinthe est conservé** (nouvelle grille seulement après un succès ou si vous passez `--new-maze-each-episode`).

**Affichage :** par défaut (`--view carte`), une **vue carte du dessus** (terrain lisible comme un plateau). Modes : `--view suive` (caméra 3D sur l’agent), `--view split` (moitié 3D + moitié carte).

## Installation

```bash
cd IA-
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer

- **Entraînement + fenêtre 3D** (chaque frame ≈ une mise à jour PPO sur un rollout de `--chunk` pas) :

```bash
python -m src.train_live --chunk 128
```

- **Sans fenêtre** (rapide, bon pour pré-entraîner un modèle) :

```bash
python -m src.train_live --headless-train --timesteps 200000
```

- **Observer un modèle sans apprentissage** :

```bash
python -m src.train_live --eval-only --load ppo_maze.zip
```

- **Vue 2D dans le navigateur** (aucune fenêtre OpenGL / Ursina — pratique si l’écran 3D reste noir sur macOS) :

```bash
pip install -r requirements.txt   # inclut flask
python -m src.train_live --web --eval-only --load ppo_maze.zip
```

Puis ouvrez [http://127.0.0.1:8765/](http://127.0.0.1:8765/) (port modifiable avec `--web-port`). Entraînement avec la même interface : `python -m src.train_live --web --timesteps 50000`. En évaluation, `--web-step-delay 0` (défaut) envoie les pas à fond ; utilisez par ex. `0.04` pour ralentir.

- **Options utiles** : `--view carte|suive|split`, `--fast`, `--seed N`, `--model …`, `--gl-profile …`, `--simple-shading` (voir ci-dessous), `--new-maze-each-episode` (régénérer le labyrinthe à chaque épisode, comme avant). L’ancien `--single-view` force `--view suive`.

Le modèle est sauvegardé dans le fichier indiqué par `--model` (défaut : `ppo_maze.zip` à la racine du dépôt) quand vous fermez la fenêtre en mode entraînement visuel.

## Écran noir / erreurs GLSL (surtout macOS)

Si la fenêtre est **noire** mais l’overlay Ursina (FPS, nombre d’entités) s’affiche, le moteur tourne mais les **shaders** ne se compilent pas (souvent `version '130' is not supported`, `#version required and missing`, etc.).

Le projet applique aussi un **shader unlit** aux entités du labyrinthe : d’abord **GLSL 150**, puis repli **GLSL 120** si la machine ne l’accepte pas (voir [`src/maze_glsl.py`](src/maze_glsl.py)).
   - `python -m src.train_live --simple-shading` (rendu plus simple, moins de dépendance aux shaders d’éclairage)
   - `python -m src.train_live --single-view` (élimine le découpage deux caméras)
   - `python -m src.train_live --gl-profile legacy21` (expérimental)
   - `python -m src.train_live --gl-profile core33` (peut aider dans de rares cas, ou empirer l’écran noir)

Profils disponibles : `default`, `core32`, `core33`, `legacy21` — configurés dans [`src/panda_gl_config.py`](src/panda_gl_config.py), appliqués **avant** `Ursina()` dans [`src/train_live.py`](src/train_live.py).

En dernier recours : **`python -m src.train_live --web`** pour suivre le labyrinthe dans le navigateur (sans OpenGL), ou `--headless-train` pour l’entraînement sans aucune fenêtre.

## Structure

| Fichier | Rôle |
|---------|------|
| [src/maze_glsl.py](src/maze_glsl.py) | Shader unlit GLSL 150 (macOS) |
| [src/maze_env.py](src/maze_env.py) | `gym.Env` + observations + récompenses |
| [src/mechanics.py](src/mechanics.py) | Données des mécaniques |
| [src/curriculum.py](src/curriculum.py) | Niveaux et promotion |
| [src/render_ursina.py](src/render_ursina.py) | Scène 3D |
| [src/train_live.py](src/train_live.py) | Boucle PPO + CLI |
| [src/web_viewer.py](src/web_viewer.py) | Serveur Flask + canvas 2D |
