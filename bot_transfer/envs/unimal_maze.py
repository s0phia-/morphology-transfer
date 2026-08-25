"""Unimal walkers in the graph_transformer maze, as a bot_transfer environment.

Lets this repo's method (Hierarchically Decoupled Imitation for Morphological Transfer)
run as a baseline against exactly the environment graph_transformer's own agents train
on, rather than an approximation of it.

    # one low level per morphology, imitating a source low level
    python scripts/train.py --alg DSAC --env MazeEnd_Unimal \\
        --env-args walker floor-1409-0-3-01-15-56-55 \\
        --env-wrapper L2Low ...

    # one high level, reused across every morphology
    python scripts/train.py --alg SAC --env MazeEnd_Unimal \\
        --env-wrapper High --low-level <trained low level> ...

WHERE THE MODEL FILES COME FROM
Not built here. graph_transformer's utils/export_maze_xml.py composes them with its own
code - walker, floor and maze walls - and writes one <walker>.xml per morphology plus a
manifest.json into assets/unimal_umaze/. Reproducing its maze construction in this repo
would be the same work plus a guarantee of drifting away from it; loading its output
means the physics is identical by construction. Regenerate whenever the maze changes.

WHAT DIFFERS FROM THIS REPO'S OWN MAZE ENVS, AND WHY
* The goal region is a RADIUS (manifest goal_tolerance, 0.45) around the goal cell's
  centre, not the whole cell. bot_transfer's Maze.step rewards being anywhere inside the
  goal cell, which at this cell size is a 4x4 box - roughly 60x the area. The tighter
  test is what graph_transformer's agents are scored on, so it is what a comparable
  baseline has to use. If a low level plateaus at "hovering near the right cell", this
  constant is the first thing to loosen.
* AGENT_DIM is computed per morphology at construction instead of being a class
  constant. Unimals vary in limb count, so no single number exists - this is the whole
  reason the method needs one low level per morphology.
* The torso body is "torso/0", not "torso".
* No maze construction in __init__ - the walls are already in the XML. This subclasses
  base.Env directly rather than maze.Maze for that reason.
"""
import json
import os

import numpy as np

from .base import Env

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "unimal_umaze")


def _load_manifest(asset_dir):
    path = os.path.join(asset_dir, "manifest.json")
    if not os.path.exists(path):
        raise IOError(
            "No manifest.json in {} - generate the assets first from the "
            "graph_transformer repo:\n"
            "  python utils/export_maze_xml.py --cfg <config> --out-dir {}".format(
                asset_dir, asset_dir))
    with open(path) as f:
        return json.load(f)


class MazeEnd_Unimal(Env):
    """One unimal walker in the graph_transformer maze.

    Parameterized by walker name rather than subclassed per morphology: this repo's
    loader already calls env_cls(**params['env_args']), so `--env-args walker <name>`
    selects one of the 100 without needing 100 generated classes.
    """

    # X, Y in the plane - the shared skill space every morphology's low level is
    # trained against, and the only thing the high level ever emits. Identical to this
    # repo's own Maze envs, which is what allows one high level to drive all of them.
    SKILL_DIM = 2
    # skill components (2) + goal position (2), per base.py's layout convention.
    TASK_DIM = 4
    # graph_transformer's UnimalEnv uses 4 (envs/tasks/unimal.py) with the same
    # ctrl[:] = action application, so the control rate matches exactly.
    FRAME_SKIP = 4

    # Size of the root joint in qpos/qvel, i.e. how much of the state is the agent's
    # pose in the world rather than its own configuration. Unimals have a free joint
    # (3 position + 4 quaternion, 6 velocities); PointMass has two slide joints.
    # reset() adds its noise past this boundary so the walker always starts at the cell
    # centre in a canonical pose rather than jittered across it.
    ROOT_QPOS = 7
    ROOT_QVEL = 6

    RANDOM_GOALS = False
    SPARSE_REWARD = 100.0
    DIST_REWARD = 0.0
    VISUALIZE = True
    # Non-None pins the class to one asset regardless of --env-args, for agents that
    # are not one of the exported unimals (see MazeEnd_PointMass_UMaze).
    WALKER_DEFAULT = None

    def __init__(self, walker=None, asset_dir=None, goal_tolerance=None):
        self.asset_dir = asset_dir or ASSET_DIR
        self.manifest = _load_manifest(self.asset_dir)

        if walker is None:
            walker = self.WALKER_DEFAULT or self.manifest["walkers"][0]
        if self.WALKER_DEFAULT is None and walker not in self.manifest["walkers"]:
            raise ValueError("walker {!r} is not in the exported set ({} available, "
                             "e.g. {})".format(walker, len(self.manifest["walkers"]),
                                               self.manifest["walkers"][0]))
        self.walker = walker

        self.nrows = self.manifest["nrows"]
        self.ncols = self.manifest["ncols"]
        self.cell_size = self.manifest["cell_size"]
        self.goal_tolerance = (goal_tolerance if goal_tolerance is not None
                               else self.manifest["goal_tolerance"])
        self.goal_cells = [tuple(c) for c in self.manifest["goal_cells"]]
        self.reset_cells = [tuple(c) for c in self.manifest["reset_cells"]]
        self.floor_top_z = self.manifest["floor_top_z"]
        self.torso_body = self.manifest["torso_body"]

        # Fixed until reset(); seeded RNG does not exist yet at this point (base.Env
        # calls seed() at the end of its own __init__), so the first goal and spawn are
        # deterministic and reset() immediately replaces them.
        self._goal_cell = self.goal_cells[0]
        self._spawn_cell = self.reset_cells[0]
        self.center_goal = np.array(self.grid_to_xy(*self._goal_cell))

        model_path = os.path.join(self.asset_dir, "{}.xml".format(walker))
        # base.Env.__init__ builds the sim, then calls step() once with a sampled
        # action to infer the observation space - which is why every attribute
        # _get_obs and step touch has to be set above this line.
        super(MazeEnd_Unimal, self).__init__(model_path=model_path,
                                             frame_skip=self.FRAME_SKIP)

        # Only knowable once the model exists, and only consulted by agent_obs /
        # skill_obs afterwards, never during __init__. Everything before the trailing
        # (skill, goal) block is agent-specific; unimals differ in limb count, so this
        # is per morphology by necessity.
        self.AGENT_DIM = len(self._get_obs()) - self.SKILL_DIM - 2

    # ------------------------------------------------------------------ #
    # Maze geometry. Reproduced from the manifest rather than imported, so this file
    # has no dependency on graph_transformer - see Maze.grid_to_xy over there.
    # ------------------------------------------------------------------ #
    def grid_to_xy(self, row, col):
        x = (col - (self.ncols - 1) / 2.0) * self.cell_size
        y = ((self.nrows - 1) / 2.0 - row) * self.cell_size
        return x, y

    def torso_xy(self):
        return self.get_body_com(self.torso_body)[:2]

    def sample_goal_pos(self):
        """Pick this episode's goal cell, never the one the agent is spawning in.

        graph_transformer excludes the agent's own cell for the same reason: both it and
        the spawn return the exact cell centre, so a goal sampled onto the spawn cell
        starts already reached and pays out for nothing. Without the exclusion that is
        1 episode in 9 on this map.
        """
        if not self.RANDOM_GOALS:
            return
        candidates = [c for c in self.goal_cells if c != self._spawn_cell]
        if not candidates:
            raise ValueError("no goal cell other than the spawn cell {}".format(
                self._spawn_cell))
        self._goal_cell = candidates[self.np_random.randint(len(candidates))]
        self.center_goal = np.array(self.grid_to_xy(*self._goal_cell))

    def sample_spawn_pos(self):
        return self.reset_cells[self.np_random.randint(len(self.reset_cells))]

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        """[agent state | torso xy | goal xy], in base.py's documented layout.

        qpos[2:] drops the root x/y so the agent-specific block is translation
        invariant - the same reason this repo's own Ant maze env slices it that way,
        and what lets a low level generalize across the maze rather than memorizing
        positions. The absolute torso xy still reaches the policy, but as the SKILL
        block, which is what the hierarchy is defined over.
        """
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            self.torso_xy(),
            self.center_goal,
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        # RADIUS, not the enclosing cell - see this module's docstring.
        dist = np.linalg.norm(self.torso_xy() - self.center_goal)
        done = bool(dist <= self.goal_tolerance)
        reward = self.SPARSE_REWARD if done else 0.0
        if self.DIST_REWARD > 0:
            reward += -self.DIST_REWARD * dist
        return obs, reward, done, {"is_success": done, "dist_to_goal": float(dist)}

    def reset(self):
        self.sim.reset()
        self._spawn_cell = self.sample_spawn_pos()
        self.sample_goal_pos()
        if self.VISUALIZE:
            # body_pos[-2] is the goal marker, [-1] the skill marker - the positional
            # convention base.Env.display_skill and this repo's Maze envs both use.
            # export_maze_xml.py appends the two bodies in that order for exactly this.
            self.model.body_pos[-2][:2] = self.center_goal

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        spawn = self.grid_to_xy(*self._spawn_cell)
        qpos[0:2] = spawn
        # Matches graph_transformer's MazeTask.reset_model: small uniform noise on the
        # agent's own joints, and none on the root pose, so the walker always starts at
        # the cell centre rather than jittered across it.
        qpos[self.ROOT_QPOS:] += self.np_random.uniform(
            low=-0.005, high=0.005, size=len(qpos) - self.ROOT_QPOS)
        qvel[self.ROOT_QVEL:] += self.np_random.uniform(
            low=-0.005, high=0.005, size=len(qvel) - self.ROOT_QVEL)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Top-down over the whole maze. The inherited chase camera sits ~3 units from
        the torso, close enough that the goal marker is routinely out of frame.
        """
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])
        self.viewer.cam.distance = max(self.nrows, self.ncols) * self.cell_size * 1.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90


class MazeSample_Unimal(MazeEnd_Unimal):
    """Goal resampled every episode instead of fixed - the setting the high level is
    trained and evaluated under, matching MazeSample_Ant's relationship to MazeEnd_Ant.
    """
    RANDOM_GOALS = True


class MazeEnd_PointMass_UMaze(MazeEnd_Unimal):
    """bot_transfer's PointMass, in the same maze as the unimals.

    The source agent for the imitation baseline. Its low level is trained here first,
    and every unimal's low level is then trained to be indistinguishable from it in
    skill space (DSAC). That argument only holds if source and target share a skill
    space, so the PointMass has to be in OUR maze at OUR scale - this repo's own
    MazeEnd_PointMass lives in bot_transfer's maze (construct_maze, SCALING 8.0) and
    would give the imitation a target defined in different coordinates.

    The asset is generated by graph_transformer's utils/export_maze_xml.py with
    --pointmass-asset, alongside the unimal files and through the same code path, so
    only the agent body differs between them.

    Nothing about the observation needs overriding: the inherited _get_obs slices
    qpos[2:], which for two slide joints is empty, leaving [qvel, torso_xy, goal] and
    AGENT_DIM 2 - exactly this repo's own MazeEnd_PointMass layout.
    """

    WALKER_DEFAULT = "pointmass"
    # Two slide joints, no free joint.
    ROOT_QPOS = 2
    ROOT_QVEL = 2
    # bot_transfer's own MazeEnd_PointMass value. Its asset also uses a 0.02 timestep
    # against the unimals' 0.005, so this is not the same amount of simulated time -
    # which is fine and expected, and is why delta_max is measured in DISTANCE.
    FRAME_SKIP = 3

    def __init__(self, walker=None, asset_dir=None, goal_tolerance=None):
        super(MazeEnd_PointMass_UMaze, self).__init__(
            walker=walker, asset_dir=asset_dir, goal_tolerance=goal_tolerance)
        # The exported asset keeps bot_transfer's own body name.
        self.torso_body = "torso"


class MazeSample_PointMass_UMaze(MazeEnd_PointMass_UMaze):
    """Goal resampled every episode - what the high level is trained against."""
    RANDOM_GOALS = True
