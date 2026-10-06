import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("main.py")
SPEC = importlib.util.spec_from_file_location("orbit_wars_agent", MODULE_PATH)
bot = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = bot
SPEC.loader.exec_module(bot)


def _obs(planets, fleets=None, player=0, angular_velocity=0.035, **extra):
    payload = {
        "player": player,
        "planets": planets,
        "fleets": fleets or [],
        "angular_velocity": angular_velocity,
        "initial_planets": planets,
        "comets": [],
        "comet_planet_ids": [],
    }
    payload.update(extra)
    return payload


def _assert_valid_moves(testcase, obs, moves):
    ships_by_planet = {int(p[0]): int(p[5]) for p in obs["planets"] if int(p[1]) == obs["player"]}
    launched = {}
    for move in moves:
        testcase.assertEqual(len(move), 3)
        source, angle, ships = move
        testcase.assertIn(source, ships_by_planet)
        testcase.assertTrue(math.isfinite(angle))
        testcase.assertIsInstance(ships, int)
        testcase.assertGreater(ships, 0)
        launched[source] = launched.get(source, 0) + ships
        testcase.assertLessEqual(launched[source], ships_by_planet[source])


class OrbitWarsAgentTests(unittest.TestCase):
    def setUp(self):
        bot.STATE["games"] = {}

    def test_expands_to_good_neutral(self):
        obs = _obs(
            [
                [1, 0, 15.0, 15.0, 2.0, 45, 3],
                [2, -1, 23.0, 18.0, 2.6, 8, 5],
                [3, -1, 80.0, 80.0, 1.7, 45, 2],
            ]
        )
        moves = bot.agent(obs)
        _assert_valid_moves(self, obs, moves)
        self.assertTrue(moves)
        self.assertEqual(moves[0][0], 1)

    def test_avoids_direct_sun_crossing(self):
        obs = _obs(
            [
                [1, 0, 2.0, 50.0, 2.0, 80, 4],
                [2, -1, 98.0, 50.0, 2.0, 6, 5],
            ]
        )
        moves = bot.agent(obs)
        _assert_valid_moves(self, obs, moves)
        self.assertEqual(moves, [])

    def test_reinforces_planet_under_incoming_attack(self):
        obs = _obs(
            [
                [1, 0, 20.0, 20.0, 2.0, 10, 2],
                [2, 0, 20.0, 35.0, 2.0, 55, 3],
                [3, 1, 80.0, 80.0, 2.0, 25, 4],
            ],
            fleets=[
                [100, 1, 27.0, 20.0, math.pi, 3, 35],
            ],
        )
        moves = bot.agent(obs)
        _assert_valid_moves(self, obs, moves)
        self.assertTrue(any(move[0] == 2 for move in moves))

    def test_handles_comet_path_observation(self):
        path = [[10.0 + i, 30.0] for i in range(80)]
        obs = _obs(
            [
                [1, 0, 12.0, 20.0, 2.0, 60, 4],
                [9, -1, 16.0, 30.0, 1.0, 5, 1],
            ],
            comets=[{"planet_ids": [9], "paths": [path], "path_index": 6}],
            comet_planet_ids=[9],
        )
        moves = bot.agent(obs)
        _assert_valid_moves(self, obs, moves)

    def test_keeps_turn_state_per_player(self):
        planets = [
            [1, 0, 15.0, 15.0, 2.0, 45, 3],
            [2, 1, 85.0, 85.0, 2.0, 45, 3],
            [3, -1, 25.0, 17.0, 2.0, 8, 4],
            [4, -1, 75.0, 83.0, 2.0, 8, 4],
        ]
        bot.agent(_obs(planets, player=0))
        bot.agent(_obs(planets, player=1))
        bot.agent(_obs(planets, player=0))
        self.assertEqual(bot.STATE["games"][0]["turn"], 2)
        self.assertEqual(bot.STATE["games"][1]["turn"], 1)

    def test_contested_neutral_requires_outbidding_enemy_fleet(self):
        planets = [
            [1, 0, 15.0, 15.0, 2.0, 70, 4],
            [2, -1, 26.0, 16.0, 2.0, 10, 5],
            [3, 1, 38.0, 16.0, 2.0, 30, 4],
        ]
        obs = _obs(
            planets,
            fleets=[
                [200, 1, 32.0, 16.0, math.pi, 3, 28],
            ],
        )
        moves = bot.agent(obs)
        _assert_valid_moves(self, obs, moves)
        if moves:
            self.assertGreaterEqual(moves[0][2], 30)

    def test_counter_profile_detects_passive_mirror(self):
        planets = [
            [1, 0, 15.0, 15.0, 2.0, 500, 4],
            [2, 1, 85.0, 85.0, 2.0, 500, 4],
            [3, -1, 45.0, 18.0, 2.0, 20, 3],
        ]
        bot.agent(
            _obs(
                planets,
                fleets=[[500, 1, 80.0, 85.0, math.pi, 2, 40]],
                step=160,
            )
        )
        bot.agent(_obs(planets, fleets=[], step=220))
        profile = bot.STATE["games"][0]["last_counter_profile"]
        self.assertTrue(profile["passive_mirror"])
        self.assertGreaterEqual(profile["quiet_turns"], 30)


if __name__ == "__main__":
    unittest.main()
