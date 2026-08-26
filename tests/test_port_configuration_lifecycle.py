import ast
import unittest
from pathlib import Path


class PortConfigurationLifecycleTests(unittest.TestCase):
    def test_port_configuration_is_committed_before_hud_wait(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "battle_queued"
                for target in node.targets
            )
        ]
        self.assertTrue(assignments)
        self.assertIn("if battle_queued and configured_this_attempt", source)
        self.assertIn("port_configured = True", source)
        self.assertIn("prepared = battle_queued and wait_for_battle", source)


if __name__ == "__main__":
    unittest.main()
