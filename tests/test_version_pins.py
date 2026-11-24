"""Package versions are written in four places; they must agree."""
from pathlib import Path
import re
import tomllib
import unittest

from nnunet_inference import deployment

REPOSITORY = Path(__file__).resolve().parents[1]


def pins(text):
    """{'package': 'version'} for every 'package==version' in the text."""
    return dict(re.findall(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9.+]*)", text))


class VersionPinTests(unittest.TestCase):
    def setUp(self):
        self.container = pins((REPOSITORY / "nnunet_inference/integration/requirements.yml").read_text())
        project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
        self.project = pins("\n".join(project["project"]["dependencies"]))
        self.readme = pins((REPOSITORY / "README.md").read_text())

    def test_container_environment_has_the_versions_the_bundle_records(self):
        for package, version in deployment.RUNTIME.items():
            self.assertEqual(self.container.get(package), version, package)

    def test_project_pins_equal_the_container_environment(self):
        self.assertTrue(self.project)
        for package, version in self.project.items():
            self.assertEqual(self.container.get(package), version, package)
        # torch is the one container pin that is left to the user's hardware.
        self.assertEqual(set(self.container) - set(self.project), {"torch"})

    def test_readme_install_command_equals_the_project_pins(self):
        # fastMONAI is installed apart from the project, for the evaluation only.
        self.readme.pop("fastMONAI", None)
        self.assertEqual(self.readme, self.project)


if __name__ == "__main__":
    unittest.main()
