"""release.py must say when the package files that build.sh copies differ from the commit."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from nnunet_inference.integration import release


def git(repository, *arguments):
    return subprocess.run(["git", "-C", str(repository), *arguments],
                          check=True, capture_output=True, text=True).stdout


class SourceDescriptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        self.package = self.repository / "nnunet_inference"
        (self.package / "tests").mkdir(parents=True)
        (self.package / "pacs.py").write_text("RELEASED = True\n")
        (self.repository / ".gitignore").write_text("local.json\n")
        git(self.repository, "init", "-q")
        git(self.repository, "add", ".")
        git(self.repository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", "release")
        self.commit = git(self.repository, "rev-parse", "--short=7", "HEAD").strip()

    def describe(self):
        return release.source_description(self.repository)

    def test_clean_tree_is_the_commit(self):
        self.assertEqual(self.describe(), self.commit)

    def test_modified_tracked_file_is_a_change(self):
        (self.package / "pacs.py").write_text("RELEASED = False\n")
        self.assertEqual(self.describe(), f"{self.commit} with uncommitted changes")

    def test_untracked_file_that_build_copies_is_a_change(self):
        # build.sh copies these whether or not Git ignores them.
        for name in ("extra.py", "extra.json", "extra.csv", "local.json"):
            path = self.package / name
            path.write_text("{}\n")
            with self.subTest(name=name):
                self.assertEqual(self.describe(), f"{self.commit} with uncommitted changes")
            path.unlink()

    def test_untracked_file_that_build_does_not_copy_is_not_a_change(self):
        for path in (self.repository / "notes.py", self.package / "notes.md",
                     self.package / "tests" / "test_extra.py"):
            path.write_text("\n")
        self.assertEqual(self.describe(), self.commit)


if __name__ == "__main__":
    unittest.main()
