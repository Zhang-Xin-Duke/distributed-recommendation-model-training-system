import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("bundle", Path(__file__).parents[1] / "scripts/original_native_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def test_materialized_inputs_survive_moving_and_removing_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "workspace"
            dep = root / "original-repo"
            ws.mkdir(); dep.mkdir()
            (ws / ".resources").mkdir()
            (ws / ".resources/rule.bzl").write_text("# rule\n")
            (ws / "WORKSPACE").write_text(f'workspace(name="test")\nlocal_repository(\n name="dep", path={str(dep)!r},\n)\n')
            (dep / "BUILD").write_text('exports_files(["input.h"])\n')
            (root / "real.h").write_bytes(b"real bytes\x00")
            (dep / "input.h").symlink_to(root / "real.h")
            (ws / "entropy").mkdir()
            (ws / "entropy/train.py").write_text("# complete framework entry point\n")
            deps = root / "deps"; metadata = root / "metadata"
            deps.write_text("@dep//:input.h\n@dep//:generated.so\n")
            metadata.write_text("@dep//:BUILD\n//.resources:rule.bzl\n")
            result = bundle.build_bundle(ws, deps, metadata, root / "out")
            (root / "out").rename(root / "moved")
            (root / "real.h").unlink()
            moved = root / "moved"
            self.assertEqual((moved / "repositories/dep/input.h").read_bytes(), b"real bytes\x00")
            self.assertFalse((moved / "repositories/dep/input.h").is_symlink())
            self.assertEqual(bundle.repositories(moved)["dep"], (moved / "repositories/dep").resolve())
            self.assertTrue((moved / "entropy/train.py").is_file())
            self.assertEqual(result["dependency_labels_without_local_file"], ["@dep//:generated.so"])
            self.assertFalse(result["native_build_completed"])
            (root / "sources").write_text("@dep//:input.h\n")
            audited = bundle.audit_bundle(moved, root / "sources")
            self.assertEqual(audited["required_host_inputs"], [])
            (root / "sources").write_text("@dep//:missing.cc\n")
            with self.assertRaisesRegex(FileNotFoundError, "missing source input"):
                bundle.audit_bundle(moved, root / "sources")
            (root / "sources").write_text("@dep//:input.h\n")
            (moved / "repositories/dep/input.h").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                bundle.audit_bundle(moved, root / "sources")

    def test_missing_build_metadata_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ws = root / "ws"; ws.mkdir()
            (ws / "WORKSPACE").write_text('workspace(name="test")\n')
            (root / "deps").write_text("")
            (root / "metadata").write_text("//missing:BUILD\n")
            with self.assertRaisesRegex(FileNotFoundError, "missing build metadata"):
                bundle.build_bundle(ws, root / "deps", root / "metadata", root / "out", ())

    def test_label_paths_cannot_escape_repository(self):
        for package, name in (("..", "file"), ("pkg", "../file"), ("/tmp", "file"), ("", "/file")):
            with self.subTest(package=package, name=name):
                with self.assertRaises(ValueError):
                    bundle.relative_file(package, name)


if __name__ == "__main__":
    unittest.main()
