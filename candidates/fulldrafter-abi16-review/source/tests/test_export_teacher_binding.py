"""CPU-only export regression; tiny fixtures, no production checkpoint writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location("exporter", Path(__file__).parents[1] / "scripts/export_fulldrafter_sidecar.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

class ExportBindingTest(unittest.TestCase):
    def test_exact_tensor_contract(self):
        import math
        shapes = e.expected_tensor_shapes()
        self.assertEqual(len(shapes), e.EXPECTED_TENSOR_COUNT)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), e.EXPECTED_PARAM_COUNT)
        tensors = {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        e.validate_tensor_shapes(tensors)
        # Same element count and bytes, different geometry: aggregate checks miss this.
        tensors["drafter.context_proj.weight"] = torch.empty((7680, 768), device="meta")
        with self.assertRaisesRegex(e.SidecarValidationError, "Invalid tensor shape"):
            e.validate_tensor_shapes(tensors)
        tensors = {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        tensors["unexpected.weight"] = tensors.pop("drafter.mask_vector")
        with self.assertRaisesRegex(e.SidecarValidationError, "Tensor names"):
            e.validate_tensor_shapes(tensors)

    def test_teacher_conflict_and_manifest_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            checkpoint = d / "checkpoint.pt"
            contract = d / "contract.json"
            output = d / "step.safetensors"
            shared = d / "sidecar_manifest.json"
            shared.write_text("unchanged")
            teacher = {"target_embedding_sha256": "a" * 64, "target_head_sha256": "b" * 64}
            torch.save({"model_state_dict": {"drafter.mask_vector": torch.zeros(2)}, "teacher": teacher}, checkpoint)
            bad = dict(teacher, target_head_sha256="c" * 64)
            contract.write_text(json.dumps({"target_hashes": bad}))
            with self.assertRaisesRegex(e.SidecarValidationError, "Teacher binding mismatch"):
                e.export_fulldrafter_sidecar(checkpoint, output, contract, strict=False)
            self.assertFalse(output.exists())
            contract.write_text(json.dumps({"target_hashes": teacher}))
            _, manifest, data = e.export_fulldrafter_sidecar(checkpoint, output, contract, strict=False)
            self.assertEqual(manifest, output.with_suffix(".manifest.json"))
            self.assertEqual(shared.read_text(), "unchanged")
            self.assertEqual(data["safetensors_file_sha256"], e.compute_file_sha256(output))
            self.assertEqual(data["target_final_norm_sha256"], "")

if __name__ == "__main__":
    unittest.main()
