import contextlib
import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import http_regression as benchmark
from dev.tests import smoke_real as smoke


class CharacterTokenizer:
    def encode(self, content, **_kwargs):
        return list(content)

    def decode(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


class HttpRegressionTests(unittest.TestCase):
    def test_any_repository_selects_matching_package_and_api_name(self):
        for selected in (
            "incoai/Qwen3.8-27B-Splash",
            "incoai/Qwen3.6-35B-A3B-Splash",
            "community/custom-splash",
            "mlx-community/Qwen3.8-27B-4bit",
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
        ):
            with self.subTest(model=selected):
                arguments = smoke.parse_args(["--model", selected])
                self.assertEqual(arguments.model, selected)
                self.assertEqual(
                    arguments.package,
                    smoke.model_artifacts.selection_link(
                        smoke.model_artifacts.MODELS, selected
                    ),
                )
        model = "incoai/Qwen3.8-27B-Splash"
        arguments = smoke.parse_args(["--package", "custom-package", "--model", model])
        self.assertEqual(str(arguments.package), "custom-package")
        self.assertEqual(arguments.model, model)

    def test_real_helpers_require_a_canonical_model(self):
        for parse in (smoke.parse_args, benchmark.parse_args):
            for arguments in ([], ["--model", "qwen3.8-27b"], ["--model", "custom"]):
                with (
                    self.subTest(parse=parse, arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(arguments)

    def test_benchmark_runs_any_installation_and_holds_its_assembly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary = root / "splash"
            binary.touch()
            (root / "splash.metallib").touch()
            models = root / "models"
            # A Splash package records manifest.json; an upstream selection
            # links an assembly that records model.json.
            legacy = "incoai/Qwen3.6-35B-A3B-Splash"
            (models / legacy).mkdir(parents=True)
            (models / legacy / "manifest.json").write_text("{}")
            upstream = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
            assembly = models / ".resolved/assembly"
            assembly.mkdir(parents=True)
            (assembly / "model.json").write_text("{}")
            (models / upstream).parent.mkdir(parents=True)
            (models / upstream).symlink_to(assembly, target_is_directory=True)

            def parse(model):
                with mock.patch.object(smoke.model_artifacts, "MODELS", models):
                    return benchmark.parse_args(
                        [
                            "--model",
                            model,
                            "--binary",
                            str(binary),
                            "--baseline-binary",
                            str(binary),
                        ]
                    )

            def hold(arguments):
                with mock.patch.object(smoke.model_artifacts, "MODELS", models):
                    smoke.hold_package(arguments)

            arguments = parse(legacy)
            self.assertEqual(arguments.package, models / legacy)
            hold(arguments)
            self.assertEqual(
                (arguments.package, arguments.held_record), (models / legacy, None)
            )
            # Parsing only names the selection link; the servers' run holds it.
            arguments = parse(upstream)
            self.assertEqual(arguments.package, models / upstream)
            self.assertFalse(smoke.assembly.is_held(assembly))
            hold(arguments)
            try:
                # Installations collect an unlinked assembly unless it is held.
                self.assertEqual(arguments.package, assembly)
                self.assertTrue(smoke.assembly.is_held(assembly))
            finally:
                arguments.held_record.close()
            self.assertFalse(smoke.assembly.is_held(assembly))
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                parse("community/not-installed")
            self.assertIn("missing installed model", error.getvalue())

    def row(self, version, sample=0, latency=10, round=None):
        return {
            "version": version,
            "round": {"baseline": 0, "candidate": 1}[version]
            if round is None
            else round,
            "sample": sample,
            "context": 2048,
            "scenario": "cold",
            "prompt_sha256": "prompt",
            "response_sha256": "answer",
            "usage": {"completion_tokens": 1},
            "metrics": {"request_latency": {"ttft_ms": latency}},
        }

    def test_exact_prompt_lengths_unique_prefix_and_reproducibility(self):
        tokenizer = CharacterTokenizer()
        prompts = benchmark.prompts(tokenizer, [256, 2048], 2, "nonce")
        self.assertEqual(prompts, benchmark.prompts(tokenizer, [256, 2048], 2, "nonce"))
        for (_sample, context), prompt in prompts.items():
            self.assertEqual(
                len(
                    tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
                ),
                context,
            )
        self.assertNotEqual(prompts[0, 256][:32], prompts[1, 256][:32])

    def abba_rows(self, latencies):
        """Samples 0 and 1 of both versions in rounds baseline, candidate,
        candidate, baseline with these per-round latencies."""
        return [
            self.row(version, sample, latency, round)
            for round, (version, sample, latency) in enumerate(
                zip(benchmark.ROUNDS, (0, 0, 1, 1), latencies)
            )
        ]

    def test_abba_rule_keeps_raw_timing_separate_from_correctness(self):
        for latencies, verdict in (
            ((10, 10.1, 10.1, 10), "pass"),
            ((10, 10.5, 10.5, 10), "fail"),
            ((9.8, 10.5, 10.5, 10.2), "pass"),
            ((9.6, 10, 10, 10.4), "inconclusive"),
        ):
            with self.subTest(latencies=latencies):
                summary = benchmark.summarize(self.abba_rows(latencies))[0]
                self.assertEqual(summary["verdict"], verdict)
                self.assertEqual(summary["pass"], verdict == "pass")
                self.assertEqual(summary["rounds"], list(latencies))
                self.assertEqual(summary["samples_per_round"], [1, 1, 1, 1])
        rows = self.abba_rows((10, 10, 10, 10))
        with self.assertRaisesRegex(ValueError, "no samples"):
            benchmark.summarize(rows[:3] + [{**rows[3], "round": 0}])

    def test_layout_identity_is_compared_only_where_keys_must_agree(self):
        def status(build, layout, kv="int8"):
            return {
                "identity": {
                    "cache": {"build_id": build, "loaded_model_layout_sha256": layout},
                    "kv": {"format": kv},
                }
            }

        baseline = {"version": "baseline", **status("b", "layout-b")}
        # Another preparation identity prepares under other keys.
        benchmark.check_identity(
            status("c", "layout-c"), "candidate", [baseline], False
        )
        for shared, current, version in (
            (True, status("c", "layout-c"), "candidate"),
            (False, status("b", "layout-c"), "baseline"),
            (False, status("b2", "layout-b"), "baseline"),
            (False, status("c", "layout-c", "bf16"), "candidate"),
        ):
            with (
                self.subTest(shared=shared, current=current, version=version),
                self.assertRaises(smoke.SmokeFailure),
            ):
                benchmark.check_identity(current, version, [baseline], shared)
        benchmark.check_identity(status("c", "layout-b"), "candidate", [baseline], True)

    def test_missing_duplicate_or_changed_transcript_fails(self):
        baseline, candidate = self.row("baseline"), self.row("candidate")
        for rows in ([baseline], [baseline, baseline, candidate]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows)
        for field in ("response_sha256", "prompt_sha256"):
            changed = copy.deepcopy(candidate)
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "transcript differs"):
                benchmark.summarize([baseline, changed])

    def test_decode_uses_native_decode_time_per_token(self):
        rows = self.abba_rows((10, 10, 10, 10))
        for row in rows:
            row["scenario"] = "decode"
            row["native_delta"] = {"decode_wall_ms": 100, "decode_output_tokens": 50}
        summary = benchmark.summarize(rows)[0]
        self.assertEqual(summary["metric"], "decode_ms_per_token")
        self.assertEqual(summary["baseline_median"], 2)


if __name__ == "__main__":
    unittest.main()
