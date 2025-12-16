from unittest import TestCase

from lmao.user_input import read_user_prompt


class UserInputTests(TestCase):
    def test_single_line(self) -> None:
        inputs = iter(["hello"])
        result = read_user_prompt("> ", input_fn=lambda _p: next(inputs))
        self.assertFalse(result.eof)
        self.assertEqual("hello", result.text)

    def test_single_line_auto_joins_pasted_lines(self) -> None:
        inputs = iter(["first"])
        result = read_user_prompt(
            "> ",
            input_fn=lambda _p: next(inputs),
            drain_available_lines_fn=lambda: ["second", "third"],
        )
        self.assertFalse(result.eof)
        self.assertEqual("first\nsecond\nthird", result.text)

    def test_sentinel_multiline(self) -> None:
        inputs = iter(['"""', "a", "b", '"""'])

        def fake_input(_p: str) -> str:
            return next(inputs)

        result = read_user_prompt("> ", input_fn=fake_input, sentinel='"""')
        self.assertFalse(result.eof)
        self.assertEqual("a\nb", result.text)

    def test_sentinel_multiline_eof_ends(self) -> None:
        seq = ['"""', "a", "b"]
        idx = 0

        def fake_input(_p: str) -> str:
            nonlocal idx
            if idx >= len(seq):
                raise EOFError()
            val = seq[idx]
            idx += 1
            return val

        result = read_user_prompt("> ", input_fn=fake_input, sentinel='"""')
        self.assertFalse(result.eof)
        self.assertEqual("a\nb", result.text)

    def test_multiline_default_reads_until_eof(self) -> None:
        seq = ["a", "b"]
        idx = 0

        def fake_input(_p: str) -> str:
            nonlocal idx
            if idx >= len(seq):
                raise EOFError()
            val = seq[idx]
            idx += 1
            return val

        result = read_user_prompt("> ", input_fn=fake_input, multiline_default=True)
        self.assertFalse(result.eof)
        self.assertEqual("a\nb", result.text)

    def test_multiline_default_eof_immediately(self) -> None:
        def fake_input(_p: str) -> str:
            raise EOFError()

        result = read_user_prompt("> ", input_fn=fake_input, multiline_default=True)
        self.assertTrue(result.eof)
        self.assertIsNone(result.text)
