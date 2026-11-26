"""Run with: python -m unittest test_chat_client -v"""
import tempfile
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace as NS

from ollama_chat_matthew import ChatClient


def chunk(text="", done=False, **kw):
    return NS(message=NS(content=text), done=done, **kw)


class FakeClient:
    def __init__(self, chunks=(), raises=None, fail_after=None):
        self.chunks, self.raises, self.fail_after = list(chunks), raises, fail_after
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self._gen()

    def _gen(self):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("boom")
            yield c


FINAL = chunk("", done=True, prompt_eval_count=10, eval_count=5, eval_duration=1_000_000_000)


class ChatClientTests(unittest.TestCase):
    def run_reply(self, client):
        return list(client.stream_reply(threading.Event()))

    def test_stream_collects_reply_and_real_stats(self):
        c = ChatClient(client=FakeClient([chunk("Hel"), chunk("lo"), FINAL]))
        c.add_user("hi")
        events = self.run_reply(c)
        self.assertEqual("".join(e.text for e in events if e.kind == "chunk"), "Hello")
        self.assertEqual(c.history[-1]["content"], "Hello")
        self.assertEqual(c.last_stats.tokens_per_sec, 5.0)
        self.assertEqual(c.context_used, 15)
        self.assertEqual(c.session_tokens, 5)

    def test_no_options_sent_and_clean_messages(self):
        fake = FakeClient([chunk("ok"), FINAL])
        c = ChatClient(client=fake)
        c.add_user("hi")
        self.run_reply(c)
        call = fake.calls[0]
        self.assertNotIn("options", call)  # Modelfile stays in charge
        self.assertEqual(call["messages"], [{"role": "user", "content": "hi"}])

    def test_context_limit_read_from_modelfile(self):
        fake = FakeClient()
        fake.show = lambda model: NS(parameters="temperature   0.7\nnum_ctx   32768")
        c = ChatClient(client=fake)
        self.assertEqual(c.refresh_context_limit(), 32768)
        fake.show = lambda model: NS(parameters="temperature   0.7")
        self.assertIsNone(c.refresh_context_limit())

    def test_no_trim_when_limit_unknown(self):
        c = ChatClient(client=FakeClient())
        for i in range(5):
            c.history.append({"role": "user", "content": "x", "tokens": 10**6})
        c.trim_history()
        self.assertEqual(len(c.history), 5)

    def test_connection_error_message(self):
        c = ChatClient(client=FakeClient(raises=ConnectionError("x")))
        c.add_user("hi")
        events = self.run_reply(c)
        self.assertEqual(events[0].kind, "error")
        self.assertIn("Ollama", events[0].text)
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(c.history[-1]["role"], "user")  # nothing saved, can retry

    def test_partial_reply_saved_on_midstream_error(self):
        c = ChatClient(client=FakeClient([chunk("par"), chunk("tial"), FINAL], fail_after=1))
        c.add_user("hi")
        self.run_reply(c)
        self.assertEqual(c.history[-1]["content"], "par")

    def test_cancel_keeps_partial(self):
        cancel = threading.Event()
        c = ChatClient(client=FakeClient([chunk("a"), chunk("b"), FINAL]))
        c.add_user("hi")
        gen = c.stream_reply(cancel)
        next(gen)
        cancel.set()
        events = list(gen)
        self.assertTrue(events[-1].cancelled)
        self.assertEqual(c.history[-1]["content"], "a")

    def test_trim_drops_oldest_and_starts_on_user(self):
        c = ChatClient(context_limit=100, client=FakeClient())
        for i in range(10):
            c.history.append({"role": "user", "content": f"q{i}", "tokens": 20})
            c.history.append({"role": "assistant", "content": f"a{i}", "tokens": 20})
        c.add_user("latest")
        c.trim_history()
        self.assertEqual(c.history[0]["role"], "user")
        self.assertEqual(c.history[-1]["content"], "latest")
        self.assertLessEqual(sum(m["tokens"] for m in c.history), 100)

    def test_pop_last_assistant(self):
        c = ChatClient(client=FakeClient())
        c.add_user("hi")
        self.assertFalse(c.pop_last_assistant())
        c.history.append({"role": "assistant", "content": "yo", "tokens": 1})
        self.assertTrue(c.pop_last_assistant())
        self.assertEqual(len(c.history), 1)

    def test_save_load_roundtrip_and_bad_role(self):
        c = ChatClient(client=FakeClient())
        c.add_user("hi")
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "chat.json")
            c.save(p)
            c2 = ChatClient(client=FakeClient())
            c2.load(p)
            self.assertEqual(c2.history, c.history)
            with open(p, "w") as f:
                f.write('{"history": [{"role": "system", "content": "x"}]}')
            with self.assertRaises(ValueError):
                c2.load(p)


if __name__ == "__main__":
    unittest.main()