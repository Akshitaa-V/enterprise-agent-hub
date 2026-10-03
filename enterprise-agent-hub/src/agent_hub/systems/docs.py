"""Document knowledge base: splits Markdown policies into sections and ranks them with BM25.

BM25 keeps the demo dependency-free and fully explainable. The DocumentStore
interface (search -> list of chunks with a source id) is what the agents rely
on, so a vector store can replace it without touching the agent code.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set(
    "a an the and or of to in on for is are be can do does how what which who when within with by at as it "
    "this that our we i my me after before than from into any all per".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "al", "es", "s", "e"):  # light stemming: approve/approval/approved -> approv
            if len(t) > 5 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        out.append(t)
    return out


@dataclass
class Chunk:
    source: str  # e.g. returns_policy.md#return-window
    title: str
    text: str


class DocumentStore:
    def __init__(self, docs_dir: Path, k1: float = 1.5, b: float = 0.75):
        self.chunks = self._load(docs_dir)
        self.k1, self.b = k1, b
        # Section headings are strong relevance signals, so their terms count twice (a light BM25F).
        self.tfs = [Counter(tokenize(c.title + " " + c.title + " " + c.text)) for c in self.chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @staticmethod
    def _load(docs_dir: Path) -> list[Chunk]:
        chunks = []
        for path in sorted(docs_dir.glob("*.md")):
            doc_title, section, buf = path.stem, None, []
            for line in path.read_text(encoding="utf-8").splitlines() + ["## __end__"]:
                if line.startswith("# "):
                    doc_title = line[2:].strip()
                elif line.startswith("## "):
                    if section and buf:
                        anchor = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
                        chunks.append(Chunk(f"{path.name}#{anchor}", f"{doc_title}: {section}", " ".join(buf).strip()))
                    section, buf = line[3:].strip(), []
                elif line.strip():
                    buf.append(line.strip())
        return chunks

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.tfs, self.lengths):
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if f:
                    norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                    score += self.idf[term] * norm
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"source": c.source, "title": c.title, "text": c.text, "score": round(s, 3)} for s, c in scored[:k]]
