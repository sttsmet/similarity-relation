from __future__ import annotations

from mvsim.graph.operators import build_citation_adjacency, bibliographic_coupling, cocitation
from mvsim.schemas import PaperRecord



def test_second_order_shapes() -> None:
    papers = [
        PaperRecord(paper_id="P1", title="a", abstract="", references=["P2"]),
        PaperRecord(paper_id="P2", title="b", abstract="", references=[]),
        PaperRecord(paper_id="P3", title="c", abstract="", references=["P2"]),
    ]
    A, _ = build_citation_adjacency(papers)
    B = bibliographic_coupling(A)
    C = cocitation(A)
    assert B.shape == (3, 3)
    assert C.shape == (3, 3)
