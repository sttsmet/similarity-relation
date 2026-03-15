from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthorRecord:
    author_id: str
    name: str


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    year: int | None = None
    venue: str | None = None
    authors: list[AuthorRecord] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PaperRecord":
        authors_raw = row.get("authors", []) or []
        authors: list[AuthorRecord] = []
        for item in authors_raw:
            if isinstance(item, dict):
                authors.append(
                    AuthorRecord(
                        author_id=str(item.get("author_id") or item.get("id") or item.get("name")),
                        name=str(item.get("name") or item.get("author_id") or item.get("id")),
                    )
                )
            else:
                authors.append(AuthorRecord(author_id=str(item), name=str(item)))

        refs = [str(x) for x in (row.get("references", []) or []) if x]
        kws = [str(x) for x in (row.get("keywords", []) or []) if x]
        year = row.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None

        return cls(
            paper_id=str(row.get("paper_id") or row.get("id")),
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            year=year,
            venue=row.get("venue"),
            authors=authors,
            references=refs,
            keywords=kws,
        )
