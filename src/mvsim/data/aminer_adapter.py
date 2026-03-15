from __future__ import annotations

from typing import Any

from mvsim.schemas import PaperRecord, AuthorRecord



def adapt_aminer_row(row: dict[str, Any]) -> PaperRecord:
    """
    Convert a generic AMiner / OAG style row into the internal PaperRecord schema.

    This adapter is intentionally permissive because exported field names may differ
    across AMiner dumps, OAG files, and custom API responses.
    """
    paper_id = row.get("paper_id") or row.get("id") or row.get("pub_id")
    title = row.get("title") or row.get("display_name") or ""
    abstract = row.get("abstract") or row.get("summary") or ""
    year = row.get("year") or row.get("publication_year")
    venue = row.get("venue") or row.get("journal") or row.get("conference")

    authors_raw = row.get("authors") or row.get("author") or row.get("author_list") or []
    authors: list[AuthorRecord] = []
    for item in authors_raw:
        if isinstance(item, dict):
            author_id = item.get("author_id") or item.get("id") or item.get("aid") or item.get("name")
            authors.append(AuthorRecord(author_id=str(author_id), name=str(item.get("name") or author_id)))
        else:
            authors.append(AuthorRecord(author_id=str(item), name=str(item)))

    refs_raw = row.get("references") or row.get("reference") or row.get("refs") or []
    references = [str(x) for x in refs_raw if x]

    kws_raw = row.get("keywords") or row.get("fields_of_study") or row.get("fos") or []
    keywords = [str(x) if not isinstance(x, dict) else str(x.get("name") or x.get("value") or x) for x in kws_raw]

    return PaperRecord(
        paper_id=str(paper_id),
        title=str(title),
        abstract=str(abstract),
        year=int(year) if year not in (None, "") else None,
        venue=str(venue) if venue is not None else None,
        authors=authors,
        references=references,
        keywords=keywords,
    )
