"""Self-contained UniDoc-Bench corpus downloader for the bench's ``pareto`` run.

Pulls real-world enterprise PDFs (and a few first-page images, to exercise
Docling's OCR path on more file types) from the ``Salesforce/UniDoc-Bench`` HF
dataset into a flat corpus directory. Owned by the bench so the ``pareto``
command runs without assuming the optimizer repo's ``scripts/`` are a sibling
on disk — a fresh redownload is fine.

Held-out QA is intentionally NOT downloaded: the pareto experiment scores
trials on the optimizer's own self-generated exam, not UniDoc's QA set.

Idempotent: files that already exist on disk are left untouched, and the HF
archive is cached after the first run.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger("agentic_autorag_bench.run")

_REPO_ID = "Salesforce/UniDoc-Bench"

# Pinned dataset revision. Left unpinned, every download resolves the repo's
# current head, so a corpus rebuilt later can quietly differ from the one the
# paper's Pareto experiment retrieved over. This sha is the revision the runs
# actually used, recovered from the local HuggingFace cache, which held exactly
# one snapshot for this dataset.
_REVISION = "2e22c592438d1e52c39bd769aab2901d596af746"

_DOMAIN_ARCHIVE = {
    "healthcare": "healthcare_pdfs.tar.gz",
    "legal": "legal_pdfs.tar.gz",
    "education": "education_pdfs.tar.gz",
    "crm": "crm_pdfs.tar.gz",
    "energy": "energy_pdfs.tar.gz",
    "construction": "construction_pdfs.tar.gz",
    "commerce_manufacturing": "commerce_manufacturing_pdfs.tar.gz",
    "finance": "finance_pdfs.tar.gz",
}


def _download_pdfs(domain: str, output_dir: Path, limit: int) -> int:
    """Extract up to ``limit`` PDFs from the domain archive as ``{domain}_{stem}.pdf``."""
    from huggingface_hub import hf_hub_download

    archive = _DOMAIN_ARCHIVE[domain]
    logger.info("UniDoc: fetching %s (cached after first run)...", archive)
    local_path = hf_hub_download(
        repo_id=_REPO_ID, filename=archive, repo_type="dataset", revision=_REVISION
    )

    extracted = 0
    with tarfile.open(local_path, "r:gz") as tar:
        for member in tar:
            if extracted >= limit:
                break
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            stem = Path(member.name).name
            if stem.startswith("._"):  # macOS AppleDouble resource fork
                continue
            dest = output_dir / f"{domain}_{stem}"
            if not dest.exists():
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                dest.write_bytes(fh.read())
            extracted += 1
    logger.info("UniDoc: %d PDF(s) in %s", extracted, output_dir)
    return extracted


def _download_images(domain: str, output_dir: Path, limit: int) -> int:
    """Download the first page image of up to ``limit`` documents as PNGs.

    Page images exercise Docling's OCR path — including them in the corpus is
    the point of using UniDoc, not an accident.
    """
    if limit <= 0:
        return 0
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    # HfFileSystem takes the revision inline as ``repo@sha``; the listing has to
    # be pinned too, or which documents get picked depends on the head listing.
    prefix = f"datasets/{_REPO_ID}@{_REVISION}/images/{domain}"
    doc_dirs = fs.ls(prefix, detail=False)

    downloaded = 0
    for doc_dir in doc_dirs:
        if downloaded >= limit:
            break
        doc_id = Path(doc_dir).name
        img_hf_path = f"{doc_dir}/{doc_id}_page_0001.png"
        # The listing paths carry the ``@sha``, so it has to come off here too or
        # the repo-relative filename keeps it and the download 404s.
        img_repo_path = img_hf_path.removeprefix(f"datasets/{_REPO_ID}@{_REVISION}/")
        dest = output_dir / f"{domain}_{doc_id}_page_0001.png"
        if not dest.exists():
            local = hf_hub_download(
                repo_id=_REPO_ID,
                filename=img_repo_path,
                repo_type="dataset",
                revision=_REVISION,
            )
            dest.write_bytes(Path(local).read_bytes())
        downloaded += 1
    logger.info("UniDoc: %d page image(s) in %s", downloaded, output_dir)
    return downloaded


def download_unidoc_corpus(
    output_dir: str | Path,
    *,
    domain: str = "healthcare",
    max_pdfs: int = 230,
    max_images: int = 20,
) -> tuple[int, int]:
    """Download a UniDoc-Bench subset into ``output_dir``. Returns ``(n_pdfs, n_images)``."""
    if domain not in _DOMAIN_ARCHIVE:
        raise ValueError(f"Unknown UniDoc domain {domain!r}; known: {sorted(_DOMAIN_ARCHIVE)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_pdfs = _download_pdfs(domain, output_dir, max_pdfs)
    n_images = _download_images(domain, output_dir, max_images)
    return n_pdfs, n_images
