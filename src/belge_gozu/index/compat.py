from pathlib import Path

from belge_gozu.index.manifest import IndexManifest, corpus_checksum


class IndexCompatibilityError(RuntimeError):
    pass


def check_compatibility(
    manifest: IndexManifest | None,
    *,
    model_name: str,
    model_revision: str | None,
    query_format_id: str,
    doc_prompt_sha256: str | None = None,
    index_dir: Path,
) -> list[str]:
    if manifest is None:
        # Final review IMPORTANT-3: eski metin `index write-manifest --legacy`
        # öneriyordu, ama o komut mask_policy="none" + query_format=cpe-0.3.18
        # yazar; aşağıdaki mask_policy kontrolü onu yine reddeder — çıkmaz sokak.
        # Gerçek çözüm indeksi yeniden inşa etmektir.
        return [
            "indekste manifest yok (v0 legacy?) — çözüm: `belge-gozu index build` ile "
            "indeksi yeniden inşa edin (manifest'i doğru değerlerle yazar). "
            "`belge-gozu index write-manifest --legacy` YALNIZCA teşhis içindir: "
            'mask_policy="none" + cpe-0.3.18 yazdığı için bu kontrolden geçmez.'
        ]
    problems: list[str] = []
    if manifest.model_name != model_name:
        problems.append(f"model_name: indeks={manifest.model_name} serve={model_name}")
    if model_revision and model_revision != "unknown" and manifest.model_revision != model_revision:
        problems.append(f"model_revision: indeks={manifest.model_revision} serve={model_revision}")
    if manifest.query_format.format_id != query_format_id:
        problems.append(
            f"query_format: indeks={manifest.query_format.format_id} serve={query_format_id}"
        )
    if (
        doc_prompt_sha256
        and doc_prompt_sha256 != "unknown"
        and manifest.doc_prompt_sha256 != doc_prompt_sha256
    ):
        problems.append(
            "doc_prompt_sha256: "
            f"indeks={manifest.doc_prompt_sha256[:12]} serve={doc_prompt_sha256[:12]}"
        )
    if manifest.mask_policy != "drop-padding":
        problems.append(f"mask_policy: indeks={manifest.mask_policy} (drop-padding bekleniyor)")
    live = corpus_checksum(index_dir)
    if manifest.corpus_checksum != live:
        problems.append("corpus_checksum: indeks manifest'i ile meta/page_ids uyuşmuyor")
    return problems
