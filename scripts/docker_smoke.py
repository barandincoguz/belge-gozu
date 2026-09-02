"""Network kullanmadan üretim uygulama fabrikasını ayağa kaldıran Docker smoke sunucusu."""

import hashlib
from datetime import UTC, datetime

import pandas as pd
import uvicorn
from PIL import Image

from belge_gozu.answer.base import Answer
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings
from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.manifest import (
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
    IndexManifest,
    RenderConfig,
    corpus_checksum,
    write_manifest,
)
from belge_gozu.index.store import PackedIndex


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text=f"smoke: {question}", citations=[pages[0].page_id])


def build_smoke_index(settings: Settings, encoder: FakeEncoder) -> None:
    index_dir = settings.index_dir
    if index_dir.exists() and any(index_dir.iterdir()):
        raise RuntimeError(f"smoke testi dolu indeks dizininin üstüne yazmaz: {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    page_ids: list[str] = []
    records: list[dict[str, object]] = []
    texts: list[str] = []
    for number, title in enumerate(("MEDENİ KANUN", "İŞ KANUNU", "CEZA KANUNU"), start=1):
        page_id = f"smoke:{number}"
        relative_image = f"images/smoke/{number:04d}.webp"
        image_path = settings.data_dir / relative_image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (32, 32), (number * 40, 10, 10))
        image.save(image_path, format="WEBP")
        images.append(image)
        page_ids.append(page_id)
        texts.append(f"{title}\nSmoke testi için yerel ve sahte metin.")
        records.append(
            {
                "page_id": page_id,
                "doc_id": "smoke",
                "doc_name": title,
                "doc_type": "kanun",
                "source_url": "https://example.invalid/smoke",
                "page_no": number,
                "image_path": relative_image,
            }
        )

    meta = pd.DataFrame.from_records(records)
    index = PackedIndex.build(page_ids, encoder.encode_pages(images))
    index.save(index_dir)
    meta.to_parquet(index_dir / "meta.parquet", index=False)
    pd.DataFrame({"page_id": page_ids, "text": texts}).to_parquet(
        index_dir / "page_texts.parquet", index=False
    )
    write_manifest(
        index_dir,
        IndexManifest(
            model_name=settings.retriever_model,
            model_revision="docker-smoke",
            engine_versions={"smoke": "1"},
            query_format=TRAIN_COMPAT_V1,
            doc_prompt_sha256=hashlib.sha256(TRAIN_COMPAT_DOC_PROMPT.encode()).hexdigest(),
            quantization="sign-1bit",
            mask_policy="drop-padding",
            render=RenderConfig(),
            corpus_checksum=corpus_checksum(index_dir),
            n_pages=len(page_ids),
            n_tokens=int(index.offsets[-1]),
            built_at=datetime.now(UTC).isoformat(),
            git_commit="docker-smoke",
        ),
    )


def main() -> None:
    settings = Settings(min_score_threshold=-1e9)
    encoder = FakeEncoder()
    build_smoke_index(settings, encoder)
    application = create_app(settings=settings, encoder=encoder, answerer=StubAnswerer())
    uvicorn.run(application, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
