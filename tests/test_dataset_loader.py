import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_absa.dataset_loader import load_semeval_xml, validate_annotations


def test_load_semeval_xml_preserves_aspect_offsets(tmp_path):
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <sentences>
      <sentence id=\"1\">
        <text>The battery is excellent.</text>
        <aspectTerms>
          <aspectTerm term=\"battery\" polarity=\"positive\" from=\"4\" to=\"11\"/>
        </aspectTerms>
      </sentence>
    </sentences>
    """
    source = tmp_path / "sample.xml"
    source.write_text(xml, encoding="utf-8")

    sentences, aspects = load_semeval_xml(source)
    validation = validate_annotations(sentences, aspects)

    assert len(sentences) == 1
    assert len(aspects) == 1
    assert aspects.loc[0, "aspect_term"] == "battery"
    assert bool(aspects.loc[0, "offset_matches"])
    assert validation["is_valid"] is True
