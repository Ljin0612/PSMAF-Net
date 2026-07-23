from psmaf_net.models import PSMAFNetPlaceholder


def test_placeholder_description_mentions_psmaf():
    description = PSMAFNetPlaceholder().describe()
    assert "PSMAF-Net" in description
    assert "UNIV" in description
