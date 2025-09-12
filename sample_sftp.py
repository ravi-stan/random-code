@pytest.fixture
def research_findings() -> ResearchFindings:
    """
    A simple, typical ResearchFindings instance for 'happy path' tests.
    """
    c1 = Citation(tool_name="SearchToolA", document_id="doc-123")
    c2 = Citation(tool_name="IndexerB", document_id="doc-456")

    b1 = BulletPoint(text="Key finding one", citations=[c1])
    b2 = BulletPoint(text="Key finding two", citations=[c2])

    return ResearchFindings(
        bullet_points=[b1, b2],
        no_bullet_points_found=False,
        error_description=""
    )
