
flowchart TD
  %% ── Initial pipeline ──
  subgraph INIT["Current UM Pipeline for PDFs"]
    A([Input PDF]) --> B[PDF to multi-page TIFF]
    B --> C[Shard TIFF into pages]
    C --> D[Orientation correction<br/> **per page**]
    D --> E[Tesseract OCR<br/> **per page**]
    E --> F([Page TSV & HOCR<br/>])
    F --> G[Re-assemble TSV & HOCR document level]
    G --> G1([Document-level HOCR])
    G1 --> H[Embed HOCR text layer<br/>on TIFF]
    H --> I([Searchable PDF])
    G --> G2([Document-level TSV])
    G2 --> J[Generate feature vectors<br/>from TSV]
    J --> J1[[Information Extraction & Auth Matching]]
    I --> Z((Milan))
  end

  %% ── New pipeline ──
  subgraph NEW["New UM Pipeline for PDFs"]
    K([Input PDF]) --> L[Azure Document Intelligence - OCR]
    L --> M([Searchable PDF])
    M --> Z1((Milan))
    L --> N([Document-level TSV])
    N --> O[Generate feature vectors<br/>from TSV]
    O --> J2[[Information Extraction & Auth Matching]]
  end

  %% dashed migration arrow — connects the two boxes
  INIT -. **migrate** .-> NEW

  %% (optional) keep the dashed look consistent
  linkStyle 0 stroke-dasharray: 5 5,stroke-width:2px

  %% define a class once
  classDef artifact fill:#F18F01,stroke:#7A4E00,color:#ffffff,stroke-width:2px;
  %% attach it to any nodes you want coloured
  class A,F,I,G1,G2,K,M,N artifact
