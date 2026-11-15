# Portfolio results transcription

This is a manual transcription of page 19 of the supplied research portfolio,
visually verified against a rendered page. Source PDF SHA-256:
`9e9176e3680036047a101a67b8254907c77ba56e80767b39c22c8c5dc4a79c0b`.

| Model | Parameters shown | Top-1 |
|---|---:|---:|
| CliffordNet-Nano | 1.4M | 77.82% |
| CliffordNet-Lite | 2.6M | 79.05% |
| CliffordNet-32 | 4.8M | 81.42% |
| CliffordNet-64 | 8.6M | 82.46% |
| Ours-Nano (16-2 shifts) | 1.4M | 78.37% |
| Ours-Lite (18-5 shifts) | 2.4M | 79.92% |
| Ours-Lite (25-3 shifts) | 2.6M | 80.57% |
| Ours (32-5 shifts) | 4.3M | 81.45% |
| Ours (48-2 shifts) | 4.1M | 81.79% |
| Ours (52-3 shifts) | 5.3M | 82.46% |

The two 82.46% entries are distinct rows: the official 64-block baseline and
the proposed 52-block Roll & Shift model. No Geometric Shuffle result appears
in this portfolio version.
