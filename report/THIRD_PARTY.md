The report uses the official [CVPR author kit](https://github.com/cvpr-org/author-kit.git),
pinned at commit `291758547e923160eb4d37079b7b9f0dfce82355` (CVPR 2026 template).

`cvpr.sty`, `ieeenat_fullname.bst` and `preamble.tex` are copied unchanged from
that repository. The document follows its camera-ready setup, two-column
letter-paper class, CVPR-blue links and bibliography style. Project content,
figures and the provided author names replace the sample content; no layout
or font-size overrides are applied to the upstream style.

The unmodified source template, preamble, style, bibliography style and README
are retained in `third_party/cvpr_author_kit`, together with commit/file hashes.
Original authorship and notices are preserved. The upstream repository does
not include a separate LICENSE file at this revision.

The DMD2/f-distill training components follow NVIDIA FastGen's SD1.5 recipe
at commit `3943fd5cfb2a8266ec53e7e541e4b8d2e5b1e94d` (Apache-2.0).
Pinned upstream files, original copyright headers, LICENSE and source hashes
are included in `third_party/fastgen_reference`. Local adaptations and source
differences are documented in `docs/RECENT_BASELINES.md`.
