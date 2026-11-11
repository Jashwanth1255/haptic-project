# Verdict: "A Survey of Tactile Virtualization: Taxonomy, Analysis, Evaluation, and Future Directions"

File reviewed: `Final_Draft_Paper_Rewritten.docx.pdf` (16 pages, about 7,700 words). Checked on 2026-11-09.

## Overall verdict: not ready to submit, but close on substance

| Area | Verdict |
|---|---|
| References | **Pass.** All 17 exist and their details are essentially correct. A few need small corrections (listed below). |
| Structure | **Pass.** Clear taxonomy, six tables, and the sections are referred to in the text. |
| Factual accuracy | **Needs fixes.** One wrong number, one wrong sensor, one wrong author name and one false claim about our own work. |
| Language | **Fail.** About 25 garbled or broken sentences, and five reviewer notes left in the body. |

The previous draft's worst problems are fixed: there are no `{{...}}` tags, there is a References section, and every citation is numbered. What remains is about two to three days of careful editing. Do not submit it as it is. The garbled sentences look like the output of a paraphrasing tool, which plagiarism checkers and reviewers both flag.

## 1. References: all 17 verified

Each one was checked against Crossref, arXiv or the publisher's listing.

| # | Reference | Found | Correction needed |
|---|---|---|---|
| 1 | Li et al., Psychology & Marketing 41(8), 1748–1768, 2024 | Yes (DOI 10.1002/mar.22008) | None |
| 2 | Lim & Yap, arXiv:2411.05161 | Yes | Published in Proc. 6th AsiaHaptics 2024, so cite the proceedings |
| 3 | Braune, Martin & Le, CDATP 2(1), 80–90, 2021 | Yes | Add DOI 10.25367/cdatp.2021.2.p80-90. **The text calls it "Le et al."** (see 3.1) |
| 4 | Eguchi et al., Scientific Data 13, 756, 2026 | Yes | Authors: M. Eguchi, T. Hayase, Y. Hiroi, T. Hiraki; add issue no. 1 |
| 5 | Racat, Reynolds & Obal, Int. J. Consumer Studies 50(2), e70189, 2026 | Yes (via Crossref) | None |
| 6 | Heravi et al., IEEE Trans. Haptics 17(4), 705–716, 2024 | Yes | None |
| 7 | Cao et al., Robotics and Autonomous Systems 176, 104688, 2024 | Yes | Add DOI 10.1016/j.robot.2024.104688; full authors G. Cao, J. Jiang, D. Bollegala, M. Li, S. Luo |
| 8 | Yao et al., Science Advances 10(42), eadq9575, 2024 | Yes | None |
| 9 | Zhang et al., TexSenseGAN, arXiv:2407.11467 | Yes | arXiv says "To be published in IEEE Transactions on Haptics"; cite that version if it is out |
| 10 | Li & Seifi, Sound2Hap, arXiv:2601.12245, 2026 | Yes | None |
| 11 | Tu, Kwan & Yick, Materials 17(20), 5009, 2024 | Yes | None |
| 12 | Majumdar et al., Arch. Comput. Methods Eng. | Yes (via Crossref) | Add vol. 33, no. 2, pp. 2205–2227; first published online 2025. Authors: A. Majumdar, R. Bhattacharyya, V. S. Yadav |
| 13 | Yu & Liu, Interdisciplinary Materials 4, 235–248, 2025 | Yes | Add issue no. 2 |
| 14 | Lee et al., Scientific Reports 15, 42345, 2025 | Yes | None |
| 15 | Zandonati et al., arXiv:2305.00596, 2023 | Yes | None |
| 16 | Park, Ahn & Lee, Applied Sciences 14(10), 4011, 2024 | Yes | None |
| 17 | Rahman et al., arXiv:2412.01555, 2024 | Yes | None |

These are named in the text but have no reference of their own; they are cited only through other papers: TeslaTouch (Bau et al.), Meyer et al., Altinsoy and Merchel, the LMT and Proton datasets, KTH-TIPS2b, FMD, CLASSNet, BioTac, and KES/FTT. Reviewers expect the original paper to be cited. Either add those references or drop the names.

## 2. Five reviewer notes left in the text: blocker

Each `[VERIFY: ...]` note must be removed, and the sentence fixed:

| Page | Note | Resolution |
|---|---|---|
| 5 | 160 recordings per surface vs 80 conditions | **Resolved.** The dataset paper states 8 directions × 5 speeds × 2 forces × 2 repetitions = 160. Write that and delete the note. |
| 6 | Retrieval family cited to Heravi et al. [6] | Heravi's model is generative, so cite Rahman et al. [17] for nearest-neighbour retrieval instead. |
| 7 | TexSenseGAN training data | The TexSenseGAN abstract says only "an open dataset of tactile vibration data". Check the full paper for "14 classes of LMT108" before keeping that detail. |
| 7, 8 | Cao et al.: 76.36% or 83.06% | **The published abstract says 83.06%** for materials never touched before. Use 83.06% in the text and in Table III, unless the paper itself reports 76.36% for a different measure (for example the harmonic mean); if so, name that measure. |

## 3. Factual errors

1. **"Le et al." is the wrong name (6 places: pp. 2, 5 ×3, 6, 13).** Reference [3] is by Braune, Martin and Le, so it should be "Braune et al.". This is the same name mix-up the earlier draft had.
2. **Wrong accelerometer (Tables II and V).** The Cluster dataset uses an STMicroelectronics IIS3DWB 3-axis MEMS accelerometer, not an MPU6050.
3. **False claim about our own work (Section VII, p. 14).** "Sliding a sensor on to these fabrics under constant speed and pressure is what we did here" is untrue: the team used the public Cluster recordings and recorded nothing. Also, the filter is a 20–1000 Hz band-pass, not a high-pass.
4. **Section VII ignores our own results.** It proposes retrieving stored descriptors as the solution. Our Phase I found that the four descriptors do not separate the 16 fabrics, and that image-based retrieval does not beat the average descriptor. The section should say so, or at least not claim more than we found.
5. **Cluster dataset year (Table V).** Table V gives 2025, but it was on arXiv in 2024 and published in Scientific Data in 2026. Use 2026 to match reference [4].
6. **"Friction" listed as a Cluster modality (Table V).** Friction is a single coefficient per material, measured with a separate device; it is not a recorded signal. Say "friction coefficient".
7. **Unchecked statistic (p. 1).** "$2.2 trillion … 60% of total e-commerce in 2023" is cited to [1]. Check that [1] actually states it, or cite the original market report.

## 4. Language: needs a full copy-edit

Examples, with pages:

- **Garbled words:** "electromagnetic haptuuuator" (p. 3), "eguchi CORCBYE et al." (p. 4), "deformationFlow", "mechanoreformulated" (p. 4), "AnANNOTHERMORE DIRECTIONAL VARIATION IS ANISOTROPY", "frictionalonaliaal" (p. 4), "themifrubbers" (p. 13), "audio-hapticREADBY" (p. 13).
- **Cut-off or broken sentences:**
  - "cast iron or walnut provide**." (p. 13)
  - "According to Racat et al ** 1," (p. 13)
  - "According to for, Cao et al." (p. 14)
  - "According to for, Lee et al." (p. 13)
  - "moving m Moving a representation" (p. 14)
  - "maintain maintaining sustaining" (p. 13)
- **Empty citations:** "Cosine similarity [9;] and Wasserstein distance [;]" (p. 11).
- **Doubled words:** "sliding sliding speed" (p. 3), "map of of" (p. 4).
- **Broken equation (p. 4):** "G (s, u (t)) d=1." should read d = g(s, u(t)), numbered (1).
- **Sentences without meaning:**
  - "Stiffness, which is an opposite term to compliance refers to be compliant" (p. 4)
  - "Camps are optimized for various purposes" (p. 7)
  - "Avoids cost of Section 3 and section IV" (p. 14)
  - "not made by currents s survey" (p. 3)
  - The abstract's "can't assure an artificial sound wave produced by them to be realistically accurate while exploring"
- **Merged headings (Section VI, pp. 13–14):** items 4, 5 and 6 have their headings stuck to the end of the previous paragraph ("...adjustment.Label Granularity And Many-To-One Collapse 4)."). Put each heading on its own line.

## 5. Smaller points

- **Authors:** five names are listed, with the guide first. Check the order with Ms. Laxmi Deepthi (the guide is usually listed last) and use full names.
- **Figure 1:** check that the timeline matches the corrected text (Braune et al., Heravi 2024).
- **Abstract and conclusion:** both should be rewritten in plain sentences after the edits above.

## What to do, in order

1. Remove the five `[VERIFY]` notes, using the resolutions in Section 2.
2. Fix the seven factual errors in Section 3.
3. Copy-edit every sentence listed in Section 4, then read the whole paper aloud once to catch the rest.
4. Apply the reference corrections in Section 1, and add primary citations for the systems named in passing.
5. Run a plagiarism check before submitting.

## Sources used for verification

- Crossref records for DOIs 10.1002/mar.22008, 10.1038/s41597-026-06760-z, 10.1111/ijcs.70189, 10.1109/TOH.2024.3382258, 10.1126/sciadv.adq9575, 10.3390/ma17205009, 10.1007/s11831-025-10366-w, 10.1038/s41598-025-26294-5, 10.3390/app14104011
- arXiv: [2411.05161](https://arxiv.org/abs/2411.05161), [2407.11467](https://arxiv.org/abs/2407.11467), [2601.12245](https://arxiv.org/abs/2601.12245), [2305.00596](https://arxiv.org/abs/2305.00596), [2412.01555](https://arxiv.org/abs/2412.01555), [2407.16206](https://arxiv.org/abs/2407.16206) (dataset details from the v4 HTML version)
- [Braune et al. on DOAJ](https://doaj.org/article/5bfc6938e4fe45b6b67ad3e4d923d99b)
- [Cao et al. project page](https://sites.google.com/view/multimodalzsl)
- [Yu & Liu, Interdisciplinary Materials](https://journal.hep.com.cn/intermat/EN/10.1002/idm2.12233)
- [Li et al., University of Edinburgh record](https://www.research.ed.ac.uk/en/publications/vibrotactile-feedback-in-m-commerce-stimulating-perceived-control/)
