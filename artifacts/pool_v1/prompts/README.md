# Paper prompt transcription

`figure8_system.txt` and `figure8_user.txt` transcribe Figure 8 in Appendix B.2 of the supplied paper (PDF page 13). The role labels `System` and `Input` are represented by the two filenames rather than repeated inside their contents.

The paper reports `temperature=0.2`, `top_p=0.9`, and `max_tokens=256`; these decoding settings are frozen in `../config.yaml`. It does not report a GPT-4o snapshot, access date, JSON serialization, or numeric boundaries for the three density labels.
