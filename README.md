# Violin Fingering Generation

Cool project that I worked on during Fall 2025 for CMSC838B: Differentiable Programming under Dr. Ming Lin @ UMD!
Access the full report here: https://www.overleaf.com/read/tkgcrfxptrzj#61e195
The PDF report is also included in this repo: https://github.com/Mimsqueeze/Violin-Fingering-Generation/blob/main/CMSC838B_Final_Report.pdf

# Abstract
In this work, we present a novel method for generating personalized violin finger-
ings for symbolic sheet music, given user preferences. Recent work has explored
this problem via statistical methods or BLSTM networks, but none have explored
the powerful transformer architecture — renowned for excelling at processing
sequential data. In this work we discuss a method for converting symbolic sheet
music (in MusicXML format) into a differentiable format compatible with trans-
former architecture. Then, we explore and iteratively refine different transformer
architectures to generate personalized fingerings, utilize LoRA + prefix tuning
for finetuning on preferences, and present qualitative results of our method work-
ing reasonably well in practice. However, we find that biggest limitation for any
method in solving this problem is the lack of a comprehensive violin fingering
dataset, so future directions should develop a pipeline to obtain that data, or use
self-supervision to take advantage of unlabeled data (as some related works do).
