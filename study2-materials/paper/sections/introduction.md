# Introduction

Large language models (LLMs) are increasingly deployed in educational contexts, with applications ranging from intelligent tutoring systems to automated assessment. A promising but underexplored application is the use of LLMs to simulate students for training human tutors and testing educational interventions. Simulated students could enable scalable, low-risk practice environments where novice tutors encounter diverse learner profiles before working with real students.

However, the validity of LLM-based student simulation rests on a critical assumption: that simulated students exhibit realistic learning behaviors, including authentic errors. When a simulated student makes a mistake, does that mistake reflect the kinds of errors real students make? Or do LLMs, optimized for correctness, produce errors that are qualitatively different from human misconceptions?

This question has significant practical implications. If LLM-simulated students make errors that don't match human misconceptions, tutors trained on these simulations may develop strategies that fail to transfer to real tutoring contexts. Similarly, intelligent tutoring systems tested on unrealistic simulated students may not perform as expected with actual learners.

## Prior Work

Recent surveys identify student simulation as an emerging application of LLMs in education, with approximately 77% of studies using simulated students for research and development, and 13% for teacher training (Author et al., 2024). Systems like TutorUp demonstrate the feasibility of using GPT-4 to simulate student engagement challenges for tutor practice (Author et al., 2025).

However, validation of simulation quality remains limited. Prior work notes that "it is unlikely for simulated students to behave like real ones" (Author et al., 2025), and identifies a core technical challenge: LLMs are inherently optimized for generating correct answers, making it difficult for them to exhibit "genuine confusion or typical struggles of real learners" (Author et al., 2024).

What is missing is systematic validation at the level of specific misconceptions. Mathematics education research has documented common student misconceptions—such as believing that multiplication always increases a number, or that larger denominators indicate larger fractions. These misconceptions produce predictable error patterns on diagnostic items. If LLM-simulated students exhibit these same patterns when they err, this would provide evidence for the validity of LLM-based student simulation.

## Research Questions

We investigate three questions about misconception alignment in LLM-simulated students:

**RQ1 (Validation)**: To what extent do LLM-simulated student errors align with documented human misconceptions? We operationalize alignment as the rate at which incorrect responses select a misconception-targeted distractor versus other wrong answers.

**RQ2 (Prompting)**: Does prompting strategy affect misconception alignment? We compare standard persona-based prompting ("act like a struggling student") with a novel "diagnose-then-simulate" approach that asks models to first identify common errors before exhibiting one.

**RQ3 (Capability)**: What is the relationship between model capability and misconception alignment? We hypothesize an inverted-U relationship: very weak models may lack sufficient reasoning to produce coherent misconception-based errors, while very strong models rarely err at all.

## Contributions

This paper makes three contributions:

1. We introduce a methodology for validating LLM student simulation against documented human misconceptions, using target distractor analysis on diagnostic mathematics items.

2. We provide empirical evidence on misconception alignment across five LLMs spanning a wide capability range (7-95% on GSM8K benchmark).

3. We evaluate a novel "diagnose-then-simulate" prompting strategy that leverages LLMs' metacognitive capabilities to improve simulation authenticity.
